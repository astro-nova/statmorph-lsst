import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage import transform as T

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.stats import sigma_clipped_stats

from photutils.segmentation import detect_sources
from photutils.background import Background2D

##### Utility functions ##########
def grow_segmask(segmap, grow_sigma=1, area_norm=10):
    """ Grow the segmentation mask based on each labelled region's size """
    areas_arr = np.concatenate([[0],segmap.areas]).astype(float)
    segmap_areas = areas_arr[segmap.data]
    segmask = ndi.gaussian_filter(segmap_areas.astype(float)/area_norm, sigma=grow_sigma, truncate=5) > 0.05
    return segmask



##### Loading and preparing raw data ##########
# 1. Load in the HLA FITS file, calculate the exposure time map and header info
# 2. Estimat the background; if needed, fit a 2D background and subtract it
# 3. Create cutouts
###############################################
def load_raw_data(row, ignore_ctx=True, path='../data'):

    """ Load in the HLA/MAST reduced image, and make a slice object to make cutouts later on based on the galaxy of interest. 
    Also calculate ZP and pixel scale from the header, and estimate the error array where needed (and possible).
    Does not actually return the cutout, since the full tile will be used later in background estimation.

    Args:
        filename: name of the MAST/HLA file to load in
        ra, dec: RA and Dec of the galaxy center
        rpet: the Petrosian radius of the galaxy in arcsec
        frac_r: the cutout will be frac_r * rpet (unless the image is too small); default=3
        ignore_ctx: if True, only use CTX extension to figure out detector area, leaving bad pixels unmasked
        path: path to the data folder if running not from root

    Returns:
        img, mask, err: full tiles corresponding to the image, mask, and 1sigma error
        header: fits Header with relevant information about the image
        cutout: a 2D slice object that can be used to make a cutout centered on the object
    """

    ####### 1. Open the image file ##################
    # Open the file
    f_sci = fits.open(f'{path}/raw/{row.filename}_drz.fits')
    img = f_sci['SCI'].data
    header = f_sci['SCI'].header

    ####### 2. Header information  ##################
    # Get pixelscale from WCS
    wcs = WCS(f_sci['SCI'].header)
    pxscale = wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value

    # Add zeropoint and pixel scale informatin to the header
    # This is sometimes in first, sometimes in the second header
    try:
        header['ZP'] = -2.5*np.log10(header['PHOTFLAM']) - 21.10 - 5*np.log10(header['PHOTPLAM']) + 18.692
    except:
        header['ZP'] = -2.5*np.log10(f_sci[0].header['PHOTFLAM']) - 21.10 - 5*np.log10(f_sci[0].header['PHOTPLAM']) + 18.692
    header['PXSCALE'] = pxscale

    ####### 3. Create masks ##################
    # CTX encodes the information about masked pixels (outside the detector; cosmic rays, etc)
    # Just by looking at it, it looks like there are two CTX bits for each detector on ACS
    # And "good" pxiels are those marked with max CTX value for detector 2, and max/2 for detector 1
    # Sometimes CTX is not what it should be - can ignore CTX in this case and just use detector area as a mask
    if ('CTX' in f_sci) and (f_sci['CTX'].data is not None):
        if f_sci['CTX'].data.ndim == 2:
            ctx = f_sci['CTX'].data
            if not ignore_ctx:
                ctx1 = np.max(ctx.data)
                ctx2 = ctx1/2
                mask = (ctx != ctx1) & (ctx != ctx2)

                # Smooth over tiny masked regions from CRs - interpolation deals with them well 
                mask_smoothed = ndi.gaussian_filter(mask.astype(float), sigma=3) > 0.7
                mask = mask_smoothed & mask
            else: 
                mask = ctx==0
        else:
            ctx = np.sum(f_sci['CTX'].data, axis=0)
            mask = ctx==0
    else:
        ctx = f_sci['WHT'].data
        mask = ctx == 0
    # Also for bg estimation, return a chip mask (i.e. only regions outside the detector)
    chipmask = ctx == 0
    
    ####### 4. Exposure time maps ##################
    expext = 'EXP' # Most of the time exptime map is called EXP but in some images it's saved in WHT
    if ('mos' in row.filename) or ('hlsp' in row.filename) or ('skycell' in row.filename) or (row.galaxy == 'ic860'):
        expext = 'WHT'     
    exptime = f_sci[expext].data
    # Multiply by gain if the units aren't in electrons
    gain = 1 if 'electrons' in header['BUNIT'].lower() else f_sci[0].header['CCDGAIN']
    exptime *= gain
    # # Smooth over inidividual masked pixels - we can use interpolated data, makes statmorph work better
    exptime = ndi.maximum_filter(exptime, size=5)
    exptime[chipmask] = 0

    ####### 5. Load the PSF ##################
    psf = np.load(f'{path}/psf/{row.detector}_4x.npy')
    psf[0,:] = psf[-1,:]
    new_psf_size = int(psf.shape[0]/4)
    new_psf_size = new_psf_size if new_psf_size%2==1 else new_psf_size+1
    psf = T.resize_local_mean(psf, (new_psf_size, new_psf_size), grid_mode=False)
    psf = psf/np.sum(psf)

    return img, mask, chipmask, exptime, psf, header

def estimate_bg(img, mask, chipmask, max_iter=30, thresh=0.05, fit_bg=True, masked_thresh=0.2, filter_size=3):
    """Estimates the large-scale background on a smoothed image iteratively 
    to ensure that all of the object flux is properly masked.
    First, background stats are estimated on a raw image. Then, we run source detection
    to find all sources 1sigma above the background, and mask these. Then, we re-calculate
    the background stats. Continue this until the background sigma converges.
    Finally, use photutils Background2D to estimate the 2D background and subtract it.
    
    Args:
        img: NxN np.array with the full tile
        mask: NxN mask showing bad pixels etc
        chipmask: similar to mask, where 1 shows the detector area and 0 blank fields
        max_iter: the number of times we recalculate mask and background before convergence (Default: 10)
        thresh: the convergence threshold in % difference between background error
        fit_background (bool): if True, also run the 2D background fit (original images only)
        masked_thresh (float): minimum number of pixels that should be unmasked for BG estimation
        filter_size (int): size of the filter to used in smoothing the image for source detection
    Returns:
        img: background-subtracted image
        median: background median
        std: background standard deviation
        bg_mask: mask (including the detected sources) used to calculate BG
        bg: NxN background image
    """

    img_conv = ndi.gaussian_filter(img, sigma=filter_size, truncate=2)

    # First pass at getting the background RMS
    mean, median, std = sigma_clipped_stats(img, sigma=3.0, mask=mask)
    std_og = std 

    detect_area = 10
    # Recursively mask sources and re-calculate bg while we do not converge
    for i in range(max_iter):
        std_prev = std
    
        segmap = detect_sources(img_conv, 1*std+median, npixels=detect_area, mask=chipmask, connectivity=4)
        segmask = segmap.data > 0

        # Stop if there are fewer than 20% unmasked pixels
        masked_frac = np.sum(segmask | mask)/img.size
        if masked_frac > (1-masked_thresh):
            break
        
        mean, median, std = sigma_clipped_stats(img, sigma=3.0, mask=mask|segmask)

        err = np.abs(std-std_prev)/std_prev
        if err < thresh: break

    
    # Grow the mask edges by an amount relative to the segment size
    segmask = grow_segmask(segmap, grow_sigma=3, area_norm=1)
    bg_mask = mask|segmask

    if fit_bg:
        # Estimate 2D background
        box_size = int(img.shape[0]*0.01)
        bkg = Background2D(img, (box_size,box_size), filter_size=3, mask=bg_mask, coverage_mask=chipmask)

        fig, axs = plt.subplots(1,3,figsize=(15,5))
        axs[0].imshow(img, vmin=median-std, vmax=median+std, cmap='gray')
        axs[1].imshow(bkg.background, vmin=median-std, vmax=median+std, cmap='gray')
        axs[2].imshow(img-bkg.background, vmin=-std, vmax=std, cmap='gray')
        for ax in axs:
            ax.contour((segmask | chipmask)>0, colors=['C9'], levels=[0.9,1.1], linewidths=0.5)
        plt.show()

        answer = input("Subtract background? y | n\n")
        if answer == 'y': 
            img -= bkg.background
            std = bkg.background_rms_median
            bg_std_map = bkg.background_rms
        else:
            mean, median, std = sigma_clipped_stats(img, sigma=3.0, mask=bg_mask)
            img -= median
            bg_std_map = std*np.ones_like(img)
    else:
        mean, median, std = sigma_clipped_stats(img, sigma=3.0, mask=bg_mask)
        img -= median
        bg_std_map = std*np.ones_like(img)
    
    # Just return the normal non-iterative background if didn't reach convergence
    if i==max_iter-1: 
        print("WARNING! Background didn't converge. Returning ismple sigma-clip background")
        std = std_og
    
    plt.close()
    return img, median, std, bg_mask, bg_std_map

def make_cutout_slice(row, img_shape, header, frac_r=3):
    """ Create a slice object to make cutouts later on based on the galaxy ra, dec, and Rpet. The cutout is 
    made within frac_r * rpet (unless the image is too small); default=3"""

    # Get pixelscale from WCS
    wcs = WCS(header)

    # Create a cutout
    xc, yc = wcs.all_world2pix( np.array([row.ra, row.dec])[np.newaxis,:], 1)[0]
    xc, yc = int(xc+0.5), int(yc+0.5)
    size = int(row.rpet*frac_r/header['pxscale']+0.5)

    # If the image is smaller than the cutout size, instead make the largest cutout possible
    edges = [xc-size, img_shape[1]-(xc+size), yc-size, img_shape[0]-(yc+size)]
    size_lim = np.min(edges)
    if size_lim < 0: size += size_lim

    # Final cutout size
    cutout_slice = slice(yc-size, yc+size), slice(xc-size, xc+size)
    return cutout_slice

def save_cutout(row, img, exp, mask, psf, bg_std, bg_std_map, header):
    """Get the background-subtracted image, calculate error, and save the cutoutt as a FITS file."""
    cutout_slice = make_cutout_slice(row, img.shape, header)

    # Calculate error map
    counts = img*exp + (bg_std_map*exp)**2
    err = np.sqrt(counts)/exp
    mask = mask | np.isnan(err) | (exp==0)

    # Background RMS in header
    header['BGSD'] = bg_std
    # Calculate surface brightness limit 
    sblim = -2.5*np.log10(bg_std/header['PXSCALE']**2) + header['ZP']
    header['SBLIM'] = sblim
    header['SBLIM0'] = sblim - 7.5*np.log10(row.z+1)  # SB limit corrected to z=0

    # Create cutouts
    img_cutout = img[cutout_slice]
    err_cutout = err[cutout_slice]
    mask_cutout = mask[cutout_slice]
    bg_cutout = bg_std_map[cutout_slice]

    # Create a new FITS file and save the data
    hdu_img = fits.PrimaryHDU(img_cutout, header=header)
    hdu_err = fits.ImageHDU(err_cutout, name='ERR')
    hdu_mask = fits.ImageHDU(mask_cutout.astype(np.uint8), name='MASK')
    hdu_bg = fits.ImageHDU(bg_cutout, name='BG_RMS')
    hdu_psf = fits.ImageHDU(psf, name='PSF')

    hdul = fits.HDUList([hdu_img, hdu_err, hdu_mask, hdu_psf, hdu_bg])
    hdul.writeto(f'../data/cutouts/{row.galaxy}.fits', overwrite=True)
    

##### Augmentations ##########
# 1. Load the image cutout
# 2. Rescale it to the desired physical resolution
# 3. Re-estimate the background / surface brightness limit 
# 4. Add noise to match the desired SB limit
# 5. Save the augmented image
###############################################