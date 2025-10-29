import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage import transform as T

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u

from utils import estimate_bg


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

    f_sci.close()

    return img, mask, chipmask, exptime, psf, header



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

def save_cutout(row, img, exp, mask, psf, bg_std, bg_std_map, header, path='../data'):
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
    exp_cutout = exp[cutout_slice]

    # Create a new FITS file and save the data
    hdu_img = fits.PrimaryHDU(img_cutout, header=header)
    hdu_err = fits.ImageHDU(err_cutout, name='ERR')
    hdu_mask = fits.ImageHDU(mask_cutout.astype(np.uint8), name='MASK')
    hdu_bg = fits.ImageHDU(bg_cutout, name='BG_RMS')
    hdu_exp = fits.ImageHDU(exp_cutout, name='EXP')
    hdu_psf = fits.ImageHDU(psf, name='PSF')

    hdul = fits.HDUList([hdu_img, hdu_err, hdu_mask, hdu_psf, hdu_bg, hdu_exp])
    hdul.writeto(f'{path}/cutouts/{row.galaxy}.fits', overwrite=True)

    # Make a diagnostic figure of the cutouts
    fig, axs = plt.subplots(1, 4, figsize=(18,4))
    axs[0].imshow(-2.5*np.log10(np.abs(img_cutout)/header['PXSCALE']**2) + header['ZP'], vmin=18, vmax=28, cmap='gray_r')
    axs[1].imshow(err_cutout, vmin=np.nanquantile(err_cutout, 0.01), vmax=np.nanquantile(err_cutout, 0.95), cmap='gray')
    axs[2].imshow(mask_cutout, cmap='gray_r')
    try:
        axs[1].imshow(bg_cutout, vmin=np.nanquantile(bg_cutout, 0.01), vmax=np.nanquantile(bg_cutout, 0.95), cmap='gray')
    except:
        axs[1].imshow(bg_cutout, cmap='gray')
    axs[0].set_title('Image (mag/arcsec^2)')
    axs[1].set_title('1sigma Error')
    axs[2].set_title('Mask')
    axs[3].set_title('BG RMS')
    plt.savefig(f'{path}/diag_figs/cutouts/{row.galaxy}.png', dpi=150)
    
def reduce_galaxy(row, ignore_ctx=True, path='../data'):
    """ Full data reduction pipeline for a single galaxy row from the input catalog."""
    # 1. Load raw data
    img, mask, chipmask, exptime, psf, header = load_raw_data(row, ignore_ctx=ignore_ctx, path=path)

    # 2. Estimate and subtract background
    img_bgsub, bg_median, bg_std, bg_mask, bg_std_map = estimate_bg(img, mask, chipmask, fit_bg=True)

    # 3. Save cutout
    save_cutout(row, img_bgsub, exptime, mask, psf, bg_std, bg_std_map, header, path=path)


# Run `reduce_galaxy` for a partiular galaxy from command line
if __name__ == '__main__':

    # Inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("galaxy", type=str, help="Galaxy name")
    parser.add_argument('--ignore_ctx', help='Mask non-zero CTX pixels?', action="store_true")
    args = parser.parse_args()

    catalog = pd.read_csv('../data/catalogs/sample.csv')
    row = catalog[catalog.galaxy==args.galaxy].iloc[0]
    reduce_galaxy(row, ignore_ctx=args.ignore_ctx, path='../data')




