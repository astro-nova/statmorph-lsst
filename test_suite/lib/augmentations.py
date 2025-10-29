# imports
import os
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
from astropy.convolution import convolve_fft, Moffat2DKernel, convolve
import skimage.transform as T
from .utils import estimate_bg
import matplotlib.pyplot as plt

##### Augmentations ##########
# 1. Load the image cutout
# 2. Rescale it to the desired physical resolution
# 3. Re-estimate the background / surface brightness limit 
# 4. Add noise to match the desired SB limit
# 5. Save the augmented image
###############################################

def get_psf(fwhm, beta=2):
    """Generate a Moffat PSF array with a given FWHM and beta parameter."""
    R = fwhm / (2*np.sqrt( np.power(2, 1/beta) - 1))
    psf = Moffat2DKernel(R, beta)
    return psf.array

def rescale_image(img, counts, exp, sky, mask, pxscale_new, pxscale_og, psf_og, psf_size_px=2, imsize_min=20):
    """Rescale the image to a new physical scale using T.downscale_local_mean.
    Args:
        img (np.ndarray): The input image in image units (e.g., ADU/s)
        counts (np.ndarray): The input image to be rescaled in electrons
        exp (np.ndarray): Exposure time map * gain to convert to image units
        sky (np.ndarray): The sky background map in electrons
        mask (np.ndarray): The mask array corresponding to the input image.
        pxscale_new (float): The desired new pixel scale in pc/px.
        pxscale_og (float): The original pixel scale in pc/px.
        psf_og (np.ndarray): The original PSF array.
        psf_size_px (int): The size of the new Moffat PSF in pixels (default is 2).
        imsize_min (int): Minimum size of the image to attempt rescaling.
    Returns:
        img_rescaled (np.ndarray): The rescaled image in image units.
        err_rescaled (np.ndarray): The rescaled error map in image units.
        psf_rescaled (np.ndarray): The rescaled PSF array.
        mask_rescaled (np.ndarray): The rescaled mask array.
        pxscale_new (float): The actual new pixel scale in pc/px after rescaling.
    """

    factor = int(pxscale_new/pxscale_og + 0.5)
    pxscale_new = pxscale_og * factor

    # If the image is too small to be rescaled, return None
    if counts.shape[0]/factor < imsize_min or counts.shape[1]/factor < imsize_min:
        print(f'Image too small to be rescaled to {pxscale_new} pc/px; continuing')
        return None, None, None, None, None

    # Generate a Moffat PSF that would be 2px in the new image. 
    # Figure out the new effective PSF, PSF_old (conv) PSF_new
    psf = get_psf(psf_size_px*factor)
    psf /= np.sum(psf)
    psf_new = convolve(psf_og, psf)
    psf_new /= np.sum(psf_new)

    # Convolve the image and convert to electrons.
    # Not convolving counts directly because not all pixels have the same exptime, which messes up convolution
    img_conv = convolve_fft(img.copy(), psf, mask=mask, nan_treatment='fill', fill_value=0)
    img_conv[mask] = img[mask] # Overwrite masked pixels from the original image
    counts_conv = img_conv * exp + sky
    counts_conv[counts_conv<=1] = 1 # Avoid zeros for Poisson sampling
    counts_conv[np.isnan(counts_conv)] = 1 # Avoid NaNs for Poisson sampling
    counts_conv[np.isinf(counts_conv)] = 1 # Avoid NaNs for Poisson sampling
    # Resample the counts array to imitate Poisson noise after convolution
    counts_conv = np.random.poisson(counts_conv)

    # Fix the mask as some areas might be nan after convolution
    mask = mask | np.isnan(counts_conv)

    # Downscale the image
    counts_rescaled = T.downscale_local_mean(counts_conv, factor) * factor**2
    sky_rescaled = T.downscale_local_mean(sky, factor) * factor**2
    exp_rescaled = T.downscale_local_mean(exp, factor)
    mask_rescaled = T.downscale_local_mean(mask.astype(float), factor) 
    mask_rescaled = mask_rescaled>0.5

    # Recalculate the error
    err_rescaled = np.sqrt(counts_rescaled)/exp_rescaled
    # Convert back to img units
    img_rescaled = (counts_rescaled - sky_rescaled) / exp_rescaled

    # Downscale the PSF
    new_psf_size = int(psf.shape[0]/factor)
    new_psf_size = new_psf_size if new_psf_size%2==1 else new_psf_size+1
    psf_rescaled = T.resize_local_mean(psf, (new_psf_size,new_psf_size), grid_mode=False)
    psf_rescaled /= np.sum(psf_rescaled)

    return img_rescaled, err_rescaled, psf_rescaled, mask_rescaled, pxscale_new

def add_noise(img, err, sblim_new, bg_std, pxscale, zp):
    """Given the image, add noise to match a new surface brightness limit `sblim_new`.
    Args:
        img (np.ndarray): The input image in image units (e.g., ADU/s)
        sblim (float): The original surface brightness limit in mag/arcsec^2.
        sblim_new (float): The desired new surface brightness limit in mag/arcsec^2.
        bg_std (float): The original background standard deviation in image units.
        pxscale (float): The pixel scale in arcsec/px.
        zp (float): The zero point magnitude of the image.
    Returns:
        img_new (np.ndarray): The image with added noise to match the new SB limit.
        bgsd_new (float): The new background standard deviation in image units.
    """

    # Prepare new image etc arrays
    img_new = img.copy()

    # Add noise
    sblim_curr = -2.5*np.log10(bg_std/pxscale**2)+zp
    noise_factor = np.power(10, 2*(sblim_curr-sblim_new)/2.5) - 1
    noise_to_add = np.sqrt(bg_std**2 * noise_factor)

    if noise_factor < 0:
        print('Image noisier than requested; continuing')
        return None, None

    sky = np.random.normal(loc=0, scale=noise_to_add, size=img_new.shape)
    img_new += sky
    err_new  = np.sqrt(err**2 + (noise_to_add)**2)
    return img_new, err_new

def augment_galaxy(row, pxscales_approx, sblims, path='../data'):

    # Create output directory
    os.makedirs(f'{path}/augments/{row.galaxy}', exist_ok=True)
    os.makedirs(f'{path}/diag_figs/augments/{row.galaxy}', exist_ok=True)

    # Range of pixel scales in pc/px and SB limits in mag/arcsec2 to make if possible
    min_img_size = 20 # Minimum size of the galaxy image in pixels to attempt augmentation
    psf_fwhm = 2 # Desired PSF FWHM in pixels in the augmented image

    # Open the cutout file 
    file = fits.open(f'{path}/cutouts/{row.galaxy}.fits')
    img = file['SCI'].data
    err = file['ERR'].data
    mask = file['MASK'].data > 0
    exp = file['EXP'].data
    bg_rms = file['BG_RMS'].data
    header = file['SCI'].header
    psf = file['PSF'].data
    mask = mask | (exp == 0) # Mask out pixels with zero exposure time

    # Convert to counts
    sky = (bg_rms*exp)**2
    counts = img*exp + sky

    # Header information
    pxscale_og = row.pc_px
    pxscale_og_arcsec = header['PXSCALE']
    pc_per_arcsec = row.pc_px / pxscale_og_arcsec
    zp = header['ZP']
    z = row.z

    px_sizes = []
    for pxsize_approx in tqdm(pxscales_approx):

        ###### Check if the pixel scale is smaller than original
        factor = int(pxsize_approx/pxscale_og + 0.5)
        if factor < 1:
            print(f'Original pixel scale {pxscale_og} pc/px is larger than desired {pxsize_approx} pc/px; continuing')
            continue
        # pxscale_new = pxscale_og * factor
        # aug_file_path = f'{path}/augments/{row.galaxy}/px{int(pxscale_new)}_sb20.00.fits'
        # if os.path.exists(aug_file_path):
        #     print(f'Augmentation for pxsize {pxsize_approx} already exists; skipping')
        #     continue

        ###### 1. Rescale the image
        img_rescaled, err_rescaled, psf_rescaled, mask_rescaled, pxscale_new = rescale_image(
            img, counts, exp, sky, mask, pxsize_approx, pxscale_og, psf,
            psf_size_px=psf_fwhm, imsize_min=min_img_size)
        
        
        #If the image is too small to be rescaled, this returns None
        if img_rescaled is None:
            break 

        # Store the new pixel scale
        px_sizes.append(pxscale_new)
        pxscale_new_arcsec = pxscale_new / pc_per_arcsec

        # Re-estimate the background sigma
        _, _, bg_std, bg_mask, _ = estimate_bg(
            img_rescaled, mask_rescaled, mask_rescaled, filter_size=0, fit_bg=False, bg_sigma=2, grow_sigma=0.5
        )

        ###### 2. Loop over desired SB limits and add noise
        for sblim_new in sblims:

            # Add noise
            img_noisy, err_noisy = add_noise(img_rescaled, err_rescaled, sblim_new, bg_std, pxscale_new_arcsec, zp)
            if img_noisy is None: 
                break

            # Re-estimate the background to make sure our noise level is good
            _, _, bg_std_new, bg_mask_new, _ = estimate_bg(img_noisy, mask_rescaled, mask_rescaled, filter_size=0, fit_bg=False, bg_sigma=2, grow_sigma=0.5)

            # Save the new FITS file
            header_noisy = header.copy()
            header_noisy['PXSCALE'] = pxscale_new_arcsec
            header_noisy['BGSD'] = bg_std_new
            header_noisy['SBLIM'] = -2.5*np.log10(bg_std_new/pxscale_new_arcsec**2) + zp
            header_noisy['SBLIM0'] = header_noisy['SBLIM'] - 7.5*np.log10(z+1)
            header_noisy['PC_PX'] = pxscale_new

            hdu_img = fits.PrimaryHDU(img_noisy, header=header_noisy)
            hdu_err = fits.ImageHDU(err_noisy, name='ERR')
            hdu_mask = fits.ImageHDU(mask_rescaled.astype(np.uint8), name='MASK')
            hdu_psf = fits.ImageHDU(psf_rescaled, name='PSF')
            hdu_bg = fits.ImageHDU(bg_std_new*np.ones_like(img_noisy), name='BG_RMS')
            hdulist = fits.HDUList([hdu_img, hdu_err, hdu_mask, hdu_bg, hdu_psf])
            hdulist.writeto(f'{path}/augments/{row.galaxy}/px{int(pxscale_new)}_sb{sblim_new:.2f}.fits', overwrite=True)

            # Also save as image
            fig, axs = plt.subplots(1,6,figsize=(25,4))
            axs[0].imshow(-2.5*np.log10(np.abs(img_rescaled)/pxscale_new_arcsec**2) + zp, vmin=17, vmax=30, cmap='gray_r')
            axs[1].imshow(mask_rescaled, cmap='gray_r')
            axs[2].imshow(-2.5*np.log10(np.abs(np.ma.array(img_rescaled, mask=bg_mask))/pxscale_new_arcsec**2)+zp, vmin=17, vmax=30, cmap='gray_r')
            axs[3].imshow(-2.5*np.log10(np.abs(img_noisy)/pxscale_new_arcsec**2) + zp, vmin=17, vmax=30, cmap='gray_r')
            axs[4].imshow(img_noisy/err_noisy, vmin=0, vmax=3, cmap='gray')
            axs[5].imshow(-2.5*np.log10(np.abs(np.ma.array(img_noisy, mask=bg_mask_new))/pxscale_new_arcsec**2)+zp, vmin=17, vmax=30, cmap='gray_r')
            plt.savefig(f'{path}/diag_figs/augments/{row.galaxy}/px{int(pxscale_new)}_sb{sblim_new:.2f}.png', dpi=150)
            plt.close()

            # Write augmentation info to a CSV file
            with open(f'{path}/catalogs/augments.csv', 'a') as aug_file:
                aug_file.write(f'{row.galaxy},{pxscale_new:0.3f},{pxscale_new_arcsec:0.3f},{sblim_new:0.3f}\n')

    file.close()



    # Run `reduce_galaxy` for a partiular galaxy from command line
if __name__ == '__main__':

    # Inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("galaxy", type=str, help="Galaxy name")
    args = parser.parse_args()

    px_sizes_approx = [25,50,75,100,150,200,250,300,400,500,600,700,800,900,1000,1200,1500,2000]
    sblims = np.arange(20,26.25,0.25)

    catalog = pd.read_csv('../data/catalogs/sample.csv')
    row = catalog[catalog.galaxy==args.galaxy].iloc[0]
    augment_galaxy(row, pxscales_approx=px_sizes_approx, sblims=sblims, path='../data')


