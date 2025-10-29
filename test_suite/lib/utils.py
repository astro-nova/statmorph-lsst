import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources
from photutils.background import Background2D
from astropy.convolution import convolve, Tophat2DKernel
import matplotlib.pyplot as plt
from skimage.restoration import inpaint
from photutils.segmentation import deblend_sources, detect_sources
from matplotlib.colors import ListedColormap
from scipy.stats import mode
from pycocotools import mask as maskUtils

def grow_segmask(segmap, grow_sigma=1, area_norm=10):
    """ Grow the segmentation mask based on each labelled region's size """
    areas_arr = np.concatenate([[0],segmap.areas]).astype(float)
    segmap_areas = areas_arr[segmap.data]
    segmask = ndi.gaussian_filter(segmap_areas.astype(float)/area_norm, sigma=grow_sigma, truncate=5) > 0.05
    return segmask

def estimate_bg(img, mask, chipmask, max_iter=30, thresh=0.05, fit_bg=True, masked_thresh=0.2, filter_size=3, bg_sigma=1, grow_sigma=3):
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
        bg_sigma (float): multiplier for the background sigma when iteratively detecting sources
        grow_sigma (float): sigma for growing the segmentation mask
    Returns:
        img: background-subtracted image
        median: background median
        std: background standard deviation
        bg_mask: mask (including the detected sources) used to calculate BG
        bg: NxN background image
    """

    img_copy = img.copy()
    if filter_size > 0:
        img_conv = ndi.gaussian_filter(img_copy, sigma=filter_size, truncate=2)
    else:
        img_conv = img_copy.copy()

    # First pass at getting the background RMS
    mean, median, std = sigma_clipped_stats(img_copy, sigma=3.0, mask=mask)
    std_og = std 

    detect_area = 10
    # Recursively mask sources and re-calculate bg while we do not converge
    for i in range(max_iter):
        std_prev = std
    
        segmap = detect_sources(img_conv, bg_sigma*std+median, npixels=detect_area, mask=chipmask, connectivity=4)
        if segmap is None:
            no_sources = True
            segmask = np.zeros_like(img_copy, dtype=bool)
        else:
            no_sources = False
            segmask = segmap.data > 0

        # Stop if there are fewer than 20% unmasked pixels
        masked_frac = np.sum(segmask | mask)/img.size
        if masked_frac > (1-masked_thresh):
            break
        
        mean, median, std = sigma_clipped_stats(img_copy, sigma=3.0, mask=mask|segmask)

        err = np.abs(std-std_prev)/std_prev
        if (err < thresh) or no_sources: break

    
    # Grow the mask edges by an amount relative to the segment size
    if not no_sources:
        segmask = grow_segmask(segmap, grow_sigma=grow_sigma, area_norm=1)
    bg_mask = mask|segmask

    if fit_bg:
        # Estimate 2D background
        box_size = int(img.shape[0]*0.01)
        bkg = Background2D(img_copy, (box_size,box_size), filter_size=3, mask=bg_mask, coverage_mask=chipmask)

        fig, axs = plt.subplots(1,3,figsize=(15,5))
        axs[0].imshow(img, vmin=median-std, vmax=median+std, cmap='gray')
        axs[1].imshow(bkg.background, vmin=median-std, vmax=median+std, cmap='gray')
        axs[2].imshow(img-bkg.background, vmin=-std, vmax=std, cmap='gray')
        for ax in axs:
            ax.contour((segmask | chipmask)>0, colors=['C9'], levels=[0.9,1.1], linewidths=0.5)
        plt.show()

        answer = input("Subtract background? y | n\n")
        if answer == 'y': 
            img_copy -= bkg.background
            std = bkg.background_rms_median
            bg_std_map = bkg.background_rms
        else:
            mean, median, std = sigma_clipped_stats(img_copy, sigma=3.0, mask=bg_mask)
            img_copy -= median
            bg_std_map = std*np.ones_like(img_copy)
    else:
        mean, median, std = sigma_clipped_stats(img_copy, sigma=3.0, mask=bg_mask)
        img_copy -= median
        bg_std_map = std*np.ones_like(img_copy)
    
    # Just return the normal non-iterative background if didn't reach convergence
    if i==max_iter-1: 
        print("WARNING! Background didn't converge. Returning ismple sigma-clip background")
        std = std_og
    
    plt.close()
    return img_copy, median, std, bg_mask, bg_std_map

def _enclosed_masked_regions(mask: np.ndarray) -> np.ndarray:
    """ Returns a mask where only the enclosed masked regions are kept (i.e. not touching edges) """
    mask_tmp1 = mask.copy()
    mask_tmp1[0,:] = False
    mask_tmp1[-1,:] = False
    partially_enclosed = ndi.binary_fill_holes(~mask_tmp1)

    mask_tmp2 = mask.copy()
    mask_tmp2[:,0] = False
    mask_tmp2[:,-1] = False
    partially_enclosed &= ndi.binary_fill_holes(~mask_tmp2)
    partially_enclosed = ~partially_enclosed

    return partially_enclosed

def _fill_nan_along_x(a):
    """
    Row-wise linear interpolation.
    Fills only NaNs that are bounded on the left & right within each row.
    Leaves edge NaNs (unbounded along x) untouched.
    """
    a = np.asarray(a, float)
    return pd.DataFrame(a).interpolate(
        axis=1, method="linear",
        limit_direction="both",
        limit_area="inside"  # <-- only fill interior NaN runs
    ).values

def _interpolate_missing_pixels(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Interpolates over the chip gap and bad pixels using astropy's interpolation.
    Args:
        image: NxN image array
        mask: NxN mask with bad pixels (but not the regions outside the detector area, chipmask)
    """
    # Only interpolate over enclosed regions
    enclosed_mask = _enclosed_masked_regions(mask)
    image_interp = image.copy()  # Avoid log(0)
    # # image_interp[mask] = -99
    # image_interp[enclosed_mask] = np.nan
    # image_interp = interpolate_replace_nans(image_interp, Box2DKernel(filtsize), convolve=convolve_fft)
    # # # image_interp[~mask] = image[~mask]
    # # image_interp[mask & ~enclosed_mask] = image[mask & ~enclosed_mask]
    # image_interp = inpaint.inpaint_biharmonic(image_interp, enclosed_mask)
    # image_interp[~enclosed_mask] = image[~enclosed_mask]
    # image_interp = image.copy()
    image_interp[enclosed_mask] = np.nan
    image_interp = _fill_nan_along_x(image_interp)
    return image_interp

def _segmap_cmap(segmap, cmap='tab20c'):
    """Create a color map for a segmentation map for visualization by randomly shuffling 
    discrete cmap such that adjacent neighbors have different colors."""
    segmap_unique = np.unique(segmap.data)
    n_labels = len(segmap_unique)
    
    # Get base colormap colors
    base_cmap = plt.get_cmap(cmap, n_labels)
    colors = base_cmap.colors.copy()
    
    # Shuffle colors for better contrast between adjacent regions
    np.random.shuffle(colors)
    
    # Set first color (index 0, background) to black
    if len(colors) > 0:
        colors[0] = [0, 0, 0, 1]  # Black for background (label==0)
    
    # Create and return the colormap
    shuffled_cmap = ListedColormap(colors)
    shuffled_cmap.set_bad(color='k')
    return shuffled_cmap

def _cent_label(segmap, imsize):
    """Get the label of the central object in the segmentation map."""
    xc, yc = int(imsize/2), int(imsize/2)
    window = np.max([3, int(0.1*imsize)])
    labels = segmap.data[yc-window:yc+window, xc-window:xc+window]
    if np.all(labels == 0):
        return 0
    else:
        cent_label = mode(labels[labels > 0], keepdims=False)[0]
        return cent_label

def _segmap(img, mask, bgsd, snr=15, min_area=5):
    """Get the hot segmentation map to mask bright sources.
    
    Args: 
        img (np.ndarray): NxN image array (with interpolated chip gaps)
        bgsd (float): standard deviation of the background
        hot_snr (float): detection limit in terms of bgsd for the hot run
        min_area (int): minimum area (in pixels) for detected sources
    Returns:
        segmap_hot (SegmentationImage): segmentation map from the hot run
    """
    # Detect all sources with high SNR
    segmap = detect_sources(img, threshold=snr*bgsd, npixels=min_area, connectivity=4, mask=mask)
    return segmap

def _deblended_mask(segmap_inp, segmask_hot, labels, cold_cent, overlap_thresh=0.2):
    """ For the main source, mask any deblended segments where the overlap with a hot source is significant"""

    segmap = segmap_inp.copy()
    imsize = segmap_inp.data.shape[0]
    cent_label_debl = _cent_label(segmap, imsize)

    segmap.keep_labels(labels)
    segmap.remove_label(cent_label_debl)
    segmap.relabel_consecutive()

    areas = segmap.areas
    labels = segmap.labels
    
    for label, area in zip(labels, areas):
        # Calculate overlap with hot segmap
        overlap = np.sum((segmap.data == label) & (segmask_hot))
        overlap_frac = overlap / area 
        # If the overlap is small, remove from the mask
        if overlap_frac < overlap_thresh:
            # Keep this deblended region
            segmap.remove_label(label)
    segmap.relabel_consecutive()
    return segmap

def get_segmap(img, mask, bgsd,  min_area=5, hot_snr=20, cold_snr=1.5, nlevels=32, contrast=0.001, overlap_thresh=0.2):
    """Get the segmentation map using a cold + hot run, first masking bright sources, then all faint sources
    other than the central galaxy.
    
    Args: 
        img (np.ndarray): NxN image array (with interpolated chip gaps)
        bgsd (float): standard deviation of the background
        pxscale (float): pixel scale in arcseconds
        hot_thresh (float): detection limit in terms of bgsd for the hot run
        cold_thresh (float): detection limit in terms of bgsd for the cold run
        contrast (float): between 0 and 1, regulates deblending strength
        nlevels (int): number of levels to use in deblending
        
    Returns:
        segmask (np.ndarray): boolean array where all contaminant sources are masked as 1
        segmap (SegmentationImage): segmentation map with only the central source
        img_conv (np.ndarra): convolved image (for use later in A_shape)
    """

    #### Interpolate over chip gaps and bad pixels only if masked regions are actually bad ####
    # enclosed_mask = _enclosed_masked_regions(mask)
    # if np.sum(np.isnan(img[enclosed_mask]) | np.isinf(img[enclosed_mask])) >= 0.5*np.sum(enclosed_mask):
    #     img_interp = _interpolate_missing_pixels(img, mask)
    # else:
    #     img_interp = img.copy()
    img_interp = _interpolate_missing_pixels(img, mask)
    mask_interp = np.isnan(img_interp) | np.isinf(img_interp)
    

    # For large images, smooth the image to avoid detecting noise peaks
    if img.shape[0] >= 300:
        img_interp = ndi.gaussian_filter(img_interp, sigma=2, truncate=1)

    #### Hot run
    segmap_hot = detect_sources(img_interp, threshold=hot_snr*bgsd, 
                                 mask=mask_interp, npixels=min_area, connectivity=4)
    
    # Mask all but the central source
    if segmap_hot is not None:
        cent_label = _cent_label(segmap_hot, img.shape[0])
        if cent_label > 0:
            segmap_hot.remove_label(cent_label)
            segmap_hot.relabel_consecutive()

        # Grow the mask based on the size of each source
        grow_sigma = max(0.01*img.shape[0], min_area/10)
        segmask_hot = grow_segmask(segmap_hot, grow_sigma=grow_sigma, area_norm=min_area)
    else:
        segmask_hot = np.zeros_like(img).astype(bool)

    


    ####### Cold run ###########################
    segmap_cold = detect_sources(img_interp, threshold=cold_snr*bgsd, 
                                 mask=mask_interp, npixels=min_area, connectivity=4)
    
    # Get the central source
    cold_cent = _cent_label(segmap_cold, img.shape[0])
    if cold_cent == 0:
        print(f"No source detected at SNR={cold_snr}!")
        return None, None
    
    # Deblend the central source only
    cold_deblended = deblend_sources(img_interp, segmap_cold, npixels=5, contrast=contrast, progress_bar=False, nlevels=nlevels, labels=cold_cent)

    ######## Combining the two ###########################
    # 1. Mask all sources in the cold segmap except the central one
    segmap_cold_other = segmap_cold.copy()
    segmap_cold_other.remove_label(cold_cent)
    segmap_cold_other.relabel_consecutive()
    # Grow the cold mask
    segmask_cold = grow_segmask(segmap_cold_other, grow_sigma=min_area/10, area_norm=min_area)

    # 2. For the main source, mask any deblended segments where the overal with a hot source is significant
    # This loops over all deblended regions in the central source and removes any
    # with little overlap with a hot source. Then we mask all that remain
    cent_labels = np.unique(cold_deblended.data[segmap_cold.data == cold_cent])
    segmap_cold_deblended = _deblended_mask(cold_deblended, segmask_hot, cent_labels, cold_cent, overlap_thresh=overlap_thresh)
    segmask_cold_deblended = grow_segmask(segmap_cold_deblended, grow_sigma=min_area/10, area_norm=min_area)
    segmask_cold_deblended = segmap_cold_deblended.data > 0

    # Combine all of these masks
    segmask = segmask_hot | segmask_cold | segmask_cold_deblended

    # Run a final source detection to get the segmap of the central source only
    segmap_final = detect_sources(img_interp, threshold=cold_snr*bgsd, 
                                 mask=mask_interp | segmask, npixels=min_area, connectivity=4)
    final_cent = _cent_label(segmap_final, img.shape[0])
    segmap_final.keep_label(final_cent)
    segmap_final.relabel_consecutive()
    return segmask, segmap_final

def segmap_to_rle(segmap):
    """Convert a segmentation map to COCO RLE format.
    
    Args:
        segmap (SegmentationImage): segmentation map
    Returns:
        rles (list): list of RLEs for each segment in the segmap
    """
    # Check if we are passed a segmap object or a numpy array
    if type(segmap) is np.ndarray:
        labels = np.unique(segmap)
        data = segmap
    else:
        labels = segmap.labels
        data = segmap.data

    labels = labels[labels > 0]  # Exclude background label 0
    if len(labels) == 0:
        return None
    
    rles = []
    for label in labels:
        binary_mask = (data == label)
        rle = maskUtils.encode(np.asfortranarray(binary_mask))
        rle["counts"] = rle["counts"].decode("ascii")
        rles.append(rle)

    if len(rles) == 1:
        rles = rles[0]

    return rles

def rle_to_segmap(rle):
    """Convert COCO RLE format back to a segmentation map.
    Args:
        rle (list or dict): list of RLEs for each segment in the segmap, or a single RLE dict
        shape (tuple): shape of the output segmentation map (height, width)
    Returns:
        segmap (np.ndarray): segmentation map
    """
    if isinstance(rle, dict):
        # Single RLE case
        if type(rle["counts"]) is str:
            rle["counts"] = rle["counts"].encode("ascii")
        binary_mask = maskUtils.decode(rle)
        segmap = binary_mask.astype(np.int32)
    else:
        # List of RLEs case
        for i, rle_item in enumerate(rle):
            if type(rle_item["counts"]) is str:
                rle_item["counts"] = rle_item["counts"].encode("ascii") 
            binary_mask = maskUtils.decode(rle_item)

            if i == 0:
                segmap = np.zeros(binary_mask.shape, dtype=np.int32)

            segmap[binary_mask > 0] = i + 1  # Labels start from 1

    return segmap