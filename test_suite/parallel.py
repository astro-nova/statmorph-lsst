# Python script to parallelize different steps of the data analysis with joblib
# Edit the main function at the bottom to choose which step to run in parallel

import numpy as np
import pandas as pd
from tqdm import tqdm
from lib.augmentations import augment_galaxy
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


def parallel_augs(row):

    px_sizes_approx = [25,50,75,100,150,200,250,300,400,500,600,700,800,900,1000,1200,1500,2000]
    sblims = np.arange(20,26.25,0.25)

    try:
        augment_galaxy(row, px_sizes_approx, sblims, path='data')
    except Exception as e:
        print(f'Error processing galaxy {row.galaxy}: {e}')

def parallel_segmap(row, path='data', hot_snr=12, cold_snr=1.5):

    # hot_snr = 7

    from lib.utils import get_segmap, segmap_to_rle
    from astropy.io import fits
    import os 
    import json

    os.makedirs(f'{path}/segmaps/{row.galaxy}', exist_ok=True)
    os.makedirs(f'{path}/diag_figs/segmaps/{row.galaxy}', exist_ok=True)

    try:
        file = fits.open(f'{path}/augments/{row.galaxy}/px{int(row.pxscale_pc)}_sb{row.sblim:0.2f}.fits')

        # Load the relevant arrays and header info
        img = file[0].data
        mask = file['MASK'].data > 0
        header = file['SCI'].header
        zp = header['ZP']
        pxscale = header['PXSCALE']
        mask = mask | np.isinf(img) | np.isnan(img)

        # Workaround because I wrote the wrong header down
        pxscale = row.pxscale_arcsec
        bgsd = np.power(10, (zp-row.sblim)/2.5) * pxscale**2

        # Create a segmentation map
        segmask, segmap = get_segmap(img, mask, bgsd,  min_area=5, hot_snr=hot_snr, cold_snr=cold_snr, nlevels=32, contrast=0.001, overlap_thresh=0.2)
        if segmap is None:
            raise ValueError("No source detected")
        
        # Save as COCO RLEs 
        segmask_rle = segmap_to_rle(segmask)
        segmap_rle = segmap_to_rle(segmap)
        rle_dict = {'segmask': segmask_rle, 'segmap': segmap_rle}
        out_dict = {'galaxy': row.galaxy, 'pxscale_pc': row.pxscale_pc, 'sblim': row.sblim, 'rles': rle_dict, 'hot_snr': hot_snr, 'cold_snr': cold_snr}
        with open(f'{path}/segmaps/{row.galaxy}/px{int(row.pxscale_pc)}_sb{row.sblim:0.2f}.json', 'w') as f:
            json.dump(rle_dict, f)
        with open(f'{path}/segmaps/segmaps.jsonl', 'a') as f:
            json.dump(out_dict, f)
            f.write("\n")

        # Plot the segmap
        plt.figure(figsize=(8, 8))
        plt.imshow(-2.5*np.log10(np.abs(img)/pxscale**2) + zp, vmin=18, vmax=27, cmap='gray_r')
        plt.contour(segmask>0, colors='#d35', linewidths=0.5, levels=[0.9,1.1])
        plt.contour(segmap.data>0, colors='k', linewidths=0.5, levels=[0.9,1.1])
        plt.savefig(f'{path}/diag_figs/segmaps/{row.galaxy}/px{int(row.pxscale_pc)}_sb{row.sblim:0.2f}.png', dpi=150)
        plt.close()

    except Exception as e:
        print(f'Error processing galaxy {row.galaxy}: {e}')
        # Write info about failed runs in a CSV file
        with open(f'{path}/catalogs/segmap_errors.csv', 'a') as aug_file:
            aug_file.write(f'{row.galaxy},{row.pxscale_pc},{row.sblim}\n')

if __name__ == '__main__':

    # catalog = pd.read_csv('data/catalogs/sample.csv')
    # catalog = catalog[catalog.galaxy == 'gmp2364']
    catalog = pd.read_csv('data/catalogs/augments.csv')
    # catalog = sample[~sample.galaxy.isin(catalog.galaxy)]


    # Parallel(n_jobs=15)(delayed(parallel_augs)(row) for idx, row in 
    #                     tqdm(catalog.iterrows(), total=len(catalog), position=0, desc="Processing galaxies"))

    # galaxy = "gin762"
    # sblim_min = 25.5
    # sblim_max = 27
    # pxscale_min = 0
    # pxscale_max = 2000

    # catalog = catalog[(catalog.galaxy == galaxy) & 
    #                   (catalog.sblim >= sblim_min) & 
    #                   (catalog.sblim <= sblim_max) & 
    #                   (catalog.pxscale_pc >= pxscale_min) & 
    #                   (catalog.pxscale_pc <= pxscale_max)]

    hot_snr = 30
    cold_snr = 2
    Parallel(n_jobs=15)(delayed(parallel_segmap)(row, hot_snr=hot_snr, cold_snr=cold_snr) for idx, row in 
                        tqdm(catalog.iterrows(), total=len(catalog), position=0, desc="Processing galaxies"))


