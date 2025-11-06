# imports
import numpy as np

def segmap_to_rle(segmap):
    """Convert a segmentation map to COCO RLE format.
    
    Args:
        segmap (SegmentationImage): segmentation map
    Returns:
        rles (list): list of RLEs for each segment in the segmap
    """
    from pycocotools import mask as maskUtils

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
    from pycocotools import mask as maskUtils

    if rle is None:
        return None
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