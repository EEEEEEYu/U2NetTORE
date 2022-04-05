import numpy as np

def gen_tore_plus(tore, threshold=None, percentile=95):
    """ Generate the PLUS version of tore volume.
        Two filtered most recent cache layers are added to the volume.
    Args:
        tore: ndarry, (6, h, w).
        percentile: float, 0~100. The percentile threshold. Works on
            neg and pose separately. Percentile has higher priority.
            80 is recommended as default percentile.
        threshold: float, 0~1. The fixed threshold for both neg and pos.
            0.5 is recommended as default threshold.

    Return:
        ndarry, (8, h, w). NTORE volume proposed.
    """
    if percentile is not None:
        pos_thres = np.percentile(tore[0][tore[0] != 0], percentile)
        neg_thres = np.percentile(tore[3][tore[3] != 0], percentile)
        pos_recent = np.where(tore[0] > pos_thres, tore[0], 0)[np.newaxis, :]
        neg_recent = np.where(tore[3] > neg_thres, tore[3], 0)[np.newaxis, :]
    elif threshold is not None:
        pos_recent = np.where(tore[0] > threshold, tore[0], 0)[np.newaxis, :]
        neg_recent = np.where(tore[3] > threshold, tore[3], 0)[np.newaxis, :]
    else:
        raise ValueError('Please specify the value of threshold or percentile!')
    tore_plus = np.concatenate((pos_recent, tore[0:3], neg_recent, tore[3:]), axis=0)
    return tore_plus

def to_float(x):
    if isinstance(x, np.ndarray):
        return x.astype(float)
    else:
        return x.float()

def ntore_to_redgreen(tore):
    """ Make each layer of tore a rgb image in which red is the positive channel and green is negative.
    Args:
        tore: ndarray, whether 6 or 8 channels.
    Return:
        combined: the combined images.
    """
    _, h, w = tore.shape
    tore = tore.reshape(2, -1, h, w)
    combined = np.concatenate((tore, np.zeros((1, tore.shape[1],h,w))), axis=0).swapaxes(0,1)
    return combined