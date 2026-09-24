import torch


def ckbd_split(y):
    """
    Split y to anchor and non-anchor
    anchor :
        0 1 0 1 0
        1 0 1 0 1
        0 1 0 1 0
        1 0 1 0 1
        0 1 0 1 0
    non-anchor:
        1 0 1 0 1
        0 1 0 1 0
        1 0 1 0 1
        0 1 0 1 0
        1 0 1 0 1
    """
    anchor = ckbd_anchor(y)
    nonanchor = ckbd_nonanchor(y)
    return anchor, nonanchor

def ckbd_merge(anchor, nonanchor):
    return anchor + nonanchor

def ckbd_anchor(y):
    anchor = torch.zeros_like(y).to(y.device)
    anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
    anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]
    return anchor

def ckbd_nonanchor(y):
    nonanchor = torch.zeros_like(y).to(y.device)
    nonanchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
    nonanchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
    return nonanchor

def calculate_bpp(compressed_data, num_pixels, bytes=True, num_bytes=None):
    """Calculate bpp given the compressed text and number of pixels."""
    scaling_factor = 8 if bytes else 1
    if num_bytes:
        return num_bytes * scaling_factor / num_pixels
    return len(compressed_data) * scaling_factor / num_pixels
