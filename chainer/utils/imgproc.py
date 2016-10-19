import numpy


def oversample(images, crop_dims):
    """Crop an image into center, corners, and mirror images."""

    # Dimensions and center.
    channels, src_h, src_w = images[0].shape
    cy, cx = src_h / 2.0, src_w / 2.0
    dst_h, dst_w = crop_dims

    # Make crop coordinates
    crops_ix = numpy.empty((5, 4), dtype=int)
    crops_ix[0, :2] = [0, 0]
    crops_ix[1, :2] = [0, src_w - dst_w]
    crops_ix[2, :2] = [src_h - dst_h, 0]
    crops_ix[3, :2] = [src_h - dst_h, src_w - dst_w]
    crops_ix[4, :2] = [int(cy - dst_h / 2.0), int(cx - dst_w / 2.0)]
    crops_ix[:, 2] = crops_ix[:, 0] + dst_h
    crops_ix[:, 3] = crops_ix[:, 1] + dst_w

    crops = numpy.empty(
        (10 * len(images), channels, dst_h, dst_w), dtype=images[0].dtype)
    ix = 0
    for img in images:
        for crop in crops_ix:
            crops[ix] = img[:, crop[0]:crop[2], crop[1]:crop[3]]
            ix += 1
        crops[ix:ix + 5] = crops[ix - 5:ix, :, :, ::-1]
        ix += 5
    return crops
