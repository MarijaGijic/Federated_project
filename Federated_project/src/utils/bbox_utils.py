import math


def points_to_bbox(points):
    if not points:
        return None

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    width = x_max - x_min
    height = y_max - y_min

    return x_min, y_min, width, height


def clip_bbox_to_image(bbox, image_width, image_height):
    """Intersect an ``(xmin, ymin, width, height)`` box with an image."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    xmin, ymin, width, height = bbox
    if width <= 0 or height <= 0:
        return None

    clipped_xmin = max(0, xmin)
    clipped_ymin = max(0, ymin)
    clipped_xmax = min(image_width, xmin + width)
    clipped_ymax = min(image_height, ymin + height)
    clipped_width = clipped_xmax - clipped_xmin
    clipped_height = clipped_ymax - clipped_ymin

    if clipped_width <= 0 or clipped_height <= 0:
        return None

    return clipped_xmin, clipped_ymin, clipped_width, clipped_height


def point_to_resized_bbox(point, image_width, image_height, target_size):
    """Give a point the minimum approximately one-target-pixel footprint.

    This is required by the target mask resolution, not an estimate of the
    anatomical size of the finding.
    """
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    bbox_width = min(image_width, max(1, math.ceil(image_width / target_size)))
    bbox_height = min(image_height, max(1, math.ceil(image_height / target_size)))
    x = round(point[0] - bbox_width / 2)
    y = round(point[1] - bbox_height / 2)
    x = min(max(0, x), image_width - bbox_width)
    y = min(max(0, y), image_height - bbox_height)
    return x, y, bbox_width, bbox_height
