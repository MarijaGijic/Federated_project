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