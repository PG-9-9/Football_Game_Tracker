def get_center_of_bbox(bbox):
    """
    Calculate the center of a bounding box.

    Args:
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).

    Returns:
        tuple: Coordinates of the center of the bounding box (x, y).
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    """
    Calculate the width of a bounding box.

    Args:
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).

    Returns:
        int: Width of the bounding box.
    """
    return bbox[2] - bbox[0]


def measure_distance(p1, p2):
    """
    Measure the Euclidean distance between two points.

    Args:
        p1 (tuple): Coordinates of the first point (x, y).
        p2 (tuple): Coordinates of the second point (x, y).

    Returns:
        float: Euclidean distance between the two points.
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def measure_xy_distance(p1, p2):
    """
    Calculate the x and y distance between two points.

    Args:
        p1 (tuple): Coordinates of the first point (x, y).
        p2 (tuple): Coordinates of the second point (x, y).

    Returns:
        tuple: x and y distances.
    """
    return p1[0] - p2[0], p1[1] - p2[1]


def get_foot_position(bbox):
    """
    Calculate the bottom-center position of a bounding box.

    Args:
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).

    Returns:
        tuple: Coordinates of the bottom-center position (x, y).
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)
