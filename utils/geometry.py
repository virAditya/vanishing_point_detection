"""
Geometry utilities for vanishing point detection
Contains all geometric calculations and line operations
"""

import numpy as np
import math


def line_intersection(line1, line2):
    """
    Find intersection point of two lines in normal form ax + by + c = 0

    Args:
        line1 (tuple): First line coefficients (a1, b1, c1)
        line2 (tuple): Second line coefficients (a2, b2, c2)

    Returns:
        tuple or None: Intersection point (x, y) or None if parallel
    """
    a1, b1, c1 = line1
    a2, b2, c2 = line2

    # Calculate determinant
    det = a1 * b2 - a2 * b1

    # Lines are parallel if determinant is close to zero
    if abs(det) < 1e-10:
        return None

    # Calculate intersection point
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det

    return (x, y)


def fit_line_svd(points):
    """
    Fit a line to a set of points using SVD (Singular Value Decomposition)

    Args:
        points (list): List of (x, y, _) tuples or (x, y) tuples

    Returns:
        tuple or None: Line equation coefficients (a, b, c) where ax + by + c = 0
    """
    if len(points) < 2:
        return None

    # Extract coordinates (handle both 2D and 3D point formats)
    try:
        x_coords = np.array([p[0] for p in points])
        y_coords = np.array([p[1] for p in points])
    except (IndexError, TypeError):
        return None

    # Calculate centroid
    centroid = np.array([np.mean(x_coords), np.mean(y_coords)])

    # Center the points
    x_centered = x_coords - centroid[0]
    y_centered = y_coords - centroid[1]

    coordinates = np.column_stack([x_centered, y_centered])

    if coordinates.shape[0] < 2:
        return None

    # Use SVD to find the best fit line
    try:
        _, _, V = np.linalg.svd(coordinates)
        # The line direction is the first principal component
        direction = V[0]

        # Convert to normal form ax + by + c = 0
        a = -direction[1]
        b = direction[0]
        c = direction[1] * centroid[0] - direction[0] * centroid[1]

        # Normalize to avoid numerical issues
        norm = math.sqrt(a*a + b*b)
        if norm > 1e-10:
            a, b, c = a/norm, b/norm, c/norm

        return (a, b, c)
    except np.linalg.LinAlgError:
        return None


def calculate_line_endpoints(line_coeffs, image_shape, extension_factor=1.5):
    """
    Calculate line endpoints for visualization across image bounds

    Args:
        line_coeffs (tuple): Line coefficients (a, b, c) where ax + by + c = 0
        image_shape (tuple): Image shape (height, width) or (height, width, channels)
        extension_factor (float): Factor to extend lines beyond image bounds

    Returns:
        tuple: ((x1, y1), (x2, y2)) line endpoints
    """
    a, b, c = line_coeffs
    h, w = image_shape[:2]

    # Extend beyond image bounds
    x_min = -int(w * (extension_factor - 1) / 2)
    x_max = int(w * extension_factor)
    y_min = -int(h * (extension_factor - 1) / 2)
    y_max = int(h * extension_factor)

    # Calculate line endpoints based on line orientation
    if abs(b) > abs(a):  # More horizontal line
        x1, x2 = x_min, x_max
        y1 = (-c - a * x1) / b if abs(b) > 1e-10 else 0
        y2 = (-c - a * x2) / b if abs(b) > 1e-10 else 0
    else:  # More vertical line
        y1, y2 = y_min, y_max
        x1 = (-c - b * y1) / a if abs(a) > 1e-10 else 0
        x2 = (-c - b * y2) / a if abs(a) > 1e-10 else 0

    return ((x1, y1), (x2, y2))


def filter_intersections_by_bounds(intersections, image_shape, boundary_factor=2.0):
    """
    Filter intersections to keep only those within reasonable bounds

    Args:
        intersections (list): List of intersection points (x, y)
        image_shape (tuple): Image shape (height, width) or (height, width, channels)
        boundary_factor (float): Factor for boundary extension

    Returns:
        list: Filtered intersections within bounds
    """
    if not intersections:
        return []

    h, w = image_shape[:2]
    filtered = []

    x_min = -w * boundary_factor
    x_max = w * (1 + boundary_factor)
    y_min = -h * boundary_factor
    y_max = h * (1 + boundary_factor)

    for x, y in intersections:
        if x_min < x < x_max and y_min < y < y_max:
            filtered.append((x, y))

    return filtered


def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points

    Args:
        point1 (tuple): First point (x, y)
        point2 (tuple): Second point (x, y)

    Returns:
        float: Distance between points
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def angle_between_lines(line1, line2):
    """
    Calculate angle between two lines in normal form

    Args:
        line1 (tuple): First line coefficients (a1, b1, c1)
        line2 (tuple): Second line coefficients (a2, b2, c2)

    Returns:
        float: Angle between lines in radians (0 to π/2)
    """
    a1, b1, _ = line1
    a2, b2, _ = line2

    # Direction vectors (perpendicular to normal vectors)
    dir1 = np.array([-b1, a1])
    dir2 = np.array([-b2, a2])

    # Normalize direction vectors
    norm1 = np.linalg.norm(dir1)
    norm2 = np.linalg.norm(dir2)

    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    dir1_normalized = dir1 / norm1
    dir2_normalized = dir2 / norm2

    # Calculate angle using dot product
    cos_angle = np.clip(np.dot(dir1_normalized, dir2_normalized), -1.0, 1.0)
    angle = math.acos(abs(cos_angle))  # Take absolute for acute angle

    return angle


def point_to_line_distance(point, line_coeffs):
    """
    Calculate perpendicular distance from point to line

    Args:
        point (tuple): Point coordinates (x, y)
        line_coeffs (tuple): Line coefficients (a, b, c)

    Returns:
        float: Distance from point to line
    """
    x, y = point
    a, b, c = line_coeffs

    # Distance formula: |ax + by + c| / sqrt(a² + b²)
    numerator = abs(a * x + b * y + c)
    denominator = math.sqrt(a * a + b * b)

    if denominator < 1e-10:
        return float('inf')

    return numerator / denominator


def line_segment_intersection(seg1, seg2):
    """
    Find intersection of two line segments

    Args:
        seg1 (tuple): First segment ((x1, y1), (x2, y2))
        seg2 (tuple): Second segment ((x3, y3), (x4, y4))

    Returns:
        tuple or None: Intersection point or None
    """
    (x1, y1), (x2, y2) = seg1
    (x3, y3), (x4, y4) = seg2

    # Calculate denominators
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:  # Lines are parallel
        return None

    # Calculate parameters
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    # Check if intersection is within both segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)

    return None


def normalize_angle(angle):
    """Normalize angle to [0, π] range"""
    return angle % math.pi


def are_lines_parallel(line1, line2, tolerance=1e-6):
    """
    Check if two lines are parallel within tolerance

    Args:
        line1, line2 (tuple): Line coefficients
        tolerance (float): Tolerance for parallel check

    Returns:
        bool: True if lines are parallel
    """
    a1, b1, _ = line1
    a2, b2, _ = line2

    cross_product = abs(a1 * b2 - a2 * b1)
    return cross_product < tolerance


def create_line_from_points(point1, point2):
    """
    Create line equation from two points

    Args:
        point1, point2 (tuple): Points (x, y)

    Returns:
        tuple: Line coefficients (a, b, c)
    """
    x1, y1 = point1
    x2, y2 = point2

    # Calculate line coefficients
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2

    # Normalize to avoid numerical issues
    norm = math.sqrt(a * a + b * b)
    if norm > 1e-10:
        a, b, c = a / norm, b / norm, c / norm

    return (a, b, c)


def is_point_on_line(point, line_coeffs, tolerance=1e-6):
    """
    Check if a point lies on a line within tolerance

    Args:
        point (tuple): Point coordinates
        line_coeffs (tuple): Line coefficients
        tolerance (float): Distance tolerance

    Returns:
        bool: True if point is on line
    """
    distance = point_to_line_distance(point, line_coeffs)
    return distance < tolerance
