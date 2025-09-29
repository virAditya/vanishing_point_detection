"""
Line detection and grouping module
Groups edge points into parallel lines and fits lines using SVD
"""

import numpy as np
from collections import defaultdict
import math


class LineDetector:
    """
    Line detection class that groups edge points into parallel lines
    and fits line equations to point groups using SVD
    """

    def __init__(self, config=None):
        """Initialize line detector with configuration"""
        try:
            from config import DEFAULT_CONFIG
            self.config = config or DEFAULT_CONFIG
        except ImportError:
            self.config = self._create_fallback_config()

        self.debug_info = {}

    def _create_fallback_config(self):
        """Create fallback configuration"""
        class FallbackConfig:
            ANGLE_TOLERANCE = 0.2
            MIN_LINE_POINTS = 10
            LINE_FITTING_METHOD = 'svd'
            MIN_PARALLEL_LINE_GROUPS = 2
            MIN_LINES_PER_GROUP = 2
            MAX_LINE_DISTANCE_THRESHOLD = 5

        return FallbackConfig()

    def group_parallel_lines(self, edge_points):
        """
        Group edge points by orientation into parallel line segments

        Args:
            edge_points (list): List of (x, y, orientation) tuples

        Returns:
            list: List of point groups representing parallel lines
        """
        if not edge_points:
            return []

        # Group points by similar orientations
        angle_groups = defaultdict(list)

        for x, y, angle in edge_points:
            # Normalize angle to [0, π] to handle opposite directions
            norm_angle = angle % np.pi
            # Group by discretized angle
            angle_key = round(norm_angle / self.config.ANGLE_TOLERANCE) * self.config.ANGLE_TOLERANCE
            angle_groups[angle_key].append((x, y, angle))

        # Convert groups to line segments (filter by minimum points)
        line_groups = []
        for angle, points in angle_groups.items():
            if len(points) >= self.config.MIN_LINE_POINTS:
                line_groups.append(points)

        # Store debug information
        self.debug_info.update({
            'total_angle_groups': len(angle_groups),
            'valid_line_groups': len(line_groups),
            'points_per_group': [len(group) for group in line_groups],
            'angle_tolerance_used': self.config.ANGLE_TOLERANCE
        })

        return line_groups

    def fit_lines_to_groups(self, line_groups):
        """
        Fit lines to each group using the configured method

        Args:
            line_groups (list): List of point groups

        Returns:
            list: List of (line_equation, points) tuples
        """
        fitted_lines = []
        successful_fits = 0
        failed_fits = 0

        for group in line_groups:
            if getattr(self.config, 'LINE_FITTING_METHOD', 'svd') == 'svd':
                line_eq = self._fit_line_svd(group)
            else:
                line_eq = self._fit_line_svd(group)  # Default to SVD

            if line_eq:
                fitted_lines.append((line_eq, group))
                successful_fits += 1
            else:
                failed_fits += 1

        # Store debug information
        self.debug_info.update({
            'input_groups': len(line_groups),
            'fitted_lines': len(fitted_lines),
            'successful_fits': successful_fits,
            'failed_fits': failed_fits,
            'fitting_method': getattr(self.config, 'LINE_FITTING_METHOD', 'svd')
        })

        return fitted_lines

    def _fit_line_svd(self, points):
        """
        Fit a line to points using SVD (Singular Value Decomposition)

        Args:
            points (list): List of (x, y, _) tuples

        Returns:
            tuple or None: Line equation (a, b, c) where ax + by + c = 0
        """
        if len(points) < 2:
            return None

        try:
            # Extract coordinates
            x_coords = np.array([p[0] for p in points])
            y_coords = np.array([p[1] for p in points])

            # Calculate centroid
            centroid = np.array([np.mean(x_coords), np.mean(y_coords)])

            # Center the points
            x_centered = x_coords - centroid[0]
            y_centered = y_coords - centroid[1]

            coordinates = np.column_stack([x_centered, y_centered])

            # Check for degenerate case
            if coordinates.shape[0] < 2:
                return None

            # Use SVD to find the best fit line
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

        except (np.linalg.LinAlgError, ValueError, IndexError) as e:
            return None

    def detect_line_segments(self, edge_points):
        """
        Complete line detection pipeline - Main method called by vanishing point detector

        Args:
            edge_points (list): List of (x, y, orientation) tuples from edge detection

        Returns:
            list: List of (line_equation, points) tuples
        """
        if not edge_points:
            return []

        # Step 1: Group parallel lines
        line_groups = self.group_parallel_lines(edge_points)

        # Step 2: Fit lines to groups
        fitted_lines = self.fit_lines_to_groups(line_groups)

        # Step 3: Store final debug info
        self.debug_info.update({
            'total_edge_points': len(edge_points),
            'final_line_count': len(fitted_lines),
            'average_points_per_line': np.mean([len(points) for _, points in fitted_lines]) if fitted_lines else 0
        })

        return fitted_lines

    def filter_parallel_groups(self, fitted_lines, min_groups=None):
        """
        Filter and group parallel lines for better vanishing point detection

        Args:
            fitted_lines (list): List of (line_equation, points) tuples
            min_groups (int): Minimum number of parallel groups needed

        Returns:
            list: List of parallel line groups
        """
        if min_groups is None:
            min_groups = getattr(self.config, 'MIN_PARALLEL_LINE_GROUPS', 2)

        # Group lines by similar orientations
        parallel_groups = defaultdict(list)

        for line_eq, points in fitted_lines:
            a, b, c = line_eq

            # Calculate line angle
            angle = math.atan2(abs(a), abs(b))  # Angle with respect to x-axis
            angle_tolerance = getattr(self.config, 'ANGLE_TOLERANCE', 0.2)
            angle_key = round(angle / angle_tolerance) * angle_tolerance

            parallel_groups[angle_key].append((line_eq, points))

        # Filter groups with minimum number of lines
        valid_groups = []
        min_lines_per_group = getattr(self.config, 'MIN_LINES_PER_GROUP', 2)

        for angle, lines in parallel_groups.items():
            if len(lines) >= min_lines_per_group:
                valid_groups.append(lines)

        return valid_groups

    def get_line_statistics(self, fitted_lines):
        """
        Get comprehensive statistics about detected lines

        Args:
            fitted_lines (list): List of fitted lines

        Returns:
            dict: Statistics dictionary
        """
        if not fitted_lines:
            return {
                'total_lines': 0,
                'avg_points_per_line': 0,
                'avg_line_length': 0,
                'angle_spread': 0,
                'min_points': 0,
                'max_points': 0
            }

        line_lengths = []
        point_counts = []
        angles = []

        for line_eq, points in fitted_lines:
            point_counts.append(len(points))

            # Calculate approximate line length from point spread
            if len(points) >= 2:
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                length = math.sqrt((max(x_coords) - min(x_coords))**2 + 
                                 (max(y_coords) - min(y_coords))**2)
                line_lengths.append(length)

                # Calculate line angle
                a, b, c = line_eq
                angle = math.atan2(abs(a), abs(b))
                angles.append(angle)

        stats = {
            'total_lines': len(fitted_lines),
            'avg_points_per_line': np.mean(point_counts) if point_counts else 0,
            'avg_line_length': np.mean(line_lengths) if line_lengths else 0,
            'angle_spread': np.std(angles) if len(angles) > 1 else 0,
            'min_points': min(point_counts) if point_counts else 0,
            'max_points': max(point_counts) if point_counts else 0,
            'total_points_used': sum(point_counts)
        }

        return stats

    def remove_outlier_lines(self, fitted_lines, max_distance_threshold=None):
        """
        Remove lines that are likely outliers based on point consistency

        Args:
            fitted_lines (list): List of fitted lines
            max_distance_threshold (float): Maximum allowed distance from line

        Returns:
            list: Filtered lines with outliers removed
        """
        if max_distance_threshold is None:
            max_distance_threshold = getattr(self.config, 'MAX_LINE_DISTANCE_THRESHOLD', 5)

        filtered_lines = []

        for line_eq, points in fitted_lines:
            # Calculate distances from points to line
            from utils.geometry import point_to_line_distance

            inlier_points = []
            for point in points:
                distance = point_to_line_distance((point[0], point[1]), line_eq)
                if distance <= max_distance_threshold:
                    inlier_points.append(point)

            # Keep line if it has enough inlier points
            min_points = getattr(self.config, 'MIN_LINE_POINTS', 10)
            if len(inlier_points) >= min_points:
                filtered_lines.append((line_eq, inlier_points))

        return filtered_lines

    def get_debug_info(self):
        """Get debug information from last detection"""
        return self.debug_info.copy()

    def print_debug_info(self):
        """Print formatted debug information"""
        if not self.debug_info:
            print("No debug information available")
            return

        print("🔍 Line Detection Debug Information:")
        print("=" * 40)
        for key, value in self.debug_info.items():
            if isinstance(value, list):
                print(f"{key}: {len(value)} items")
                if value and len(value) <= 10:  # Show small lists
                    print(f"  Values: {value}")
            elif isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")


# Utility functions for line detection
def detect_lines_simple(edge_points, angle_tolerance=0.2, min_points=10):
    """
    Simple line detection function for quick usage

    Args:
        edge_points (list): List of edge points
        angle_tolerance (float): Angle tolerance for grouping
        min_points (int): Minimum points per line

    Returns:
        list: List of fitted lines
    """
    class SimpleConfig:
        ANGLE_TOLERANCE = angle_tolerance
        MIN_LINE_POINTS = min_points
        LINE_FITTING_METHOD = 'svd'
        MIN_PARALLEL_LINE_GROUPS = 2
        MIN_LINES_PER_GROUP = 2
        MAX_LINE_DISTANCE_THRESHOLD = 5

    detector = LineDetector(SimpleConfig())
    return detector.detect_line_segments(edge_points)


def group_lines_by_orientation(edge_points, angle_tolerance=0.2):
    """
    Group edge points by orientation (standalone function)

    Args:
        edge_points (list): List of (x, y, orientation) tuples
        angle_tolerance (float): Tolerance for grouping similar angles

    Returns:
        dict: Dictionary mapping angle bins to point lists
    """
    angle_groups = defaultdict(list)

    for x, y, angle in edge_points:
        norm_angle = angle % np.pi
        angle_key = round(norm_angle / angle_tolerance) * angle_tolerance
        angle_groups[angle_key].append((x, y, angle))

    return dict(angle_groups)


def fit_single_line_to_points(points, method='svd'):
    """
    Fit a single line to a set of points

    Args:
        points (list): List of point tuples
        method (str): Fitting method ('svd' or 'least_squares')

    Returns:
        tuple or None: Line coefficients (a, b, c)
    """
    if method == 'svd':
        detector = LineDetector()
        return detector._fit_line_svd(points)
    else:
        # Could implement other methods here
        detector = LineDetector()
        return detector._fit_line_svd(points)


def calculate_line_quality_score(line_eq, points):
    """
    Calculate a quality score for a fitted line

    Args:
        line_eq (tuple): Line equation coefficients
        points (list): Points used to fit the line

    Returns:
        float: Quality score (higher is better)
    """
    if not points or not line_eq:
        return 0.0

    from utils.geometry import point_to_line_distance

    # Calculate average distance from points to line
    distances = []
    for point in points:
        distance = point_to_line_distance((point[0], point[1]), line_eq)
        distances.append(distance)

    avg_distance = np.mean(distances)

    # Score based on inverse of average distance and number of points
    point_score = min(len(points) / 50.0, 1.0)  # Normalize to [0,1]
    distance_score = 1.0 / (1.0 + avg_distance)  # Inverse distance

    return (point_score + distance_score) / 2.0
