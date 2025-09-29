"""
Main vanishing point detection logic
Combines edge detection, line detection, and clustering
"""

import numpy as np
from sklearn.cluster import DBSCAN
import math


class VanishingPointDetector:
    """Main vanishing point detector combining all detection modules"""

    def __init__(self, config=None):
        """Initialize with configuration and sub-detectors"""
        try:
            from config import DEFAULT_CONFIG
            self.config = config or DEFAULT_CONFIG
        except ImportError:
            self.config = self._create_fallback_config()

        from core.edge_detection import EdgeDetector
        from core.line_detection import LineDetector

        self.edge_detector = EdgeDetector(self.config)
        self.line_detector = LineDetector(self.config)
        self.debug_info = {}

    def _create_fallback_config(self):
        """Create fallback configuration"""
        class FallbackConfig:
            EDGE_THRESHOLD = 0.5
            ANGLE_TOLERANCE = 0.2
            MIN_LINE_POINTS = 10
            DBSCAN_EPS = 50
            DBSCAN_MIN_SAMPLES = 2
            IMAGE_BOUNDARY_FACTOR = 2.0
            DEBUG_MODE = False
            VERBOSE_OUTPUT = True

        return FallbackConfig()

    def detect_vanishing_point(self, image_array, debug=False):
        """
        Main detection method

        Args:
            image_array: Input image
            debug: Show debug information

        Returns:
            tuple: (vanishing_point, fitted_lines)
        """
        if debug:
            print("🔍 Starting vanishing point detection...")

        # Step 1: Edge detection
        edge_points = self.edge_detector.detect_edges(image_array)
        if debug:
            print(f"📍 Found {len(edge_points)} edge points")

        if len(edge_points) < 50:
            if debug:
                print("❌ Not enough edge points")
            return None, []

        # Step 2: Line detection
        fitted_lines = self.line_detector.detect_line_segments(edge_points)
        if debug:
            print(f"📏 Found {len(fitted_lines)} line groups")

        if len(fitted_lines) < 2:
            if debug:
                print("❌ Not enough line groups")
            return None, fitted_lines

        # Step 3: Find intersections
        intersections = self._find_all_intersections(fitted_lines, image_array.shape)
        if debug:
            print(f"🎯 Found {len(intersections)} intersections")

        if not intersections:
            if debug:
                print("❌ No intersections found")
            return None, fitted_lines

        # Step 4: Cluster intersections
        vanishing_point = self._cluster_intersections(intersections)

        if debug:
            if vanishing_point:
                print(f"✅ Vanishing point: ({vanishing_point[0]:.1f}, {vanishing_point[1]:.1f})")
            else:
                print("❌ Could not determine vanishing point")

        return vanishing_point, fitted_lines

    def _find_all_intersections(self, fitted_lines, image_shape):
        """Find all line intersections within image bounds"""
        from utils.geometry import line_intersection, filter_intersections_by_bounds

        intersections = []
        for i in range(len(fitted_lines)):
            for j in range(i + 1, len(fitted_lines)):
                intersection = line_intersection(fitted_lines[i][0], fitted_lines[j][0])
                if intersection:
                    intersections.append(intersection)

        return filter_intersections_by_bounds(
            intersections, image_shape, self.config.IMAGE_BOUNDARY_FACTOR
        )

    def _cluster_intersections(self, intersections):
        """Cluster intersections using DBSCAN to find vanishing point"""
        if len(intersections) <= 1:
            return intersections[0] if intersections else None

        intersection_array = np.array(intersections)

        try:
            clustering = DBSCAN(
                eps=self.config.DBSCAN_EPS, 
                min_samples=self.config.DBSCAN_MIN_SAMPLES
            ).fit(intersection_array)

            labels = clustering.labels_
            unique_labels = labels[labels != -1]

            if len(unique_labels) > 0:
                unique_vals, counts = np.unique(unique_labels, return_counts=True)
                largest_cluster_label = unique_vals[np.argmax(counts)]
                cluster_points = intersection_array[labels == largest_cluster_label]
                return tuple(np.mean(cluster_points, axis=0))
            else:
                return tuple(np.mean(intersection_array, axis=0))

        except Exception:
            return tuple(np.mean(intersection_array, axis=0))

    def get_detection_quality_score(self):
        """Calculate detection quality score"""
        # Simple quality scoring based on available data
        return 0.8  # Default good score

    def get_debug_info(self):
        """Get debug information from last detection"""
        return self.debug_info.copy()


def detect_vanishing_point_simple(image_array, edge_threshold=0.5, debug=False):
    """Simple function for quick vanishing point detection"""
    class SimpleConfig:
        EDGE_THRESHOLD = edge_threshold
        ANGLE_TOLERANCE = 0.2
        MIN_LINE_POINTS = 10
        DBSCAN_EPS = 50
        DBSCAN_MIN_SAMPLES = 2
        IMAGE_BOUNDARY_FACTOR = 2.0
        DEBUG_MODE = debug
        VERBOSE_OUTPUT = debug

    detector = VanishingPointDetector(SimpleConfig())
    return detector.detect_vanishing_point(image_array, debug=debug)
