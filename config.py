"""
Configuration module for vanishing point detection
Contains all parameters and presets for the detection system
"""

import json


class VanishingPointConfig:
    """
    Comprehensive configuration class for vanishing point detection
    """

    # ==== EDGE DETECTION PARAMETERS ====
    EDGE_THRESHOLD = 0.5              # Threshold for edge detection (0.0-1.0)
    GAUSSIAN_SIGMA = 1.0              # Sigma for Gaussian smoothing
    USE_NON_MAX_SUPPRESSION = False   # Apply non-maximum suppression
    MIN_EDGE_STRENGTH_PERCENTILE = 75 # Minimum edge strength percentile
    MAX_EDGE_POINTS = 10000           # Maximum edge points to process

    # ==== LINE DETECTION PARAMETERS ====
    ANGLE_TOLERANCE = 0.2             # Tolerance for grouping parallel lines (radians)
    MIN_LINE_POINTS = 10              # Minimum points required to form a line
    LINE_FITTING_METHOD = 'svd'       # Method for line fitting
    MAX_LINE_DISTANCE_THRESHOLD = 5   # Max distance from point to line

    # ==== CLUSTERING PARAMETERS ====
    DBSCAN_EPS = 50                   # DBSCAN epsilon parameter
    DBSCAN_MIN_SAMPLES = 2            # DBSCAN minimum samples
    IMAGE_BOUNDARY_FACTOR = 2.0       # Factor for filtering intersections
    MIN_INTERSECTIONS_FOR_VP = 3      # Min intersections for vanishing point

    # ==== PARALLEL LINE DETECTION ====
    MIN_PARALLEL_LINE_GROUPS = 2      # Min parallel line groups needed
    PARALLEL_ANGLE_THRESHOLD = 0.1    # Threshold for parallel lines
    MIN_LINES_PER_GROUP = 2           # Min lines per parallel group

    # ==== VISUALIZATION PARAMETERS ====
    VIZ_FIGURE_SIZE = (15, 7)         # Figure size for plots
    VIZ_LINE_WIDTH = 2                # Line width for visualization
    VIZ_LINE_ALPHA = 0.8              # Alpha transparency for lines
    VP_MARKER_SIZE = 200              # Vanishing point marker size
    VP_MARKER_COLOR = 'red'           # Vanishing point color
    INTERSECTION_MARKER_SIZE = 30     # Intersection marker size
    INTERSECTION_MARKER_COLOR = 'lightblue'  # Intersection color

    # ==== DEBUG AND PERFORMANCE ====
    DEBUG_MODE = False                # Enable debug output
    VERBOSE_OUTPUT = True             # Enable verbose console output
    SAVE_INTERMEDIATE_RESULTS = False # Save intermediate results
    PROCESSING_TIMEOUT = 30           # Max processing time in seconds

    # ==== IMAGE PREPROCESSING ====
    AUTO_RESIZE_LARGE_IMAGES = True   # Auto-resize large images
    MAX_IMAGE_WIDTH = 1920            # Maximum image width
    MAX_IMAGE_HEIGHT = 1080           # Maximum image height

    @classmethod
    def create_custom_config(cls, **kwargs):
        """Create custom configuration with overridden parameters"""
        config = cls()
        for key, value in kwargs.items():
            attr_name = key.upper()
            if hasattr(config, attr_name):
                setattr(config, attr_name, value)
            else:
                print(f"Warning: Unknown parameter '{key}'")
        return config

    def get_edge_detection_params(self):
        """Get edge detection specific parameters"""
        return {
            'edge_threshold': self.EDGE_THRESHOLD,
            'gaussian_sigma': self.GAUSSIAN_SIGMA,
            'use_non_max_suppression': self.USE_NON_MAX_SUPPRESSION,
            'min_edge_strength_percentile': self.MIN_EDGE_STRENGTH_PERCENTILE,
            'max_edge_points': self.MAX_EDGE_POINTS
        }

    def get_line_detection_params(self):
        """Get line detection specific parameters"""
        return {
            'angle_tolerance': self.ANGLE_TOLERANCE,
            'min_line_points': self.MIN_LINE_POINTS,
            'line_fitting_method': self.LINE_FITTING_METHOD,
            'max_line_distance_threshold': self.MAX_LINE_DISTANCE_THRESHOLD
        }

    def get_clustering_params(self):
        """Get clustering specific parameters"""
        return {
            'dbscan_eps': self.DBSCAN_EPS,
            'dbscan_min_samples': self.DBSCAN_MIN_SAMPLES,
            'image_boundary_factor': self.IMAGE_BOUNDARY_FACTOR,
            'min_intersections_for_vp': self.MIN_INTERSECTIONS_FOR_VP
        }

    def print_config(self):
        """Print current configuration"""
        print("🔧 Vanishing Point Detection Configuration:")
        print("=" * 50)

        sections = [
            ("Edge Detection", self.get_edge_detection_params()),
            ("Line Detection", self.get_line_detection_params()),
            ("Clustering", self.get_clustering_params())
        ]

        for section_name, params in sections:
            print(f"\n{section_name}:")
            for key, value in params.items():
                print(f"  {key}: {value}")


class ConfigPresets:
    """Predefined configuration presets for different scenarios"""

    @staticmethod
    def high_precision():
        """High precision detection (slower but more accurate)"""
        return VanishingPointConfig.create_custom_config(
            edge_threshold=0.3,
            angle_tolerance=0.1,
            min_line_points=20,
            dbscan_eps=30,
            gaussian_sigma=0.8,
            use_non_max_suppression=True
        )

    @staticmethod
    def fast_detection():
        """Fast detection (quicker but less precise)"""
        return VanishingPointConfig.create_custom_config(
            edge_threshold=0.6,
            angle_tolerance=0.3,
            min_line_points=8,
            dbscan_eps=80,
            max_edge_points=5000
        )

    @staticmethod
    def noisy_images():
        """Optimized for noisy/blurry images"""
        return VanishingPointConfig.create_custom_config(
            edge_threshold=0.4,
            gaussian_sigma=1.5,
            min_line_points=15,
            dbscan_eps=60,
            min_edge_strength_percentile=80,
            use_non_max_suppression=True
        )

    @staticmethod
    def architectural():
        """Optimized for architectural images"""
        return VanishingPointConfig.create_custom_config(
            edge_threshold=0.4,
            angle_tolerance=0.15,
            min_line_points=12,
            parallel_angle_threshold=0.08,
            dbscan_eps=40
        )

    @staticmethod
    def road_scenes():
        """Optimized for road/street scenes"""
        return VanishingPointConfig.create_custom_config(
            edge_threshold=0.5,
            angle_tolerance=0.25,
            min_line_points=10,
            image_boundary_factor=3.0,
            dbscan_eps=70
        )


# Default configuration instance
DEFAULT_CONFIG = VanishingPointConfig()

# Export commonly used configurations
FAST_CONFIG = ConfigPresets.fast_detection()
HIGH_PRECISION_CONFIG = ConfigPresets.high_precision()
NOISY_IMAGE_CONFIG = ConfigPresets.noisy_images()
ARCHITECTURAL_CONFIG = ConfigPresets.architectural()
ROAD_SCENE_CONFIG = ConfigPresets.road_scenes()
