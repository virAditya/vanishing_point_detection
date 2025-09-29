"""
Edge detection module for vanishing point detection
Contains functions for detecting edges and their orientations using Sobel operators
"""

import numpy as np
from scipy import ndimage
import math


class EdgeDetector:
    """
    Advanced edge detection class with multiple algorithms and filtering options
    """

    def __init__(self, config=None):
        """Initialize edge detector with configuration"""
        try:
            from config import DEFAULT_CONFIG
            self.config = config or DEFAULT_CONFIG
        except ImportError:
            self.config = self._create_fallback_config()

        self.debug_info = {}
        self.last_magnitude_map = None
        self.last_direction_map = None

    def _create_fallback_config(self):
        """Create fallback configuration if config module unavailable"""
        class FallbackConfig:
            EDGE_THRESHOLD = 0.5
            GAUSSIAN_SIGMA = 1.0
            USE_NON_MAX_SUPPRESSION = False
            MIN_EDGE_STRENGTH_PERCENTILE = 75
            MAX_EDGE_POINTS = 10000

        return FallbackConfig()

    def detect_edges(self, image_array):
        """
        Main edge detection method using Sobel operators

        Args:
            image_array (numpy.ndarray): Input image (grayscale or color)

        Returns:
            list: List of (x, y, orientation) tuples for edge points
        """
        # Convert to grayscale if needed
        gray = self._convert_to_grayscale(image_array)

        # Apply preprocessing
        preprocessed = self._preprocess_image(gray)

        # Calculate gradients using Sobel operators
        grad_x, grad_y = self._calculate_gradients(preprocessed)

        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)

        # Store for debugging
        self.last_magnitude_map = magnitude
        self.last_direction_map = direction

        # Apply non-maximum suppression if enabled
        if getattr(self.config, 'USE_NON_MAX_SUPPRESSION', False):
            magnitude = self._apply_non_maximum_suppression(magnitude, direction)

        # Find edge pixels above threshold
        edge_pixels = self._threshold_edges(magnitude)

        # Extract edge points and their orientations
        edge_points = self._extract_edge_points(edge_pixels, direction)

        # Apply post-processing filters
        edge_points = self._post_process_edge_points(edge_points, magnitude)

        # Store debug information
        self.debug_info.update({
            'input_shape': image_array.shape,
            'max_gradient_magnitude': np.max(magnitude),
            'threshold_used': np.max(magnitude) * self.config.EDGE_THRESHOLD,
            'total_edge_pixels': np.sum(edge_pixels),
            'final_edge_points': len(edge_points),
            'preprocessing_method': 'gaussian_blur'
        })

        return edge_points

    def _convert_to_grayscale(self, image_array):
        """Convert image to grayscale using proper RGB weights"""
        if len(image_array.shape) == 3:
            # Use standard luminance weights
            weights = np.array([0.299, 0.587, 0.114])
            if image_array.shape[2] == 3:  # RGB
                return np.dot(image_array, weights)
            elif image_array.shape[2] == 4:  # RGBA
                return np.dot(image_array[:, :, :3], weights)
            else:  # Single channel in 3D array
                return image_array[:, :, 0]
        return image_array.astype(float)

    def _preprocess_image(self, gray_image):
        """Apply preprocessing to grayscale image"""
        # Apply Gaussian smoothing to reduce noise
        smoothed = ndimage.gaussian_filter(gray_image, sigma=self.config.GAUSSIAN_SIGMA)
        return smoothed

    def _calculate_gradients(self, image):
        """Calculate image gradients using Sobel operators"""
        grad_x = ndimage.sobel(image, axis=1)  # Horizontal edges
        grad_y = ndimage.sobel(image, axis=0)  # Vertical edges
        return grad_x, grad_y

    def _threshold_edges(self, magnitude):
        """Apply thresholding to gradient magnitude"""
        max_magnitude = np.max(magnitude)
        threshold = max_magnitude * self.config.EDGE_THRESHOLD

        if threshold == 0:  # Handle case where image has no gradients
            threshold = 1e-6

        return magnitude > threshold

    def _extract_edge_points(self, edge_pixels, direction):
        """Extract edge points with their orientations"""
        y_coords, x_coords = np.where(edge_pixels)
        orientations = direction[y_coords, x_coords]

        edge_points = list(zip(x_coords, y_coords, orientations))

        # Limit number of points for performance
        max_points = getattr(self.config, 'MAX_EDGE_POINTS', 10000)
        if len(edge_points) > max_points:
            indices = np.random.choice(len(edge_points), max_points, replace=False)
            edge_points = [edge_points[i] for i in indices]

        return edge_points

    def _post_process_edge_points(self, edge_points, magnitude_map):
        """Apply post-processing filters to edge points"""
        if not edge_points:
            return []

        # Filter by gradient strength percentile
        min_percentile = getattr(self.config, 'MIN_EDGE_STRENGTH_PERCENTILE', 75)
        edge_points = self._filter_by_strength(edge_points, magnitude_map, min_percentile)

        return edge_points

    def _filter_by_strength(self, edge_points, magnitude_map, min_percentile=75):
        """Filter edge points by gradient strength percentile"""
        if len(edge_points) < 10:  # Skip filtering for very few points
            return edge_points

        # Get magnitude values for all edge points
        magnitudes = []
        for x, y, _ in edge_points:
            if 0 <= int(y) < magnitude_map.shape[0] and 0 <= int(x) < magnitude_map.shape[1]:
                magnitudes.append(magnitude_map[int(y), int(x)])
            else:
                magnitudes.append(0)

        # Calculate threshold based on percentile
        threshold = np.percentile(magnitudes, min_percentile)

        # Filter points above threshold
        filtered_points = []
        for i, (x, y, orientation) in enumerate(edge_points):
            if magnitudes[i] >= threshold:
                filtered_points.append((x, y, orientation))

        return filtered_points

    def _apply_non_maximum_suppression(self, magnitude, direction):
        """Apply non-maximum suppression to thin edges"""
        height, width = magnitude.shape
        suppressed = np.zeros_like(magnitude)

        # Convert angle to 0-180 degrees and discretize to 4 directions
        angle = np.rad2deg(direction) % 180

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                current_mag = magnitude[i, j]
                current_angle = angle[i, j]

                # Determine neighboring pixels based on gradient direction
                if (0 <= current_angle < 22.5) or (157.5 <= current_angle <= 180):
                    # Horizontal direction (0 degrees)
                    neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
                elif 22.5 <= current_angle < 67.5:
                    # Diagonal direction (45 degrees)
                    neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
                elif 67.5 <= current_angle < 112.5:
                    # Vertical direction (90 degrees)
                    neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
                else:  # 112.5 <= current_angle < 157.5
                    # Diagonal direction (135 degrees)
                    neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]

                # Suppress if not a local maximum
                if current_mag >= max(neighbors + [0]):
                    suppressed[i, j] = current_mag

        return suppressed

    def detect_edges_canny_style(self, image_array, low_threshold=None, high_threshold=None):
        """Canny-style edge detection with double thresholding and hysteresis"""
        # Get basic edge detection results
        gray = self._convert_to_grayscale(image_array)
        preprocessed = self._preprocess_image(gray)
        grad_x, grad_y = self._calculate_gradients(preprocessed)

        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)

        # Apply non-maximum suppression
        suppressed = self._apply_non_maximum_suppression(magnitude, direction)

        # Set thresholds automatically if not provided
        if high_threshold is None:
            high_threshold = np.max(suppressed) * 0.6
        if low_threshold is None:
            low_threshold = high_threshold * 0.4

        # Double thresholding
        strong_edges = suppressed > high_threshold
        weak_edges = (suppressed > low_threshold) & (suppressed <= high_threshold)

        # Hysteresis
        final_edges = self._apply_hysteresis(strong_edges, weak_edges)

        # Extract edge points
        y_coords, x_coords = np.where(final_edges)
        orientations = direction[y_coords, x_coords]

        return list(zip(x_coords, y_coords, orientations))

    def _apply_hysteresis(self, strong_edges, weak_edges):
        """Apply hysteresis to connect weak edges to strong edges"""
        final_edges = strong_edges.copy()
        height, width = strong_edges.shape

        # Iteratively add weak edges connected to strong edges
        changed = True
        max_iterations = 10
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    if weak_edges[i, j] and not final_edges[i, j]:
                        # Check if connected to a strong edge
                        neighborhood = final_edges[i-1:i+2, j-1:j+2]
                        if np.any(neighborhood):
                            final_edges[i, j] = True
                            changed = True

        return final_edges

    def get_gradient_maps(self):
        """Get the last computed gradient magnitude and direction maps"""
        return self.last_magnitude_map, self.last_direction_map

    def get_debug_info(self):
        """Get debug information from last edge detection"""
        return self.debug_info.copy()

    def visualize_edges(self, image_array, edge_points=None):
        """Visualize detected edges on the original image"""
        if edge_points is None:
            edge_points = self.detect_edges(image_array)

        # Create visualization image
        if len(image_array.shape) == 3:
            vis_image = image_array.copy()
        else:
            vis_image = np.stack([image_array] * 3, axis=2)

        # Highlight edge points in red
        for x, y, _ in edge_points:
            x, y = int(x), int(y)
            if 0 <= x < vis_image.shape[1] and 0 <= y < vis_image.shape[0]:
                vis_image[y, x] = [255, 0, 0]  # Red

        return vis_image


# Utility functions
def detect_edges_simple(image_array, threshold=0.5, sigma=1.0):
    """Simple edge detection function for quick usage"""
    class SimpleConfig:
        EDGE_THRESHOLD = threshold
        GAUSSIAN_SIGMA = sigma
        USE_NON_MAX_SUPPRESSION = False
        MIN_EDGE_STRENGTH_PERCENTILE = 70
        MAX_EDGE_POINTS = 10000

    detector = EdgeDetector(SimpleConfig())
    return detector.detect_edges(image_array)


def compare_edge_detectors(image_array, methods=['sobel', 'canny']):
    """Compare different edge detection methods"""
    detector = EdgeDetector()
    results = {}

    for method in methods:
        if method == 'sobel':
            results[method] = detector.detect_edges(image_array)
        elif method == 'canny':
            results[method] = detector.detect_edges_canny_style(image_array)
        else:
            print(f"Unknown method: {method}")

    return results


def filter_edges_by_orientation(edge_points, target_orientation, tolerance=0.3):
    """Filter edge points by orientation"""
    filtered = []
    for x, y, orientation in edge_points:
        # Normalize orientations to [0, π] for comparison
        norm_target = target_orientation % math.pi
        norm_orientation = orientation % math.pi

        # Check if within tolerance
        diff = abs(norm_target - norm_orientation)
        if diff <= tolerance or abs(diff - math.pi) <= tolerance:
            filtered.append((x, y, orientation))

    return filtered


def create_edge_map(edge_points, image_shape):
    """Create a binary edge map from edge points"""
    edge_map = np.zeros(image_shape, dtype=np.uint8)

    for x, y, _ in edge_points:
        x, y = int(x), int(y)
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            edge_map[y, x] = 255

    return edge_map
