"""
Visualization module for vanishing point detection results
UPDATED: Added clean vanishing point-only visualization
"""

import matplotlib.pyplot as plt
import numpy as np


class VanishingPointVisualizer:
    """Comprehensive visualization class with clean VP display option"""

    def __init__(self, config=None):
        """Initialize visualizer with configuration"""
        try:
            from config import DEFAULT_CONFIG
            self.config = config or DEFAULT_CONFIG
        except ImportError:
            self.config = self._create_fallback_config()

    def _create_fallback_config(self):
        """Create fallback configuration"""
        class FallbackConfig:
            VIZ_FIGURE_SIZE = (15, 7)
            VIZ_LINE_WIDTH = 2
            VIZ_LINE_ALPHA = 0.8
            VP_MARKER_SIZE = 200
            VP_MARKER_COLOR = 'red'
            INTERSECTION_MARKER_SIZE = 30
            INTERSECTION_MARKER_COLOR = 'lightblue'

        return FallbackConfig()

    def plot_results(self, image_array, fitted_lines, intersections, vanishing_point):
        """Create comprehensive visualization of detection results"""
        fig, axes = plt.subplots(1, 2, figsize=self.config.VIZ_FIGURE_SIZE)

        # Plot original image
        self._plot_original_image(axes[0], image_array)

        # Plot detection results
        self._plot_detection_results(axes[1], image_array, fitted_lines, 
                                   intersections, vanishing_point)

        plt.tight_layout()
        plt.show()

    def _plot_original_image(self, ax, image_array):
        """Plot the original input image"""
        if len(image_array.shape) == 3:
            ax.imshow(image_array)
        else:
            ax.imshow(image_array, cmap='gray')

        ax.set_title('Original Image', fontsize=14, fontweight='bold')
        ax.axis('off')

    def _plot_detection_results(self, ax, image_array, fitted_lines, intersections, vanishing_point):
        """Plot detection results with lines, intersections, and vanishing point"""
        # Show original image as background
        if len(image_array.shape) == 3:
            ax.imshow(image_array, alpha=0.7)
        else:
            ax.imshow(image_array, cmap='gray', alpha=0.7)

        h, w = image_array.shape[:2]

        # Draw detected lines in different colors
        if fitted_lines:
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(fitted_lines), 1)))

            for i, (line_eq, points) in enumerate(fitted_lines):
                endpoints = self._calculate_line_endpoints(line_eq, image_array.shape)
                (x1, y1), (x2, y2) = endpoints

                ax.plot([x1, x2], [y1, y2], 
                       color=colors[i % len(colors)], 
                       linewidth=self.config.VIZ_LINE_WIDTH, 
                       alpha=self.config.VIZ_LINE_ALPHA,
                       label=f'Line {i+1}')

        # Draw intersection points
        if intersections:
            intersection_array = np.array(intersections)
            ax.scatter(intersection_array[:, 0], intersection_array[:, 1], 
                      c=self.config.INTERSECTION_MARKER_COLOR, 
                      s=self.config.INTERSECTION_MARKER_SIZE, 
                      alpha=0.7, edgecolor='blue', linewidth=1,
                      label='Intersections', zorder=4)

        # Draw vanishing point
        if vanishing_point:
            ax.scatter(vanishing_point[0], vanishing_point[1], 
                      c=self.config.VP_MARKER_COLOR, 
                      s=self.config.VP_MARKER_SIZE, 
                      marker='*', edgecolor='darkred', linewidth=2,
                      label='Vanishing Point', zorder=5)

            # Add coordinate annotation
            ax.annotate(f'VP: ({int(vanishing_point[0])}, {int(vanishing_point[1])})', 
                       xy=vanishing_point, xytext=(20, 20), 
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                       fontsize=11, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        ax.set_title('Vanishing Point Detection Results', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlim(-w//4, w + w//4)
        ax.set_ylim(h + h//4, -h//4)  # Invert y-axis
        ax.grid(True, alpha=0.3)

    def _calculate_line_endpoints(self, line_coeffs, image_shape, extension_factor=1.5):
        """Calculate line endpoints for visualization"""
        a, b, c = line_coeffs
        h, w = image_shape[:2]

        x_min = -int(w * (extension_factor - 1) / 2)
        x_max = int(w * extension_factor)
        y_min = -int(h * (extension_factor - 1) / 2)
        y_max = int(h * extension_factor)

        if abs(b) > abs(a):  # More horizontal line
            x1, x2 = x_min, x_max
            y1 = (-c - a * x1) / b if abs(b) > 1e-10 else 0
            y2 = (-c - a * x2) / b if abs(b) > 1e-10 else 0
        else:  # More vertical line
            y1, y2 = y_min, y_max
            x1 = (-c - b * y1) / a if abs(a) > 1e-10 else 0
            x2 = (-c - b * y2) / a if abs(a) > 1e-10 else 0

        return ((x1, y1), (x2, y2))

    def _prepare_edge_image(self, image_array):
        """Prepare image for edge visualization - handles all formats"""
        if len(image_array.shape) == 2:
            return np.stack([image_array] * 3, axis=2).astype(np.uint8)
        elif len(image_array.shape) == 3:
            if image_array.shape[2] == 1:
                single_channel = image_array[:, :, 0]
                return np.stack([single_channel] * 3, axis=2).astype(np.uint8)
            elif image_array.shape[2] == 3:
                return image_array.astype(np.uint8).copy()
            elif image_array.shape[2] == 4:
                return image_array[:, :, :3].astype(np.uint8).copy()
            else:
                return np.stack([image_array[:, :, 0]] * 3, axis=2).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported image shape: {image_array.shape}")

    def plot_vanishing_point_only(self, image_array, vanishing_point, dot_size=400, dot_color='red'):
        """
        NEW: Show only the original image with vanishing point marked as a thick red dot

        Args:
            image_array: Original image
            vanishing_point: Vanishing point coordinates (x, y)
            dot_size: Size of the vanishing point marker (default: 400 for thick dot)
            dot_color: Color of the vanishing point marker
        """
        print("\n🎯 Displaying clean vanishing point visualization...")

        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        # Show original image
        if len(image_array.shape) == 3:
            ax.imshow(image_array)
        else:
            ax.imshow(image_array, cmap='gray')

        # Draw vanishing point as a thick red dot with white border
        if vanishing_point:
            # Main red dot
            ax.scatter(vanishing_point[0], vanishing_point[1], 
                      c=dot_color, s=dot_size, 
                      marker='o',  # Circle marker
                      edgecolor='white', linewidth=4,  # Thick white border
                      alpha=0.95, zorder=5)

            # Add a smaller inner dot for better visibility
            ax.scatter(vanishing_point[0], vanishing_point[1], 
                      c='darkred', s=dot_size//4, 
                      marker='o', alpha=1.0, zorder=6)

            # Add coordinate text with clean styling
            ax.annotate(f'Vanishing Point\n({int(vanishing_point[0])}, {int(vanishing_point[1])})', 
                       xy=vanishing_point, xytext=(40, 40), 
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=1.0', 
                               facecolor='white', alpha=0.95, 
                               edgecolor='red', linewidth=2),
                       fontsize=14, fontweight='bold', color='darkred',
                       ha='center',
                       arrowprops=dict(arrowstyle='->', color='red', lw=3))

        ax.set_title('Detected Vanishing Point', fontsize=18, fontweight='bold', pad=25)
        ax.axis('off')  # Remove axes for clean look

        plt.tight_layout()
        plt.show()

    def plot_step_by_step(self, image_array, edge_points, fitted_lines, intersections, vanishing_point):
        """Create step-by-step visualization, then show clean VP visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # Step 1: Original image
        if len(image_array.shape) == 3:
            axes[0].imshow(image_array)
        else:
            axes[0].imshow(image_array, cmap='gray')
        axes[0].set_title('1. Original Image', fontweight='bold')
        axes[0].axis('off')

        # Step 2: Edge detection - FIXED for channel compatibility
        try:
            edge_image = self._prepare_edge_image(image_array)

            if edge_points:
                # Sample points to avoid overcrowding
                max_points = 2000
                if len(edge_points) > max_points:
                    step = len(edge_points) // max_points
                    sample_points = edge_points[::step]
                else:
                    sample_points = edge_points

                # Set edge pixels to red
                for x, y, _ in sample_points:
                    x, y = int(x), int(y)
                    if (0 <= x < edge_image.shape[1] and 
                        0 <= y < edge_image.shape[0]):
                        edge_image[y, x] = [255, 0, 0]  # Red color

            axes[1].imshow(edge_image)
            axes[1].set_title(f'2. Edge Detection ({len(edge_points)} points)', fontweight='bold')
            axes[1].axis('off')

        except Exception as e:
            if len(image_array.shape) == 3:
                axes[1].imshow(image_array)
            else:
                axes[1].imshow(image_array, cmap='gray')
            axes[1].set_title(f'2. Edge Detection (Visualization Error)', fontweight='bold')
            axes[1].axis('off')

        # Step 3: Line fitting
        if len(image_array.shape) == 3:
            axes[2].imshow(image_array, alpha=0.7)
        else:
            axes[2].imshow(image_array, cmap='gray', alpha=0.7)

        if fitted_lines:
            colors = plt.cm.Set3(np.linspace(0, 1, len(fitted_lines)))
            for i, (line_eq, points) in enumerate(fitted_lines):
                endpoints = self._calculate_line_endpoints(line_eq, image_array.shape)
                (x1, y1), (x2, y2) = endpoints
                axes[2].plot([x1, x2], [y1, y2], color=colors[i], linewidth=2)

        axes[2].set_title(f'3. Line Fitting ({len(fitted_lines)} lines)', fontweight='bold')
        axes[2].axis('off')

        # Step 4: Intersections
        if len(image_array.shape) == 3:
            axes[3].imshow(image_array, alpha=0.6)
        else:
            axes[3].imshow(image_array, cmap='gray', alpha=0.6)

        if fitted_lines:
            colors = plt.cm.Set3(np.linspace(0, 1, len(fitted_lines)))
            for i, (line_eq, points) in enumerate(fitted_lines):
                endpoints = self._calculate_line_endpoints(line_eq, image_array.shape)
                (x1, y1), (x2, y2) = endpoints
                axes[3].plot([x1, x2], [y1, y2], color=colors[i], linewidth=1, alpha=0.7)

        if intersections:
            int_array = np.array(intersections)
            axes[3].scatter(int_array[:, 0], int_array[:, 1], c='blue', s=20, alpha=0.8)

        axes[3].set_title(f'4. Intersections ({len(intersections)} points)', fontweight='bold')
        axes[3].axis('off')

        # Step 5: Clustering
        if len(image_array.shape) == 3:
            axes[4].imshow(image_array, alpha=0.6)
        else:
            axes[4].imshow(image_array, cmap='gray', alpha=0.6)

        if intersections and vanishing_point:
            int_array = np.array(intersections)
            axes[4].scatter(int_array[:, 0], int_array[:, 1], c='lightblue', s=30, alpha=0.6)
            axes[4].scatter(vanishing_point[0], vanishing_point[1], c='red', s=100, marker='*')

        axes[4].set_title('5. Clustering', fontweight='bold')
        axes[4].axis('off')

        # Step 6: Final result
        self._plot_detection_results(axes[5], image_array, fitted_lines, intersections, vanishing_point)
        axes[5].set_title('6. Final Result', fontweight='bold')

        plt.tight_layout()
        plt.show()

        # NEW: After the step-by-step, automatically show the clean vanishing point visualization
        if vanishing_point:
            self.plot_vanishing_point_only(image_array, vanishing_point)

    def save_visualization(self, image_array, fitted_lines, intersections, vanishing_point, 
                          filename='vanishing_point_result.png', dpi=300):
        """Save comprehensive visualization to file"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        self._plot_detection_results(ax, image_array, fitted_lines, intersections, vanishing_point)

        plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Comprehensive visualization saved as {filename}")

    def save_vanishing_point_only(self, image_array, vanishing_point, 
                                 filename='vanishing_point_clean.png', dpi=300):
        """
        Save the clean vanishing point visualization to file
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        # Show original image
        if len(image_array.shape) == 3:
            ax.imshow(image_array)
        else:
            ax.imshow(image_array, cmap='gray')

        # Draw vanishing point with thick red dot
        if vanishing_point:
            ax.scatter(vanishing_point[0], vanishing_point[1], 
                      c='red', s=400, 
                      marker='o', edgecolor='white', linewidth=4,
                      alpha=0.95, zorder=5)

            ax.scatter(vanishing_point[0], vanishing_point[1], 
                      c='darkred', s=100, 
                      marker='o', alpha=1.0, zorder=6)

            ax.annotate(f'Vanishing Point\n({int(vanishing_point[0])}, {int(vanishing_point[1])})', 
                       xy=vanishing_point, xytext=(40, 40), 
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=1.0', facecolor='white', alpha=0.95, 
                               edgecolor='red', linewidth=2),
                       fontsize=14, fontweight='bold', color='darkred', ha='center',
                       arrowprops=dict(arrowstyle='->', color='red', lw=3))

        ax.set_title('Detected Vanishing Point', fontsize=18, fontweight='bold', pad=25)
        ax.axis('off')

        plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Clean vanishing point visualization saved as {filename}")


# Utility functions
def plot_simple(image_array, vanishing_point, lines=None):
    """Simple plotting function for quick visualization"""
    visualizer = VanishingPointVisualizer()
    fitted_lines = lines or []
    visualizer.plot_results(image_array, fitted_lines, [], vanishing_point)


def plot_vanishing_point_clean(image_array, vanishing_point):
    """
    Quick function to show just the vanishing point on original image

    Args:
        image_array: Original image
        vanishing_point: Vanishing point coordinates (x, y)
    """
    visualizer = VanishingPointVisualizer()
    visualizer.plot_vanishing_point_only(image_array, vanishing_point)
