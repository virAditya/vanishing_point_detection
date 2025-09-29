"""
Image utilities for vanishing point detection
Contains functions for creating test images and image processing utilities
"""

import numpy as np
import math


def create_test_image(vp_x=300, vp_y=200, image_width=600, image_height=400, 
                      num_lines=4, add_horizontal_lines=True, line_thickness=3):
    """
    Create a synthetic test image with parallel lines converging to a vanishing point

    Args:
        vp_x, vp_y (int): Vanishing point coordinates
        image_width, image_height (int): Image dimensions
        num_lines (int): Number of converging line pairs
        add_horizontal_lines (bool): Whether to add horizontal reference lines
        line_thickness (int): Thickness of drawn lines

    Returns:
        numpy.ndarray: Synthetic test image with shape (height, width, 3)
    """
    # Create blank image
    img = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    def draw_thick_line(img, x1, y1, x2, y2, color, thickness=2):
        """Draw a thick line using Bresenham's algorithm"""
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Bresenham's line algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        x, y = x1, y1
        points_drawn = set()

        max_iterations = max(dx, dy) * 2 + 200
        iteration = 0

        while iteration < max_iterations:
            # Draw thick point
            for t_x in range(-thickness//2, thickness//2 + 1):
                for t_y in range(-thickness//2, thickness//2 + 1):
                    px, py = x + t_x, y + t_y
                    if (0 <= px < img.shape[1] and 0 <= py < img.shape[0] and
                        (px, py) not in points_drawn):
                        img[py, px] = color
                        points_drawn.add((px, py))

            if abs(x - x2) <= 1 and abs(y - y2) <= 1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

            iteration += 1

        return img

    # Generate line colors with good contrast
    line_colors = []
    for i in range(num_lines):
        intensity = max(100, 255 - i * 30)
        line_colors.append((intensity, intensity, intensity))

    # Draw converging lines from both sides
    for i, color in enumerate(line_colors):
        # Left side lines
        start_x_left = 50 + i * (120 // max(num_lines, 1))
        start_y_left = image_height - 50 - i * 15

        # Right side lines  
        start_x_right = image_width - 50 - i * (120 // max(num_lines, 1))
        start_y_right = image_height - 50 - i * 15

        for start_x, start_y in [(start_x_left, start_y_left), (start_x_right, start_y_right)]:
            # Calculate direction towards vanishing point
            dx = vp_x - start_x
            dy = vp_y - start_y
            length = math.sqrt(dx*dx + dy*dy)

            if length > 0:
                # Extend line beyond vanishing point
                factor = 2.5
                end_x = int(start_x + (dx/length) * length * factor)
                end_y = int(start_y + (dy/length) * length * factor)

                draw_thick_line(img, start_x, start_y, end_x, end_y, color, line_thickness)

    # Add horizontal reference lines
    if add_horizontal_lines:
        horizontal_color = (180, 180, 180)
        horizontal_positions = [image_height//4, image_height//3, int(image_height//2.2)]

        for y_pos in horizontal_positions:
            if 50 < y_pos < image_height - 50:
                draw_thick_line(img, 100, y_pos, image_width - 100, y_pos, 
                               horizontal_color, thickness=2)

    return img


def create_complex_test_image(image_width=800, image_height=600, add_noise=False):
    """
    Create a more complex test image with architectural elements

    Args:
        image_width, image_height (int): Image dimensions
        add_noise (bool): Whether to add noise for realism

    Returns:
        numpy.ndarray: Complex synthetic test image
    """
    img = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # Main vanishing point (central perspective)
    main_vp = (image_width // 2, image_height // 3)

    def draw_line(img, x1, y1, x2, y2, color, thickness=1):
        """Simple line drawing function"""
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        x, y = x1, y1

        for _ in range(max(dx, dy) + 1):
            for t in range(thickness):
                for s in range(thickness):
                    px, py = x + t, y + s
                    if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                        img[py, px] = color

            if x == x2 and y == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return img

    # Create building-like vertical lines
    pillar_color = (200, 200, 200)
    for x in range(120, image_width - 120, 90):
        draw_line(img, x, image_height // 2, x, image_height - 50, pillar_color, 4)

    # Create receding lines toward vanishing point
    floor_lines = 8
    for i in range(floor_lines):
        start_y = image_height - 80 + i * 6
        color_intensity = 255 - i * 25
        color = (color_intensity, color_intensity, color_intensity)

        # Left and right receding lines
        for start_x in [50, image_width - 50]:
            dx = main_vp[0] - start_x
            dy = main_vp[1] - start_y
            length = math.sqrt(dx*dx + dy*dy)

            if length > 0:
                factor = 1.8
                end_x = start_x + (dx/length) * length * factor
                end_y = start_y + (dy/length) * length * factor
                draw_line(img, start_x, start_y, end_x, end_y, color, 2)

    # Add ceiling line
    ceiling_y = image_height // 4
    ceiling_color = (180, 180, 180)
    draw_line(img, 100, ceiling_y, image_width - 100, ceiling_y, ceiling_color, 2)

    # Add noise if requested
    if add_noise:
        img = add_noise_to_image(img, noise_level=0.05)

    return img


def create_grid_perspective_image(image_width=600, image_height=400, 
                                 grid_size=20, vp_x=None, vp_y=None):
    """
    Create a perspective grid image for testing

    Args:
        image_width, image_height (int): Image dimensions
        grid_size (int): Size of grid squares
        vp_x, vp_y (int): Vanishing point coordinates

    Returns:
        numpy.ndarray: Grid perspective image
    """
    if vp_x is None:
        vp_x = image_width // 2
    if vp_y is None:
        vp_y = image_height // 2

    img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    line_color = (200, 200, 200)

    def draw_simple_line(img, x1, y1, x2, y2, color):
        """Simple line drawing"""
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        x, y = x1, y1

        while True:
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                img[y, x] = color

            if x == x2 and y == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    # Draw horizontal lines converging to vanishing point
    for y in range(grid_size, image_height, grid_size):
        if y != vp_y:
            # Find intersection with right edge
            if vp_x != 0:
                t = (image_width) / vp_x
                y2 = y + t * (vp_y - y)
                if 0 <= y2 <= image_height:
                    draw_simple_line(img, 0, y, image_width, int(y2), line_color)

    # Draw vertical receding lines
    for x in range(grid_size, image_width, grid_size):
        dx = vp_x - x
        dy = vp_y - image_height

        if dy != 0:
            factor = 2.0
            x2 = x + dx * factor
            y2 = image_height + dy * factor
            draw_simple_line(img, x, image_height, int(x2), int(y2), line_color)

    return img


def convert_to_grayscale(image_array):
    """Convert color image to grayscale using standard weights"""
    if len(image_array.shape) == 3:
        return 0.299 * image_array[:, :, 0] + 0.587 * image_array[:, :, 1] + 0.114 * image_array[:, :, 2]
    return image_array.astype(float)


def add_noise_to_image(image_array, noise_level=0.1):
    """Add Gaussian noise to an image for testing robustness"""
    noise = np.random.normal(0, noise_level * 255, image_array.shape)
    noisy_image = image_array.astype(float) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def resize_image_simple(image_array, new_width, new_height):
    """Simple image resizing using nearest neighbor interpolation"""
    old_height, old_width = image_array.shape[:2]

    x_ratio = old_width / new_width
    y_ratio = old_height / new_height

    if len(image_array.shape) == 3:
        resized = np.zeros((new_height, new_width, image_array.shape[2]), dtype=image_array.dtype)
    else:
        resized = np.zeros((new_height, new_width), dtype=image_array.dtype)

    for i in range(new_height):
        for j in range(new_width):
            orig_i = min(int(i * y_ratio), old_height - 1)
            orig_j = min(int(j * x_ratio), old_width - 1)
            resized[i, j] = image_array[orig_i, orig_j]

    return resized


def validate_image(image_array):
    """
    Validate that image array is in correct format

    Args:
        image_array (numpy.ndarray): Image to validate

    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(image_array, np.ndarray):
        return False, "Input is not a numpy array"

    if len(image_array.shape) not in [2, 3]:
        return False, f"Invalid image shape: {image_array.shape}"

    if len(image_array.shape) == 3 and image_array.shape[2] not in [1, 3, 4]:
        return False, f"Invalid number of channels: {image_array.shape[2]}"

    if image_array.dtype not in [np.uint8, np.float32, np.float64]:
        return False, f"Invalid data type: {image_array.dtype}"

    if image_array.size == 0:
        return False, "Image array is empty"

    return True, "Valid image"


def normalize_image(image_array, target_range=(0, 255)):
    """Normalize image values to target range"""
    min_val, max_val = target_range
    img_min, img_max = np.min(image_array), np.max(image_array)

    if img_max - img_min < 1e-10:
        return np.full_like(image_array, min_val)

    normalized = (image_array - img_min) / (img_max - img_min)
    scaled = normalized * (max_val - min_val) + min_val

    return scaled.astype(image_array.dtype)


def apply_gaussian_blur(image_array, kernel_size=5, sigma=1.0):
    """Apply Gaussian blur to image"""
    if kernel_size % 2 == 0:
        kernel_size += 1

    k = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size))

    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - k, j - k
            kernel[i, j] = np.exp(-(x*x + y*y) / (2 * sigma * sigma))

    kernel = kernel / np.sum(kernel)

    if len(image_array.shape) == 2:
        return apply_convolution_2d(image_array, kernel)
    else:
        result = np.zeros_like(image_array)
        for c in range(image_array.shape[2]):
            result[:, :, c] = apply_convolution_2d(image_array[:, :, c], kernel)
        return result


def apply_convolution_2d(image, kernel):
    """Apply 2D convolution to grayscale image"""
    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    result = np.zeros_like(image, dtype=np.float64)

    for i in range(img_h):
        for j in range(img_w):
            patch = padded[i:i + kernel_h, j:j + kernel_w]
            result[i, j] = np.sum(patch * kernel)

    return np.clip(result, 0, 255).astype(image.dtype)


# Test image creation functions
def create_hallway_image(width=800, height=600):
    """Create a hallway-like image with strong perspective"""
    return create_complex_test_image(width, height, add_noise=False)


def create_road_image(width=600, height=400):
    """Create a road-like image with vanishing point"""
    return create_test_image(vp_x=width//2, vp_y=height//3, 
                           image_width=width, image_height=height,
                           num_lines=6, add_horizontal_lines=True)


def create_building_facade_image(width=600, height=400):
    """Create a building facade with perspective"""
    return create_grid_perspective_image(width, height, grid_size=30)
