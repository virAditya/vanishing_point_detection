"""
Main entry point for vanishing point detection
Complete interactive interface with image upload functionality and Windows path support
"""

import os
import sys
from pathlib import Path
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.vanishing_point import VanishingPointDetector
from visualization.plotter import VanishingPointVisualizer
from utils.image_utils import create_test_image, create_complex_test_image, validate_image
from config import VanishingPointConfig, ConfigPresets


def clean_file_path(path_input):
    """
    Clean user input path by removing quotes and handling raw string prefix

    Args:
        path_input (str): Raw user input

    Returns:
        str: Cleaned file path
    """
    path = path_input.strip()

    # Remove raw string prefix if present
    if path.startswith('r"') and path.endswith('"'):
        path = path[2:-1]  # Remove r" and "
    elif path.startswith("r'") and path.endswith("'"):
        path = path[2:-1]  # Remove r' and '
    # Remove quotes
    elif path.startswith('"') and path.endswith('"'):
        path = path[1:-1]  # Remove " and "
    elif path.startswith("'") and path.endswith("'"):
        path = path[1:-1]  # Remove ' and '

    # Convert to proper path format
    path = os.path.normpath(path)

    return path


def load_image(image_path):
    """
    Load image from file path using available libraries

    Args:
        image_path (str): Path to image file

    Returns:
        numpy.ndarray: Loaded image array
    """
    try:
        import matplotlib.pyplot as plt
        image = plt.imread(image_path)
        return image
    except ImportError:
        try:
            from PIL import Image
            image = Image.open(image_path)
            return np.array(image)
        except ImportError:
            print("❌ No image loading library available. Please install matplotlib or PIL.")
            return None
    except Exception as e:
        print(f"❌ Error loading image: {str(e)}")
        return None


def check_file_exists_and_format(file_path):
    """
    Check if file exists and is a valid image format

    Args:
        file_path (str): Path to check

    Returns:
        tuple: (exists, is_image, error_message)
    """
    if not os.path.exists(file_path):
        return False, False, f"File does not exist: {file_path}"

    if not os.path.isfile(file_path):
        return False, False, f"Path is not a file: {file_path}"

    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
    file_ext = Path(file_path).suffix.lower()

    if file_ext not in valid_extensions:
        return True, False, f"Unsupported format: {file_ext}. Supported: {', '.join(valid_extensions)}"

    return True, True, "Valid image file"


def get_image_files():
    """
    Get list of image files from user input or current directory

    Returns:
        list: List of image file paths
    """
    print("🖼️ Image Upload Options:")
    print("1. Enter image file paths manually")
    print("2. Process all images in current directory")
    print("3. Use synthetic test images")

    choice = input("\nSelect option (1-3): ").strip()

    if choice == "1":
        # Manual file path entry with improved handling
        image_paths = []
        print("\n📝 Enter image file paths:")
        print("💡 Tips:")
        print("   • You can paste paths with or without quotes")
        print("   • Press Enter without input to finish")
        print("   • Example: C:\\Users\\YourName\\Pictures\\image.png")
        print()

        while True:
            path_input = input("Image path: ").strip()
            if not path_input:
                break

            # Clean the input path
            cleaned_path = clean_file_path(path_input)
            print(f"🔍 Checking: {cleaned_path}")

            # Check if file exists and is valid
            exists, is_image, message = check_file_exists_and_format(cleaned_path)

            if exists and is_image:
                image_paths.append(cleaned_path)
                print(f"✅ Added: {cleaned_path}")
            else:
                print(f"❌ {message}")

                # Suggest alternatives if file doesn't exist
                if not exists:
                    parent_dir = os.path.dirname(cleaned_path)
                    if os.path.exists(parent_dir):
                        print(f"💡 Directory exists: {parent_dir}")
                        try:
                            files = os.listdir(parent_dir)
                            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                            if image_files:
                                print(f"💡 Found these images in directory:")
                                for img_file in image_files[:5]:
                                    print(f"   - {img_file}")
                                if len(image_files) > 5:
                                    print(f"   ... and {len(image_files) - 5} more")
                        except:
                            pass
                    else:
                        print(f"💡 Directory doesn't exist: {parent_dir}")

        return image_paths

    elif choice == "2":
        # Process all images in current directory
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
        current_dir = Path('.')

        image_paths = []
        for ext in image_extensions:
            image_paths.extend(list(current_dir.glob(f'*{ext}')))
            image_paths.extend(list(current_dir.glob(f'*{ext.upper()}')))

        image_paths = [str(path) for path in image_paths]

        if image_paths:
            print(f"\n📁 Found {len(image_paths)} images in current directory:")
            for path in image_paths[:10]:
                print(f"   - {path}")
            if len(image_paths) > 10:
                print(f"   ... and {len(image_paths) - 10} more")
        else:
            print("❌ No image files found in current directory")
            print(f"💡 Current directory: {os.getcwd()}")

        return image_paths

    elif choice == "3":
        # Use synthetic test images
        return ["synthetic"]

    else:
        print("❌ Invalid choice. Using synthetic test images.")
        return ["synthetic"]


def select_detection_config():
    """
    Allow user to select detection configuration preset

    Returns:
        VanishingPointConfig: Selected configuration
    """
    print("\n⚙️ Detection Configuration Options:")
    print("1. Default configuration")
    print("2. High precision (slower but more accurate)")
    print("3. Fast detection (faster but less precise)")
    print("4. Noisy images (optimized for noisy/blurry images)")
    print("5. Architectural scenes (optimized for buildings)")
    print("6. Road scenes (optimized for streets/roads)")
    print("7. Custom configuration")

    choice = input("Select configuration (1-7): ").strip()

    if choice == "2":
        return ConfigPresets.high_precision()
    elif choice == "3":
        return ConfigPresets.fast_detection()
    elif choice == "4":
        return ConfigPresets.noisy_images()
    elif choice == "5":
        return ConfigPresets.architectural()
    elif choice == "6":
        return ConfigPresets.road_scenes()
    elif choice == "7":
        # Custom configuration
        try:
            edge_threshold = float(input("Edge threshold (0.1-0.9, default 0.5): ") or "0.5")
            angle_tolerance = float(input("Angle tolerance (0.1-0.5, default 0.2): ") or "0.2")
            min_line_points = int(input("Min points per line (5-30, default 10): ") or "10")

            return VanishingPointConfig.create_custom_config(
                edge_threshold=edge_threshold,
                angle_tolerance=angle_tolerance,
                min_line_points=min_line_points
            )
        except ValueError:
            print("❌ Invalid input. Using default configuration.")
            return VanishingPointConfig()
    else:
        return VanishingPointConfig()  # Default


def process_single_image(image_path, detector, visualizer, save_results=False):
    """
    Process a single image for vanishing point detection

    Args:
        image_path (str): Path to image file or "synthetic"
        detector (VanishingPointDetector): Initialized detector
        visualizer (VanishingPointVisualizer): Initialized visualizer
        save_results (bool): Whether to save visualization results

    Returns:
        tuple: (success, vanishing_point, quality_score)
    """
    print(f"\n🔍 Processing: {os.path.basename(image_path) if image_path != 'synthetic' else 'Synthetic Test Image'}")

    # Load or create image
    if image_path == "synthetic":
        print("🎨 Creating synthetic test image...")
        image = create_test_image(vp_x=300, vp_y=180, image_width=600, image_height=400)
        display_name = "Synthetic Test Image"
    else:
        print(f"📂 Loading image from: {image_path}")
        image = load_image(image_path)
        display_name = os.path.basename(image_path)

        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return False, None, 0.0

    # Validate image
    is_valid, error_msg = validate_image(image)
    if not is_valid:
        print(f"❌ Invalid image: {error_msg}")
        return False, None, 0.0

    print(f"📐 Image shape: {image.shape}")
    print(f"📊 Image info: {image.dtype}, min={np.min(image)}, max={np.max(image)}")

    # Detect vanishing point
    try:
        vanishing_point, fitted_lines = detector.detect_vanishing_point(image, debug=True)

        # Get quality score
        quality_score = detector.get_detection_quality_score()

        if vanishing_point:
            print(f"✅ SUCCESS! Vanishing point detected at: ({vanishing_point[0]:.1f}, {vanishing_point[1]:.1f})")
            print(f"📊 Quality score: {quality_score:.2f}")
            print(f"📏 Number of lines detected: {len(fitted_lines)}")

            # Visualize results
            print("📊 Displaying visualization...")
            visualizer.plot_results(image, fitted_lines, [], vanishing_point)

            # Save results if requested
            if save_results:
                filename = f"vp_result_{Path(image_path).stem if image_path != 'synthetic' else 'synthetic'}.png"
                visualizer.save_visualization(image, fitted_lines, [], vanishing_point, filename)

            return True, vanishing_point, quality_score
        else:
            print("❌ No vanishing point detected")
            print(f"📊 Quality score: {quality_score:.2f}")

            # Show debug info to help understand why detection failed
            debug_info = detector.get_debug_info()
            print("🔍 Debug information:")
            for key, value in debug_info.items():
                if isinstance(value, dict):
                    print(f"   {key}:")
                    for k, v in value.items():
                        print(f"     {k}: {v}")
                else:
                    print(f"   {key}: {value}")

            return False, None, quality_score

    except Exception as e:
        print(f"❌ Error during detection: {str(e)}")
        import traceback
        print("Full error traceback:")
        traceback.print_exc()
        return False, None, 0.0


def interactive_mode():
    """
    Interactive mode for single image processing with step-by-step visualization
    """
    print("\n🎮 Interactive Mode - Step-by-step processing")

    # Get single image
    image_paths = get_image_files()
    if not image_paths:
        print("❌ No images selected")
        return

    image_path = image_paths[0]  # Use first image
    config = select_detection_config()

    # Initialize components
    detector = VanishingPointDetector(config)
    visualizer = VanishingPointVisualizer(config)

    # Load image
    if image_path == "synthetic":
        image = create_test_image(vp_x=300, vp_y=180)
    else:
        image = load_image(image_path)
        if image is None:
            print("❌ Failed to load image")
            return

    print("\n🔍 Performing step-by-step detection...")

    # Manual step-by-step processing
    try:
        edge_points = detector.edge_detector.detect_edges(image)
        fitted_lines = detector.line_detector.detect_line_segments(edge_points)
        intersections = detector._find_all_intersections(fitted_lines, image.shape)
        vanishing_point = detector._cluster_intersections(intersections)

        # Show step-by-step visualization
        visualizer.plot_step_by_step(image, edge_points, fitted_lines, intersections, vanishing_point)
    except Exception as e:
        print(f"❌ Error in step-by-step processing: {str(e)}")
        import traceback
        traceback.print_exc()


def batch_process_images(image_paths, config):
    """
    Process multiple images in batch mode

    Args:
        image_paths (list): List of image file paths
        config (VanishingPointConfig): Detection configuration
    """
    print(f"\n🔄 Batch Processing {len(image_paths)} images...")

    # Initialize detector and visualizer
    detector = VanishingPointDetector(config)
    visualizer = VanishingPointVisualizer(config)

    # Process each image
    results = []
    successful_detections = 0

    save_results = input("\nSave visualization results? (y/n): ").lower().startswith('y')

    for i, image_path in enumerate(image_paths):
        print(f"\n{'='*60}")
        print(f"Processing image {i+1}/{len(image_paths)}")
        print('='*60)

        success, vp, quality = process_single_image(image_path, detector, visualizer, save_results)
        results.append((image_path, success, vp, quality))

        if success:
            successful_detections += 1

    # Print summary
    print(f"\n{'='*60}")
    print("🎯 BATCH PROCESSING SUMMARY")
    print('='*60)
    print(f"Total images processed: {len(image_paths)}")
    print(f"Successful detections: {successful_detections}")
    print(f"Success rate: {(successful_detections/len(image_paths)*100):.1f}%")

    # Show detailed results
    print(f"\nDetailed Results:")
    for image_path, success, vp, quality in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        vp_str = f"({vp[0]:.1f}, {vp[1]:.1f})" if vp else "N/A"
        filename = os.path.basename(image_path) if image_path != "synthetic" else "Synthetic"
        print(f"  {filename}: {status} | VP: {vp_str} | Quality: {quality:.2f}")


def main():
    """Main function with user interface"""
    print("🎨 Modular Vanishing Point Detection System")
    print("=" * 60)
    print("Inspired by classical paintings like Vermeer's 'The Music Lesson'")
    print("Detects vanishing points in images with parallel lines")
    print("=" * 60)
    print(f"💻 Running on: {os.name} - {os.getcwd()}")

    # Main menu
    while True:
        print("\n📋 Main Menu:")
        print("1. Batch process multiple images")
        print("2. Interactive single image processing")
        print("3. Quick test with synthetic image")
        print("4. Configuration help")
        print("5. Exit")

        choice = input("\nSelect option (1-5): ").strip()

        if choice == "1":
            # Batch processing mode
            image_paths = get_image_files()
            if not image_paths:
                print("❌ No images selected")
                continue

            config = select_detection_config()
            batch_process_images(image_paths, config)

        elif choice == "2":
            # Interactive mode
            interactive_mode()

        elif choice == "3":
            # Quick test
            print("\n🧪 Quick Test with Synthetic Image")
            config = VanishingPointConfig()
            detector = VanishingPointDetector(config)
            visualizer = VanishingPointVisualizer(config)

            success, vp, quality = process_single_image("synthetic", detector, visualizer)

        elif choice == "4":
            # Configuration help
            print("\n⚙️ Configuration Help:")
            print("• Edge Threshold (0.1-0.9): Lower = more edges, higher = fewer edges")
            print("• Angle Tolerance (0.1-0.5): Lower = stricter line grouping")
            print("• Min Line Points (5-30): Minimum points needed to form a line")
            print("\nPresets:")
            print("• High Precision: Best accuracy, slower processing")
            print("• Fast Detection: Quick results, may miss some details")
            print("• Noisy Images: Optimized for blurry or noisy photos")
            print("• Architectural: Best for building/structure photos")
            print("• Road Scenes: Optimized for street/road images")

        elif choice == "5":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please select 1-5.")


def detect_from_file(image_path, debug=True):
    """
    Simple function to detect vanishing point from a single file

    Args:
        image_path (str): Path to image file
        debug (bool): Show debug information

    Returns:
        tuple: (vanishing_point, fitted_lines)
    """
    detector = VanishingPointDetector()
    visualizer = VanishingPointVisualizer()

    # Clean the path
    cleaned_path = clean_file_path(image_path)

    image = load_image(cleaned_path)
    if image is None:
        return None, []

    vp, lines = detector.detect_vanishing_point(image, debug=debug)
    if vp:
        visualizer.plot_results(image, lines, [], vp)

    return vp, lines


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("Please check your image files and try again.")
        import traceback
        traceback.print_exc()
