#!/usr/bin/env python3
"""
Precision Background Remover - Main Entry Point

A state-of-the-art background removal system with precision-grade quality.
Supports command-line interface, batch processing, and quality analysis.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np

# Import core functionality
try:
    from src.core import (
        remove_background_precision_grade,
        analyze_image_quality,
        optimize_processing_parameters
    )
    PRECISION_MODE_AVAILABLE = True
except ImportError:
    from src.core import remove_background
    PRECISION_MODE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('precision_bg_remover.log')
    ]
)
logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Precision Background Remover",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.jpg output.png
  %(prog)s input.jpg output.png --precision-mode ultra_high
  %(prog)s input.jpg output.png --analyze-only
  %(prog)s --batch input_dir/ output_dir/
        """
    )
    
    # Input/Output arguments
    parser.add_argument('input', help='Input image path or directory for batch processing')
    parser.add_argument('output', nargs='?', help='Output image path or directory')
    
    # Processing options
    parser.add_argument(
        '--precision-mode',
        choices=['high', 'ultra_high', 'precision'],
        default='ultra_high',
        help='Precision level for processing (default: ultra_high)'
    )
    
    parser.add_argument(
        '--model',
        choices=['birefnet-general', 'birefnet-portrait', 'birefnet-hd', 'u2net'],
        default='birefnet-general',
        help='Model variant to use (default: birefnet-general)'
    )
    
    # Feature flags
    parser.add_argument(
        '--use-sam2',
        action='store_true',
        help='Enable Precision SAM2 for enhanced segmentation'
    )
    
    parser.add_argument(
        '--enable-hair-enhancement',
        action='store_true',
        default=True,
        help='Enable specialized hair detail processing'
    )
    
    parser.add_argument(
        '--alpha-matting',
        action='store_true',
        default=True,
        help='Enable alpha matting for better edges'
    )
    
    # Analysis options
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze image quality without processing'
    )
    
    parser.add_argument(
        '--quality-validation',
        action='store_true',
        default=True,
        help='Enable comprehensive quality validation'
    )
    
    # Batch processing
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all images in input directory'
    )
    
    parser.add_argument(
        '--format',
        choices=['png', 'jpg', 'webp'],
        default='png',
        help='Output format for batch processing (default: png)'
    )
    
    # Optimization
    parser.add_argument(
        '--auto-optimize',
        action='store_true',
        default=True,
        help='Automatically optimize parameters based on image characteristics'
    )
    
    # Verbosity
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress non-error output'
    )
    
    return parser


def process_single_image(
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace
) -> bool:
    """Process a single image with precision-grade background removal.

    Parameters
    ----------
    input_path : Path
        Path to the input image file.
    output_path : Path
        Path where the processed image should be saved.
    args : argparse.Namespace
        Command line arguments containing processing parameters.

    Returns
    -------
    bool
        True if processing completed successfully, False otherwise.

    Notes
    -----
    This function handles the complete pipeline for single image processing:
    - Image loading with error handling
    - Quality analysis (if --analyze-only flag is set)
    - Automatic parameter optimization (if enabled)  
    - Precision-grade or standard processing based on availability
    - Results saving with comprehensive error handling
    """
    try:
        # Load image
        logger.info(f"Loading image: {input_path}")
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.error(f"Could not load image: {input_path}")
            return False
        
        # Analyze image quality if requested
        if args.analyze_only:
            logger.info("Performing image quality analysis...")
            metrics = analyze_image_quality(image)
            
            print(f"\nImage Quality Analysis for {input_path.name}:")
            print("-" * 50)
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
            
            # Provide recommendations
            print("\nRecommendations:")
            if metrics.get('sharpness', 0) < 100:
                print("  - Image appears blurry, consider using higher quality input")
            if metrics.get('noise_level', 0) > 50:
                print("  - High noise detected, consider noise reduction")
            if metrics.get('edge_density', 0) > 0.1:
                print("  - High detail image, excellent for enhanced processing")
            
            return True
        
        # Process with precision background removal
        if PRECISION_MODE_AVAILABLE:
            # Setup processing parameters
            params = {
                'precision_mode': args.precision_mode,
                'use_sam2': args.use_sam2,
                'use_enhanced_birefnet': True,
                'quality_validation': args.quality_validation,
                'model': args.model,
                'alpha_matting': args.alpha_matting,
                'enable_hair_enhancement': args.enable_hair_enhancement,
            }
            
            # Auto-optimize parameters if requested
            if args.auto_optimize:
                logger.info("Auto-optimizing processing parameters...")
                optimized_params = optimize_processing_parameters(image, args.precision_mode)
                params.update(optimized_params)
            
            logger.info(f"Processing with precision-grade quality...")
            result, metrics = remove_background_precision_grade(image, **params)
            
            if result is not None:
                # Save result
                success = cv2.imwrite(str(output_path), result)
                if success:
                    logger.info(f"Result saved to: {output_path}")
                    
                    # Display quality metrics
                    quality_score = metrics.get('quality_score', 0)
                    print(f"\nProcessing Results for {input_path.name}:")
                    print("-" * 50)
                    print(f"  Quality Score: {quality_score:.3f}")
                    print(f"  Model Used: {metrics.get('model_used', 'Unknown')}")
                    
                    if quality_score >= 0.8:
                        print("  Result: Excellent quality achieved!")
                    elif quality_score >= 0.6:
                        print("  Result: Good quality achieved!")
                    else:
                        print("  Result: Quality could be improved")
                    
                    return True
                else:
                    logger.error(f"Failed to save result to: {output_path}")
                    return False
            else:
                logger.error("Processing failed")
                return False
        else:
            # Fallback to standard processing
            logger.warning("Precision-grade features not available, using standard processing")
            result = remove_background(image, model=args.model)
            
            if result is not None:
                success = cv2.imwrite(str(output_path), result)
                if success:
                    logger.info(f"Result saved to: {output_path}")
                    return True
                else:
                    logger.error(f"Failed to save result to: {output_path}")
                    return False
            else:
                logger.error("Processing failed")
                return False
                
    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        return False


def process_batch(
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace
) -> int:
    """Process all images in a directory."""
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        logger.error(f"No image files found in {input_dir}")
        return 0
    
    logger.info(f"Found {len(image_files)} images to process")
    
    successful = 0
    failed = 0
    
    for image_file in image_files:
        output_file = output_dir / f"{image_file.stem}.{args.format}"
        
        logger.info(f"Processing {image_file.name} ({successful + failed + 1}/{len(image_files)})")
        
        if process_single_image(image_file, output_file, args):
            successful += 1
        else:
            failed += 1
    
    print(f"\nBatch Processing Results:")
    print("-" * 30)
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(image_files)}")
    
    return successful


def main() -> int:
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1
    
    try:
        if args.batch or input_path.is_dir():
            # Batch processing
            if not args.output:
                logger.error("Output directory required for batch processing")
                return 1
            
            output_path = Path(args.output)
            successful = process_batch(input_path, output_path, args)
            return 0 if successful > 0 else 1
        else:
            # Single image processing
            if not args.output and not args.analyze_only:
                # Generate output filename
                output_path = input_path.parent / f"{input_path.stem}_processed.png"
            elif args.output:
                output_path = Path(args.output)
            else:
                output_path = None
            
            success = process_single_image(input_path, output_path, args)
            return 0 if success else 1
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())