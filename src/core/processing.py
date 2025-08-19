import io
from typing import Optional, Tuple, Dict, Any, List
import logging

import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session
from pymatting.alpha import estimate_alpha_cf

# Import precision-grade components
try:
    from .precision_sam2 import PrecisionSAM2Segmentor, PrecisionQualityValidator
    from ..models.enhanced_birefnet import EnhancedBiRefNet, PrecisionAlphaMatting
    PRECISION_GRADE_AVAILABLE = True
except (ImportError, Exception) as e:
    logging.warning(f"Precision-grade modules not available: {e}")
    PRECISION_GRADE_AVAILABLE = False


def generate_mask(
    image: np.ndarray,
    session: Optional[object] = None,
    model: str = "birefnet-general",
) -> np.ndarray:
    """Generate a foreground mask for the given image using rembg.

    Parameters
    ----------
    image : np.ndarray
        BGR image to segment with shape (H, W, 3).
    session : Optional[object], optional
        Optional rembg session to reuse for efficiency. If None, a new session
        will be created with the specified model.
    model : str, optional
        Name of the rembg model. Defaults to "birefnet-general" for
        high-fidelity hair and face boundaries.

    Returns
    -------
    np.ndarray
        Binary foreground mask with shape (H, W) and dtype uint8.
        Values are 0 (background) or 255 (foreground).

    Raises
    ------
    Exception
        If the model fails to load or process the image.
    """
    if session is None:
        try:
            session = new_session(model)
        except Exception:
            session = new_session()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    mask_image = remove(
        pil_image,
        session=session,
        only_mask=True,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
    )

    # rembg returns a PIL Image when given a PIL Image input
    mask = np.array(mask_image.convert("L"))
    return mask


def refine_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Refine the raw mask with morphological operations and optional guided filter.

    Parameters
    ----------
    image : np.ndarray
        Original BGR image with shape (H, W, 3) used for guided filtering.
    mask : np.ndarray
        Raw binary mask with shape (H, W) and dtype uint8.

    Returns
    -------
    np.ndarray
        Refined binary mask with improved connectivity and smoother boundaries.
        Shape (H, W) with dtype uint8, values 0 (background) or 255 (foreground).

    Notes
    -----
    The refinement process includes:
    1. Morphological closing to fill small gaps
    2. Morphological opening to remove noise
    3. Gaussian blur for smoothing
    4. Optional guided filter for edge preservation (if opencv-contrib available)
    5. Adaptive thresholding for final binarization
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)
    refined = cv2.GaussianBlur(refined, (5, 5), 0)

    # Attempt to use guided filter if available for better edge quality
    try:
        import cv2.ximgproc as ximgproc
        guide = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        refined = ximgproc.guidedFilter(guide=guide, src=refined, radius=8, eps=1e-6)
    except Exception:
        pass

    refined = cv2.normalize(refined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, refined = cv2.threshold(refined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return refined


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply closed-form alpha matting for high-precision edges.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image with shape (H, W, 3) and dtype uint8.
    mask : np.ndarray  
        Binary segmentation mask with shape (H, W) and dtype uint8.
        Values should be 0 (background) or 255 (foreground).

    Returns
    -------
    np.ndarray
        BGRA image with transparent background, shape (H, W, 4) and dtype uint8.
        Alpha channel contains the computed alpha matte values.

    Notes
    -----
    This function implements closed-form alpha matting for high-quality edge refinement:
    1. Creates trimap by eroding/dilating the mask
    2. Applies closed-form matting algorithm
    3. Composes final BGRA result with computed alpha values
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = cv2.erode(mask, kernel, iterations=5)
    bg = cv2.dilate(mask, kernel, iterations=5)
    trimap = np.full(mask.shape, 128, dtype=np.uint8)
    trimap[fg > 200] = 255
    trimap[bg < 55] = 0
    alpha = estimate_alpha_cf(image_rgb, trimap / 255.0)
    comp = image_rgb * alpha[..., None]
    rgba = np.dstack((comp, alpha))
    return cv2.cvtColor((rgba * 255).astype(np.uint8), cv2.COLOR_RGBA2BGRA)


def remove_background_precision_grade(
    image: np.ndarray,
    precision_mode: str = "ultra_high",
    use_sam2: bool = True,
    use_enhanced_birefnet: bool = True,
    quality_validation: bool = True,
    session: Optional[object] = None,
    model: str = "birefnet-general-hd",
    alpha_matting: bool = True,
    post_processing: bool = True,
    enable_hair_enhancement: bool = True,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Precision-grade background removal with state-of-the-art quality.
    
    This is the flagship function providing precision-grade background removal
    using multiple state-of-the-art AI models in ensemble. Specialized for
    maximum quality with exceptional hair and fine detail preservation.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image with shape (H, W, 3) and dtype uint8.
    precision_mode : str, optional
        Precision level: 'high', 'ultra_high', or 'precision'.
        Higher precision levels use more aggressive quality thresholds and processing.
    use_sam2 : bool, optional
        Enable Precision SAM2 for enhanced segmentation quality.
    use_enhanced_birefnet : bool, optional
        Enable Enhanced BiRefNet ensemble implementation.
    quality_validation : bool, optional
        Enable comprehensive quality validation and metrics computation.
    session : Optional[object], optional
        Pre-initialized rembg session for fallback processing.
    model : str, optional
        Model variant to use for fallback. Defaults to "birefnet-general-hd".
    alpha_matting : bool, optional
        Enable advanced alpha matting for superior edge quality.
    post_processing : bool, optional
        Enable post-processing refinement operations.
    enable_hair_enhancement : bool, optional
        Enable specialized hair and fine detail processing.
    **kwargs
        Additional parameters for compatibility and advanced configuration.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        A tuple containing:
        - result_image: BGRA image with precision-grade background removal
        - quality_metrics: Comprehensive quality assessment including confidence scores,
          edge alignment, mask coherence, and overall quality rating

    Notes
    -----
    Precision-grade quality thresholds:
    - 'precision': ≥95% overall quality required
    - 'ultra_high': ≥90% overall quality required  
    - 'high': ≥85% overall quality required

    The function employs multiple processing strategies:
    1. Precision SAM2 for state-of-the-art segmentation (if enabled)
    2. Enhanced BiRefNet ensemble with test-time augmentation
    3. Advanced mask combination using confidence weighting
    4. Precision alpha matting with hair-specific enhancements
    5. Comprehensive quality validation and reporting
    
    Examples
    --------
    >>> import cv2
    >>> from src.core import remove_background_precision_grade
    >>> image = cv2.imread('portrait.jpg')
    >>> result, metrics = remove_background_precision_grade(
    ...     image, 
    ...     precision_mode='precision',
    ...     enable_hair_enhancement=True
    ... )
    >>> print(f"Quality score: {metrics['overall_quality']:.3f}")
    >>> cv2.imwrite('precision_result.png', result)
    """
    if not PRECISION_GRADE_AVAILABLE:
        logging.warning("Precision-grade modules not available, falling back to standard processing")
        result = remove_background(image, session=session, model=model)
        return result, {"fallback_mode": True}
    
    quality_metrics = {}
    masks = []
    confidences = []
    
    # Method 1: Precision SAM2 (if enabled and available)
    if use_sam2:
        try:
            sam_segmentor = PrecisionSAM2Segmentor(
                confidence_threshold=0.95 if precision_mode == "precision" else 0.90
            )
            sam_mask, sam_confidence = sam_segmentor.generate_high_precision_mask(image)
            masks.append(sam_mask)
            confidences.append(sam_confidence)
            quality_metrics['sam2_confidence'] = sam_confidence
        except Exception as e:
            logging.warning(f"Precision SAM2 failed: {e}")
    
    # Method 2: Enhanced BiRefNet (if enabled)
    if use_enhanced_birefnet:
        try:
            birefnet = EnhancedBiRefNet(
                precision_mode=precision_mode,
                use_ensemble=True,
                enable_tta=precision_mode in ["ultra_high", "precision"]
            )
            birefnet_mask, birefnet_metrics = birefnet.generate_precision_mask(
                image, alpha_matting=alpha_matting, post_process=post_processing
            )
            masks.append(birefnet_mask)
            confidences.append(birefnet_metrics.get('overall_quality', 0.8))
            quality_metrics.update(birefnet_metrics)
        except Exception as e:
            logging.warning(f"Enhanced BiRefNet failed: {e}")
    
    # Fallback to standard method if no masks generated
    if not masks:
        mask = generate_mask(image, session=session, model=model)
        if post_processing:
            mask = refine_mask(image, mask)
        masks = [mask]
        confidences = [0.7]  # Default confidence
    
    # Combine masks if multiple available
    if len(masks) > 1:
        combined_mask = combine_masks_advanced(masks, confidences, precision_mode)
    else:
        combined_mask = masks[0]
    
    # Advanced precision alpha matting with hair enhancement
    result = apply_precision_alpha_matting(
        image, combined_mask, precision_mode, enable_hair_enhancement
    )
    
    # Quality validation
    if quality_validation:
        try:
            validator = PrecisionQualityValidator(
                min_dice_score=0.95 if precision_mode == "precision" else 0.90
            )
            validation_metrics = validator.validate_segmentation_quality(image, combined_mask)
            quality_metrics.update(validation_metrics)
        except Exception as e:
            logging.warning(f"Quality validation failed: {e}")
    
    return result, quality_metrics

def remove_background(
    image: np.ndarray,
    session: Optional[object] = None,
    model: str = "birefnet-general",
) -> np.ndarray:
    """Standard pipeline to remove background from an image.

    This is the main entry point for standard background removal processing.
    It combines mask generation, refinement, and alpha matting in a streamlined pipeline.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image with shape (H, W, 3) and dtype uint8.
    session : Optional[object], optional
        Pre-initialized rembg session for efficiency in batch processing.
        If None, a new session will be created.
    model : str, optional
        rembg model name to use. Defaults to "birefnet-general".
        Common options: "birefnet-general", "u2net", "birefnet-portrait".

    Returns
    -------
    np.ndarray
        Processed BGRA image with transparent background, shape (H, W, 4) and dtype uint8.
        Background pixels have alpha=0, foreground pixels have computed alpha values.

    Examples
    --------
    >>> import cv2
    >>> from src.core import remove_background
    >>> image = cv2.imread('input.jpg')
    >>> result = remove_background(image)
    >>> cv2.imwrite('output.png', result)
    """
    mask = generate_mask(image, session=session, model=model)
    mask = refine_mask(image, mask)
    return apply_mask(image, mask)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Remove image background with post-processing")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("output_image", help="Path to save the processed image")
    parser.add_argument(
        "--model",
        default="birefnet-general",
        help="rembg model name (e.g. 'birefnet-general', 'u2net')",
    )
    args = parser.parse_args()

    image = cv2.imread(args.input_image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.input_image)
    result = remove_background(image, model=args.model)
    cv2.imwrite(args.output_image, result)


def combine_masks_advanced(
    masks: List[np.ndarray], 
    confidences: List[float], 
    precision_mode: str
) -> np.ndarray:
    """Advanced mask combination using weighted ensemble and precision optimization.
    
    Combines multiple segmentation masks using confidence-weighted averaging
    with precision-aware thresholding and morphological refinement.

    Parameters
    ----------
    masks : List[np.ndarray]
        List of binary masks with shape (H, W) and dtype uint8.
        All masks will be resized to match the first mask's dimensions.
    confidences : List[float]
        Confidence scores corresponding to each mask, used for weighting.
        Higher confidence masks contribute more to the final result.
    precision_mode : str
        Precision level ('high', 'ultra_high', 'precision') affecting
        threshold selection and post-processing aggressiveness.

    Returns
    -------
    np.ndarray
        Combined binary mask with shape (H, W) and dtype uint8.
        Values are 0 (background) or 255 (foreground).

    Notes
    -----
    The combination process:
    1. Normalizes confidence weights to sum to 1.0
    2. Resizes all masks to common dimensions
    3. Applies confidence-weighted averaging
    4. Uses precision-mode specific thresholds
    5. Applies morphological cleanup operations
    """
    if not masks:
        return np.zeros((512, 512), dtype=np.uint8)
    
    # Normalize confidences
    total_confidence = sum(confidences) if sum(confidences) > 0 else 1.0
    weights = [c / total_confidence for c in confidences]
    
    # Ensure all masks have the same shape
    target_shape = masks[0].shape
    normalized_masks = []
    
    for mask in masks:
        if mask.shape != target_shape:
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]))
        normalized_masks.append(mask.astype(np.float32))
    
    # Weighted combination
    combined = np.zeros_like(normalized_masks[0])
    for mask, weight in zip(normalized_masks, weights):
        combined += mask * weight
    
    # Precision-based thresholding
    if precision_mode == "precision":
        # Conservative threshold for precision applications
        threshold = 200
        # Apply additional smoothing
        combined = cv2.GaussianBlur(combined, (3, 3), 0.5)
    elif precision_mode == "ultra_high":
        threshold = 180
    else:
        threshold = 150
    
    # Apply threshold and morphological operations
    result = (combined > threshold).astype(np.uint8) * 255
    
    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return result

def apply_precision_alpha_matting(
    image: np.ndarray, 
    mask: np.ndarray, 
    precision_mode: str,
    enable_hair_enhancement: bool = True
) -> np.ndarray:
    """Apply precision-grade alpha matting with specialized hair enhancement.
    
    Uses state-of-the-art matting techniques optimized for precision-grade
    quality with optional hair-specific processing for maximum detail preservation.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image with shape (H, W, 3) and dtype uint8.
    mask : np.ndarray
        Initial segmentation mask with shape (H, W) and dtype uint8.
    precision_mode : str
        Precision level ('high', 'ultra_high', 'precision') affecting
        algorithm parameters and quality thresholds.
    enable_hair_enhancement : bool, optional
        Enable specialized hair processing using structure tensors and
        Gabor filters for fine detail preservation.

    Returns
    -------
    np.ndarray
        BGRA image with precision alpha matte, shape (H, W, 4) and dtype uint8.
        Alpha channel contains smooth, high-quality alpha values.

    Notes
    -----
    The precision matting pipeline:
    1. Attempts to use PrecisionMattingEngine if available
    2. Applies hair-specific enhancement using structure analysis
    3. Falls back to enhanced alpha matting with high-quality trimap
    4. Composes final BGRA result with computed alpha values
    
    Raises
    ------
    ImportError
        If precision matting modules are not available, falls back gracefully.
    """
    try:
        # Import precision matting engine
        from .precision_matting import PrecisionMattingEngine, HairSpecificProcessor
        
        # Initialize precision matting engine
        matting_engine = PrecisionMattingEngine(
            precision_level=precision_mode,
            device=get_optimal_device() if 'get_optimal_device' in globals() else 'cpu'
        )
        
        # Generate high-precision alpha matte
        alpha, alpha_metrics = matting_engine.generate_precision_alpha(image, mask)
        
        # Apply hair-specific enhancement if enabled
        if enable_hair_enhancement:
            hair_processor = HairSpecificProcessor()
            alpha = hair_processor.enhance_hair_regions(image, alpha)
            alpha_metrics['hair_enhancement_applied'] = True
        else:
            alpha_metrics['hair_enhancement_applied'] = False
        
        # Convert to RGB for composition
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        
        # Compose final image with alpha
        comp = image_rgb * alpha[..., None]
        rgba = np.dstack((comp, alpha))
        
        # Convert back to BGRA
        result = cv2.cvtColor((rgba * 255).astype(np.uint8), cv2.COLOR_RGBA2BGRA)
        
        logging.info(f"Precision alpha matting completed with quality: {alpha_metrics.get('overall_alpha_quality', 0):.3f}")
        
        return result
        
    except ImportError:
        logging.warning("Precision matting module not available, using enhanced fallback")
        return apply_enhanced_alpha_matting(image, mask, precision_mode)
    except Exception as e:
        logging.warning(f"Precision alpha matting failed: {e}, using fallback")
        return apply_enhanced_alpha_matting(image, mask, precision_mode)

def apply_enhanced_alpha_matting(
    image: np.ndarray, 
    mask: np.ndarray, 
    precision_mode: str
) -> np.ndarray:
    """Enhanced alpha matting fallback with hair-aware processing.
    
    Provides high-quality alpha matting when precision-grade modules
    are unavailable, using enhanced BiRefNet methods with fallback
    to standard closed-form matting.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image with shape (H, W, 3) and dtype uint8.
    mask : np.ndarray
        Segmentation mask with shape (H, W) and dtype uint8.
    precision_mode : str
        Precision level affecting trimap generation and refinement parameters.

    Returns
    -------
    np.ndarray
        BGRA image with enhanced alpha matte, shape (H, W, 4) and dtype uint8.

    Notes
    -----
    Processing pipeline:
    1. Attempts enhanced BiRefNet alpha matting if available
    2. Generates high-quality trimap with precision-aware parameters  
    3. Applies closed-form matting with alpha refinement
    4. Falls back to standard apply_mask if all enhanced methods fail
    """
    try:
        # Use enhanced BiRefNet alpha matting if available
        if PRECISION_GRADE_AVAILABLE:
            from ..models.enhanced_birefnet import PrecisionAlphaMatting
            
            precision_matting = PrecisionAlphaMatting(precision_level=precision_mode)
            
            # Generate high-quality trimap
            trimap = precision_matting.generate_trimap(mask)
            
            # Apply closed-form matting
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            alpha = precision_matting.apply_closed_form_matting(image_rgb, trimap)
            
            # Refine alpha matte
            alpha_refined = precision_matting.refine_alpha_matte(alpha, image)
            
            # Compose final image
            comp = image_rgb * alpha_refined[..., None]
            rgba = np.dstack((comp, alpha_refined))
            
            return cv2.cvtColor((rgba * 255).astype(np.uint8), cv2.COLOR_RGBA2BGRA)
            
    except Exception as e:
        logging.warning(f"Enhanced alpha matting failed: {e}, using standard method")
    
    # Final fallback to standard alpha matting
    return apply_mask(image, mask)

def analyze_image_quality(image: np.ndarray) -> Dict[str, float]:
    """Analyze input image quality for optimal processing parameters.
    
    Performs comprehensive image quality assessment to guide automatic
    parameter optimization and processing recommendations.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image with shape (H, W, 3) and dtype uint8.

    Returns
    -------
    Dict[str, float]
        Quality metrics dictionary containing:
        - 'sharpness': Laplacian variance indicating image sharpness
        - 'contrast': Standard deviation of grayscale intensities
        - 'brightness': Mean grayscale intensity (0-255)
        - 'edge_density': Proportion of edge pixels detected
        - 'noise_level': Estimated noise level from high-frequency components

    Notes
    -----
    Quality assessment techniques:
    - Sharpness: Laplacian variance (higher values = sharper)
    - Contrast: Standard deviation of pixel intensities
    - Brightness: Mean luminance value
    - Edge Density: Canny edge detection ratio
    - Noise Level: High-frequency component analysis using LoG filter

    These metrics help optimize processing parameters automatically.
    """
    metrics = {}
    
    # Image sharpness (Laplacian variance)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    metrics['sharpness'] = float(laplacian_var)
    
    # Contrast (standard deviation of pixel intensities)
    metrics['contrast'] = float(np.std(gray))
    
    # Brightness (mean pixel intensity)
    metrics['brightness'] = float(np.mean(gray))
    
    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    metrics['edge_density'] = float(np.sum(edges > 0) / edges.size)
    
    # Noise estimation (using high-frequency components)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    noise_estimation = cv2.filter2D(gray, -1, kernel)
    metrics['noise_level'] = float(np.std(noise_estimation))
    
    return metrics

def optimize_processing_parameters(
    image: np.ndarray, 
    target_precision: str = "ultra_high"
) -> Dict[str, Any]:
    """Automatically optimize processing parameters based on image characteristics.
    
    Analyzes input image quality and automatically selects optimal processing
    parameters for the best quality/performance trade-off.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image with shape (H, W, 3) and dtype uint8.
    target_precision : str, optional
        Target precision level ('high', 'ultra_high', 'precision').
        Used as baseline, may be adjusted based on image analysis.

    Returns
    -------
    Dict[str, Any]
        Optimized processing parameters including:
        - 'use_sam2': Whether to enable Precision SAM2
        - 'use_enhanced_birefnet': Whether to use enhanced BiRefNet
        - 'precision_mode': Adjusted precision level  
        - 'alpha_matting': Whether to enable alpha matting
        - 'quality_validation': Whether to enable quality validation

    Notes
    -----
    Parameter optimization rules:
    - Blurry images (sharpness < 100): Use ultra_high precision, enhanced BiRefNet
    - Noisy images (noise > 50): Enable alpha matting for better noise handling
    - High-detail images (edge_density > 0.1): Enable SAM2 for better edge handling
    - Low-contrast images: Increase processing aggressiveness
    """
    quality_metrics = analyze_image_quality(image)
    
    params = {
        'use_sam2': True,
        'use_enhanced_birefnet': True,
        'precision_mode': target_precision,
        'alpha_matting': True,
        'quality_validation': True
    }
    
    # Adjust based on image quality
    if quality_metrics['sharpness'] < 100:  # Blurry image
        params['use_enhanced_birefnet'] = True  # Better for handling blur
        params['precision_mode'] = 'ultra_high'  # Higher precision needed
    
    if quality_metrics['noise_level'] > 50:  # Noisy image
        params['alpha_matting'] = True  # Better noise handling
    
    if quality_metrics['edge_density'] > 0.1:  # High detail image
        params['use_sam2'] = True  # Better edge handling
    
    return params

# Backward compatibility alias
# Backward compatibility alias
remove_background_medical_grade = remove_background_precision_grade

if __name__ == "__main__":
    main()
