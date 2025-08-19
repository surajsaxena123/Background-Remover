"""
Enhanced BiRefNet Implementation for Precision-Grade Background Removal
Integrates multiple BiRefNet variants with advanced processing techniques
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Union, List, Dict, Tuple
from PIL import Image
import logging
from pathlib import Path

try:
    from rembg import remove, new_session
    from rembg.sessions import BiRefNetSession
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logging.warning("rembg not available. Install rembg for BiRefNet functionality.")

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    logging.warning("albumentations not available. Install for advanced augmentations.")
except Exception as e:
    ALBUMENTATIONS_AVAILABLE = False
    logging.warning(f"albumentations import failed: {e}. Using fallback augmentations.")

class EnhancedBiRefNet:
    """
    Enhanced BiRefNet implementation with multiple model variants and precision optimizations
    Specifically designed for precision-grade background removal applications
    """
    
    def __init__(
        self,
        model_variant: str = "birefnet-general-hd",
        precision_mode: str = "ultra_high",
        device: Optional[str] = None,
        use_ensemble: bool = True,
        enable_tta: bool = True  # Test Time Augmentation
    ):
        """
        Initialize Enhanced BiRefNet
        
        Args:
            model_variant: BiRefNet model variant to use
            precision_mode: Precision level ('high', 'ultra_high', 'precision')
            device: Computing device
            use_ensemble: Use ensemble of multiple models
            enable_tta: Enable test-time augmentation
        """
        self.device = device or self._get_optimal_device()
        self.model_variant = model_variant
        self.precision_mode = precision_mode
        self.use_ensemble = use_ensemble
        self.enable_tta = enable_tta
        
        # Model variants for ensemble
        self.model_variants = [
            "birefnet-general",
            "birefnet-general-lite",
            "birefnet-portrait", 
            "birefnet-dis"
        ]
        
        # Initialize sessions
        self.sessions = {}
        self.primary_session = None
        self._initialize_sessions()
        
        # Setup augmentation pipeline
        self.tta_transforms = self._setup_tta_transforms() if enable_tta else None
        
    def _get_optimal_device(self) -> str:
        """Get optimal computing device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _initialize_sessions(self):
        """Initialize BiRefNet sessions with error handling"""
        if not REMBG_AVAILABLE:
            logging.error("rembg not available - BiRefNet functionality disabled")
            return
        
        try:
            # Initialize primary session
            self.primary_session = new_session(self.model_variant)
            
            # Initialize ensemble sessions if enabled
            if self.use_ensemble:
                for variant in self.model_variants:
                    try:
                        self.sessions[variant] = new_session(variant)
                        logging.info(f"Initialized session for {variant}")
                    except Exception as e:
                        logging.warning(f"Failed to initialize {variant}: {e}")
                        
        except Exception as e:
            logging.error(f"Failed to initialize BiRefNet sessions: {e}")
            self.primary_session = None
    
    def _setup_tta_transforms(self) -> List:
        """Setup test-time augmentation transforms"""
        if not ALBUMENTATIONS_AVAILABLE:
            return self._setup_fallback_tta_transforms()
        
        try:
            transforms = []
            
            # Original image
            transforms.append(A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]))
            
            # Horizontal flip
            transforms.append(A.Compose([
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]))
            
            # Brightness adjustment
            transforms.append(A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]))
            
            # Multi-scale testing
            for scale in [0.9, 1.1]:
                transforms.append(A.Compose([
                    A.Resize(height=int(512 * scale), width=int(512 * scale)),
                    A.Resize(height=512, width=512),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ]))
            
            return transforms
        except Exception as e:
            logging.warning(f"Failed to setup albumentations transforms: {e}")
            return self._setup_fallback_tta_transforms()
    
    def _setup_fallback_tta_transforms(self) -> List:
        """Setup fallback TTA transforms using OpenCV"""
        return [
            {"type": "original"},
            {"type": "horizontal_flip"},
            {"type": "brightness", "factor": 1.1},
            {"type": "scale", "factor": 0.9},
            {"type": "scale", "factor": 1.1}
        ]
    
    def generate_precision_mask(
        self,
        image: np.ndarray,
        alpha_matting: bool = True,
        post_process: bool = True
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Generate high-precision mask using enhanced BiRefNet
        
        Args:
            image: Input image (BGR format)
            alpha_matting: Enable alpha matting
            post_process: Enable post-processing
            
        Returns:
            Tuple of (mask, quality_metrics)
        """
        if self.primary_session is None:
            raise RuntimeError("BiRefNet sessions not initialized")
        
        # Convert to RGB for processing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        masks = []
        confidences = []
        
        # Primary model prediction
        primary_mask, primary_confidence = self._predict_single_model(
            pil_image, self.primary_session, alpha_matting
        )
        masks.append(primary_mask)
        confidences.append(primary_confidence)
        
        # Ensemble predictions if enabled
        if self.use_ensemble and self.sessions:
            for variant, session in self.sessions.items():
                try:
                    mask, confidence = self._predict_single_model(
                        pil_image, session, alpha_matting
                    )
                    masks.append(mask)
                    confidences.append(confidence)
                except Exception as e:
                    logging.warning(f"Ensemble prediction failed for {variant}: {e}")
        
        # Test-time augmentation if enabled
        if self.enable_tta and self.tta_transforms:
            tta_masks = self._apply_tta(pil_image, alpha_matting)
            masks.extend(tta_masks)
            confidences.extend([0.8] * len(tta_masks))  # Default TTA confidence
        
        # Combine predictions
        if len(masks) > 1:
            final_mask = self._ensemble_masks(masks, confidences)
        else:
            final_mask = masks[0]
        
        # Post-processing
        if post_process:
            final_mask = self._advanced_post_process(image, final_mask)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(image, final_mask, masks)
        
        return final_mask, quality_metrics
    
    def _predict_single_model(
        self,
        pil_image: Image.Image,
        session,
        alpha_matting: bool
    ) -> Tuple[np.ndarray, float]:
        """Predict mask using a single model"""
        try:
            # Configure alpha matting parameters for medical precision
            alpha_params = {
                "alpha_matting": alpha_matting,
                "alpha_matting_foreground_threshold": 250,  # Higher threshold for precision
                "alpha_matting_background_threshold": 5,    # Lower threshold for precision
                "alpha_matting_erode_size": 3               # Fine-tune erosion
            } if alpha_matting else {"alpha_matting": False}
            
            mask_image = remove(
                pil_image,
                session=session,
                only_mask=True,
                **alpha_params
            )
            
            mask = np.array(mask_image.convert("L"))
            
            # Calculate confidence based on mask quality
            confidence = self._calculate_mask_confidence(mask)
            
            return mask, confidence
            
        except Exception as e:
            logging.error(f"Single model prediction failed: {e}")
            # Return empty mask with low confidence
            h, w = pil_image.size[1], pil_image.size[0]
            return np.zeros((h, w), dtype=np.uint8), 0.0
    
    def _apply_tta(self, pil_image: Image.Image, alpha_matting: bool) -> List[np.ndarray]:
        """Apply test-time augmentation"""
        if not self.tta_transforms or not self.primary_session:
            return []
        
        tta_masks = []
        rgb_array = np.array(pil_image)
        
        for transform in self.tta_transforms:
            try:
                if ALBUMENTATIONS_AVAILABLE and hasattr(transform, '__call__'):
                    # albumentations format
                    transformed = transform(image=rgb_array)
                    if isinstance(transformed, dict) and 'image' in transformed:
                        transformed_array = transformed['image']
                        if isinstance(transformed_array, torch.Tensor):
                            # Convert back to numpy and denormalize
                            transformed_array = transformed_array.permute(1, 2, 0).numpy()
                            # Denormalize
                            mean = np.array([0.485, 0.456, 0.406])
                            std = np.array([0.229, 0.224, 0.225])
                            transformed_array = (transformed_array * std + mean) * 255
                            transformed_array = np.clip(transformed_array, 0, 255).astype(np.uint8)
                        
                        transformed_pil = Image.fromarray(transformed_array)
                    else:
                        transformed_pil = pil_image
                else:
                    # Fallback transforms
                    transformed_array = self._apply_fallback_transform(rgb_array, transform)
                    transformed_pil = Image.fromarray(transformed_array)
                
                # Get mask
                mask, _ = self._predict_single_model(transformed_pil, self.primary_session, alpha_matting)
                
                # Reverse transformation if needed (e.g., horizontal flip)
                if isinstance(transform, dict) and transform.get("type") == "horizontal_flip":
                    mask = cv2.flip(mask, 1)
                
                tta_masks.append(mask)
                
            except Exception as e:
                logging.warning(f"TTA transform failed: {e}")
                continue
        
        return tta_masks
    
    def _apply_fallback_transform(self, image: np.ndarray, transform: dict) -> np.ndarray:
        """Apply fallback transforms using OpenCV"""
        result = image.copy()
        
        transform_type = transform.get("type", "original")
        
        if transform_type == "horizontal_flip":
            result = cv2.flip(result, 1)
        elif transform_type == "brightness":
            factor = transform.get("factor", 1.0)
            result = cv2.convertScaleAbs(result, alpha=factor, beta=0)
        elif transform_type == "scale":
            factor = transform.get("factor", 1.0)
            h, w = result.shape[:2]
            new_h, new_w = int(h * factor), int(w * factor)
            result = cv2.resize(result, (new_w, new_h))
            result = cv2.resize(result, (w, h))  # Resize back
        
        return result
    
    def _ensemble_masks(self, masks: List[np.ndarray], confidences: List[float]) -> np.ndarray:
        """Combine multiple masks using weighted ensemble"""
        if not masks:
            return np.zeros((512, 512), dtype=np.uint8)
        
        # Normalize confidences
        total_confidence = sum(confidences) if sum(confidences) > 0 else 1.0
        weights = [c / total_confidence for c in confidences]
        
        # Weighted average
        ensemble_mask = np.zeros_like(masks[0], dtype=np.float32)
        for mask, weight in zip(masks, weights):
            # Ensure masks have the same shape
            if mask.shape != ensemble_mask.shape:
                mask = cv2.resize(mask, (ensemble_mask.shape[1], ensemble_mask.shape[0]))
            ensemble_mask += mask.astype(np.float32) * weight
        
        # Apply precision thresholding based on mode
        if self.precision_mode == "precision":
            # More conservative thresholding for medical applications
            threshold = 200
        elif self.precision_mode == "ultra_high":
            threshold = 180
        else:  # high
            threshold = 128
        
        final_mask = (ensemble_mask > threshold).astype(np.uint8) * 255
        
        return final_mask
    
    def _advanced_post_process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Advanced post-processing for precision-grade quality"""
        
        # 1. Morphological refinement with precision kernels
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Close small gaps
        refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        
        # Remove small noise
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # 2. Edge-preserving smoothing
        refined = cv2.bilateralFilter(refined, 9, 75, 75)
        
        # 3. Guided filter for edge refinement (if available)
        try:
            import cv2.ximgproc as ximgproc
            guide = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            refined = ximgproc.guidedFilter(guide=guide, src=refined, radius=8, eps=1e-6)
        except (ImportError, AttributeError):
            pass
        
        # 4. Gradient-based edge refinement
        refined = self._gradient_edge_refinement(image, refined)
        
        # 5. Final precision thresholding
        refined = cv2.normalize(refined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Medical-grade precision: Use adaptive thresholding
        if self.precision_mode == "precision":
            refined = cv2.adaptiveThreshold(
                refined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:
            _, refined = cv2.threshold(refined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return refined
    
    def _gradient_edge_refinement(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Refine mask edges using image gradients"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient
        gradient_norm = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
        
        # Create edge mask from strong gradients
        _, edge_mask = cv2.threshold(gradient_norm, 0.3, 1, cv2.THRESH_BINARY)
        edge_mask = edge_mask.astype(np.uint8)
        
        # Refine original mask using edge information
        refined_mask = mask.copy().astype(np.float32) / 255.0
        
        # Increase mask confidence near strong edges
        refined_mask = refined_mask + 0.3 * edge_mask
        refined_mask = np.clip(refined_mask, 0, 1)
        
        return (refined_mask * 255).astype(np.uint8)
    
    def _calculate_mask_confidence(self, mask: np.ndarray) -> float:
        """Calculate confidence score for a mask"""
        if mask.size == 0:
            return 0.0
        
        # Factor 1: Foreground coverage (not too small, not too large)
        coverage = np.sum(mask > 128) / mask.size
        coverage_score = 1.0 - abs(coverage - 0.3)  # Optimal around 30% coverage
        coverage_score = max(0, coverage_score)
        
        # Factor 2: Edge definition (gradient strength at mask boundaries)
        edges = cv2.Canny(mask, 50, 150)
        edge_ratio = np.sum(edges > 0) / mask.size
        edge_score = min(1.0, edge_ratio * 10)  # More edges = better definition
        
        # Factor 3: Shape coherence (connected components)
        binary_mask = (mask > 128).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary_mask)
        
        if num_labels <= 1:
            coherence_score = 1.0
        else:
            # Penalize multiple disconnected components
            coherence_score = 1.0 / num_labels
        
        # Combined confidence score
        confidence = (coverage_score * 0.4 + edge_score * 0.3 + coherence_score * 0.3)
        return float(np.clip(confidence, 0, 1))
    
    def _calculate_quality_metrics(
        self,
        image: np.ndarray,
        final_mask: np.ndarray,
        all_masks: List[np.ndarray]
    ) -> Dict[str, float]:
        """Calculate comprehensive quality metrics"""
        metrics = {}
        
        # Basic mask properties
        metrics['foreground_coverage'] = np.sum(final_mask > 128) / final_mask.size
        metrics['mask_confidence'] = self._calculate_mask_confidence(final_mask)
        
        # Edge quality
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        mask_edges = cv2.Canny(final_mask, 50, 150)
        edge_pixels = np.where(mask_edges > 0)
        
        if len(edge_pixels[0]) > 0:
            edge_gradient_values = gradient_magnitude[edge_pixels]
            metrics['edge_alignment'] = float(np.mean(edge_gradient_values) / (gradient_magnitude.max() + 1e-8))
        else:
            metrics['edge_alignment'] = 0.0
        
        # Ensemble consistency (if multiple masks available)
        if len(all_masks) > 1:
            consistency_scores = []
            for i in range(len(all_masks)):
                for j in range(i + 1, len(all_masks)):
                    # Calculate IoU between masks
                    mask1 = (all_masks[i] > 128).astype(bool)
                    mask2 = (all_masks[j] > 128).astype(bool)
                    intersection = np.logical_and(mask1, mask2)
                    union = np.logical_or(mask1, mask2)
                    iou = intersection.sum() / (union.sum() + 1e-8)
                    consistency_scores.append(iou)
            
            metrics['ensemble_consistency'] = float(np.mean(consistency_scores)) if consistency_scores else 1.0
        else:
            metrics['ensemble_consistency'] = 1.0
        
        # Overall quality score
        metrics['overall_quality'] = (
            metrics['mask_confidence'] * 0.4 +
            metrics['edge_alignment'] * 0.3 +
            metrics['ensemble_consistency'] * 0.3
        )
        
        return metrics

class PrecisionAlphaMatting:
    """
    Advanced alpha matting implementation for precision-grade edge refinement
    """
    
    def __init__(self, precision_level: str = "ultra_high"):
        self.precision_level = precision_level
        
        # Precision-specific parameters
        if precision_level == "precision":
            self.fg_threshold = 250
            self.bg_threshold = 5
            self.erode_iterations = 3
            self.dilate_iterations = 3
        elif precision_level == "ultra_high":
            self.fg_threshold = 240
            self.bg_threshold = 10
            self.erode_iterations = 2
            self.dilate_iterations = 2
        else:  # high
            self.fg_threshold = 230
            self.bg_threshold = 15
            self.erode_iterations = 1
            self.dilate_iterations = 1
    
    def generate_trimap(self, mask: np.ndarray) -> np.ndarray:
        """Generate high-quality trimap for alpha matting"""
        # Ensure binary mask
        binary_mask = (mask > 128).astype(np.uint8) * 255
        
        # Create kernels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Generate sure foreground and background regions
        sure_fg = cv2.erode(binary_mask, kernel, iterations=self.erode_iterations)
        sure_bg = cv2.dilate(binary_mask, kernel, iterations=self.dilate_iterations)
        
        # Create trimap
        trimap = np.full(mask.shape, 128, dtype=np.uint8)  # Unknown region
        trimap[sure_fg > 250] = 255  # Foreground
        trimap[sure_bg < 5] = 0      # Background
        
        return trimap
    
    def apply_closed_form_matting(
        self,
        image: np.ndarray,
        trimap: np.ndarray
    ) -> np.ndarray:
        """Apply closed-form alpha matting"""
        try:
            from pymatting.alpha import estimate_alpha_cf
            from pymatting.foreground import estimate_foreground_ml
            
            # Normalize image
            image_normalized = image.astype(np.float64) / 255.0
            trimap_normalized = trimap.astype(np.float64) / 255.0
            
            # Estimate alpha matte
            alpha = estimate_alpha_cf(image_normalized, trimap_normalized)
            
            # Ensure alpha is in valid range
            alpha = np.clip(alpha, 0, 1)
            
            return alpha
            
        except ImportError:
            logging.warning("pymatting not available - using fallback trimap")
            return trimap.astype(np.float64) / 255.0
        except Exception as e:
            logging.error(f"Alpha matting failed: {e}")
            return trimap.astype(np.float64) / 255.0
    
    def refine_alpha_matte(self, alpha: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Additional refinement of alpha matte"""
        
        # 1. Guided filter smoothing
        try:
            import cv2.ximgproc as ximgproc
            alpha_refined = ximgproc.guidedFilter(
                guide=image, src=(alpha * 255).astype(np.uint8), radius=4, eps=1e-8
            ).astype(np.float64) / 255.0
        except (ImportError, AttributeError):
            alpha_refined = alpha
        
        # 2. Edge-preserving smoothing
        alpha_uint8 = (alpha_refined * 255).astype(np.uint8)
        alpha_smooth = cv2.bilateralFilter(alpha_uint8, 9, 50, 50)
        alpha_refined = alpha_smooth.astype(np.float64) / 255.0
        
        # 3. Precision-based final adjustment
        if self.precision_level == "precision":
            # More conservative alpha values for medical applications
            alpha_refined = np.where(alpha_refined > 0.95, 1.0, alpha_refined)
            alpha_refined = np.where(alpha_refined < 0.05, 0.0, alpha_refined)
        
        return np.clip(alpha_refined, 0, 1)