"""
Precision SAM2 Integration for High-Quality Background Removal
Implements state-of-the-art segmentation models for precision-grade quality
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import cv2
from PIL import Image
import logging
from pathlib import Path

try:
    from ultralytics import SAM
    from segment_anything_hq import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logging.warning("SAM models not available. Install ultralytics and segment-anything-hq for enhanced precision.")

class PrecisionSAM2Segmentor:
    """Precision-grade segmentation using SAM2 and advanced techniques.
    
    Implements state-of-the-art SAM2 models optimized for precision-grade 
    background removal with exceptional edge quality and confidence validation.
    
    This class provides high-precision segmentation capabilities with automatic
    device optimization, confidence thresholding, and optional edge refinement.
    
    Attributes
    ----------
    device : str
        Computing device ('cuda', 'cpu', 'mps') selected automatically or manually.
    model_type : str
        SAM2 model variant being used.
    confidence_threshold : float
        Minimum confidence score for accepting segmentation results.
    edge_refinement : bool
        Whether edge refinement post-processing is enabled.
    """
    
    def __init__(
        self,
        model_type: str = "sam2_hiera_large",
        device: Optional[str] = None,
        confidence_threshold: float = 0.95,
        edge_refinement: bool = True
    ):
        """Initialize Precision SAM2 segmentor.

        Parameters
        ----------
        model_type : str, optional
            SAM2 model variant to use. Options: 'sam2_hiera_large', 'sam2_hiera_base'.
            Larger models provide better quality at the cost of memory and speed.
        device : Optional[str], optional
            Computing device ('cuda', 'cpu', 'mps'). If None, automatically
            selects the best available device.
        confidence_threshold : float, optional
            Minimum confidence score for accepting segmentation results (0.0-1.0).
            Higher values ensure better quality but may reject valid results.
        edge_refinement : bool, optional
            Enable advanced edge refinement post-processing for smoother boundaries.
        
        Notes
        -----
        The segmentor will fallback gracefully if SAM models are not available,
        logging appropriate warnings for missing dependencies.
        """
        self.device = device or self._get_optimal_device()
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.edge_refinement = edge_refinement
        
        self.sam_model = None
        self.predictor = None
        self._initialize_models()
        
    def _get_optimal_device(self) -> str:
        """Automatically select the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _initialize_models(self):
        """Initialize SAM2 models with error handling"""
        if not SAM_AVAILABLE:
            logging.warning("SAM models not available - falling back to basic segmentation")
            return
            
        try:
            # Initialize SAM2 with Ultralytics
            self.sam_model = SAM(f"{self.model_type}.pt")
            self.sam_model.to(self.device)
            
            # Initialize high-quality SAM predictor for refinement
            try:
                sam_hq = sam_model_registry["vit_h"](checkpoint="sam_hq_vit_h.pth")
                sam_hq.to(device=self.device)
                self.predictor = SamPredictor(sam_hq)
            except Exception as e:
                logging.warning(f"High-quality SAM not available: {e}")
                
        except Exception as e:
            logging.error(f"Failed to initialize SAM models: {e}")
            self.sam_model = None
            self.predictor = None
    
    def generate_high_precision_mask(
        self,
        image: np.ndarray,
        prompts: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Generate high-precision segmentation mask using Precision SAM2
        
        Args:
            image: Input image (BGR format)
            prompts: Optional prompts for guided segmentation
            
        Returns:
            Tuple of (mask, confidence_score)
        """
        if self.sam_model is None:
            return self._fallback_segmentation(image)
        
        try:
            # Convert to RGB for SAM
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Auto-segmentation with SAM2
            results = self.sam_model(rgb_image, verbose=False)
            
            if not results or len(results) == 0:
                return self._fallback_segmentation(image)
            
            # Extract the best mask based on confidence
            best_mask = None
            best_confidence = 0.0
            
            for result in results:
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    confidences = getattr(result, 'conf', torch.ones(len(masks))).cpu().numpy()
                    
                    for mask, conf in zip(masks, confidences):
                        if conf > best_confidence and conf >= self.confidence_threshold:
                            best_mask = mask
                            best_confidence = float(conf)
            
            if best_mask is None:
                return self._fallback_segmentation(image)
            
            # Convert to uint8 format
            mask = (best_mask * 255).astype(np.uint8)
            
            # Apply high-quality refinement if available
            if self.predictor is not None and self.edge_refinement:
                mask = self._refine_with_sam_hq(rgb_image, mask)
            
            return mask, best_confidence
            
        except Exception as e:
            logging.error(f"SAM2 segmentation failed: {e}")
            return self._fallback_segmentation(image)
    
    def _refine_with_sam_hq(self, rgb_image: np.ndarray, initial_mask: np.ndarray) -> np.ndarray:
        """Refine mask using high-quality SAM predictor"""
        try:
            self.predictor.set_image(rgb_image)
            
            # Generate box prompt from initial mask
            coords = np.where(initial_mask > 128)
            if len(coords[0]) == 0:
                return initial_mask
                
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            box = np.array([x_min, y_min, x_max, y_max])
            
            # Predict with high-quality model
            masks, scores, _ = self.predictor.predict(
                box=box,
                multimask_output=True
            )
            
            # Select best mask
            best_idx = np.argmax(scores)
            refined_mask = (masks[best_idx] * 255).astype(np.uint8)
            
            return refined_mask
            
        except Exception as e:
            logging.warning(f"High-quality refinement failed: {e}")
            return initial_mask
    
    def _fallback_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fallback segmentation when SAM is not available"""
        from .processing import generate_mask
        try:
            mask = generate_mask(image, model="birefnet-general")
            return mask, 0.8  # Default confidence for fallback
        except Exception as e:
            logging.error(f"Fallback segmentation failed: {e}")
            # Create a simple foreground mask as last resort
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return mask, 0.5

class PrecisionQualityValidator:
    """
    Quality validation for precision-grade segmentation results
    Implements metrics and checks for precision assessment
    """
    
    def __init__(self, min_dice_score: float = 0.95, min_edge_quality: float = 0.90):
        self.min_dice_score = min_dice_score
        self.min_edge_quality = min_edge_quality
    
    def validate_segmentation_quality(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        reference_mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Comprehensive quality validation for segmentation results
        
        Args:
            image: Original image
            mask: Generated segmentation mask
            reference_mask: Optional ground truth mask for comparison
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Basic mask quality metrics
        metrics['mask_coverage'] = np.sum(mask > 0) / mask.size
        metrics['mask_coherence'] = self._calculate_coherence(mask)
        metrics['edge_quality'] = self._calculate_edge_quality(image, mask)
        
        # Advanced precision metrics
        metrics['gradient_consistency'] = self._calculate_gradient_consistency(image, mask)
        metrics['boundary_smoothness'] = self._calculate_boundary_smoothness(mask)
        
        # Comparison with reference if available
        if reference_mask is not None:
            metrics['dice_score'] = self._calculate_dice_score(mask, reference_mask)
            metrics['iou_score'] = self._calculate_iou_score(mask, reference_mask)
            metrics['hausdorff_distance'] = self._calculate_hausdorff_distance(mask, reference_mask)
        
        # Overall quality score
        metrics['overall_quality'] = self._calculate_overall_quality(metrics)
        
        return metrics
    
    def _calculate_coherence(self, mask: np.ndarray) -> float:
        """Calculate mask coherence (connected components analysis)"""
        binary_mask = (mask > 128).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary_mask)
        
        if num_labels <= 1:
            return 1.0
        
        # Calculate size of largest component relative to total
        largest_component = 0
        for i in range(1, num_labels):
            component_size = np.sum(labels == i)
            largest_component = max(largest_component, component_size)
        
        total_foreground = np.sum(binary_mask)
        return largest_component / total_foreground if total_foreground > 0 else 0.0
    
    def _calculate_edge_quality(self, image: np.ndarray, mask: np.ndarray) -> float:
        """Calculate edge quality using gradient analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate image gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate mask edges
        mask_edges = cv2.Canny((mask > 128).astype(np.uint8) * 255, 50, 150)
        
        # Measure alignment between mask edges and image gradients
        edge_pixels = np.where(mask_edges > 0)
        if len(edge_pixels[0]) == 0:
            return 0.0
        
        edge_gradient_values = gradient_magnitude[edge_pixels]
        normalized_gradient = edge_gradient_values / (gradient_magnitude.max() + 1e-8)
        
        return float(np.mean(normalized_gradient))
    
    def _calculate_gradient_consistency(self, image: np.ndarray, mask: np.ndarray) -> float:
        """Calculate gradient consistency across mask boundaries"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Dilate and erode to get boundary region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary = dilated - eroded
        
        boundary_pixels = np.where(boundary > 0)
        if len(boundary_pixels[0]) == 0:
            return 1.0
        
        # Calculate gradient variance in boundary region
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        boundary_grad_x = grad_x[boundary_pixels]
        boundary_grad_y = grad_y[boundary_pixels]
        
        # Consistency is inverse of gradient variance
        grad_variance = np.var(boundary_grad_x) + np.var(boundary_grad_y)
        consistency = 1.0 / (1.0 + grad_variance / 1000.0)  # Normalized
        
        return float(consistency)
    
    def _calculate_boundary_smoothness(self, mask: np.ndarray) -> float:
        """Calculate boundary smoothness using contour analysis"""
        binary_mask = (mask > 128).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return 0.0
        
        # Analyze largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) < 10:
            return 0.0
        
        # Calculate curvature variance as smoothness metric
        contour_points = largest_contour.reshape(-1, 2)
        
        # Calculate local curvature
        curvatures = []
        for i in range(len(contour_points)):
            p1 = contour_points[(i - 1) % len(contour_points)]
            p2 = contour_points[i]
            p3 = contour_points[(i + 1) % len(contour_points)]
            
            # Calculate angle change
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            curvatures.append(angle)
        
        # Smoothness is inverse of curvature variance
        curvature_variance = np.var(curvatures)
        smoothness = 1.0 / (1.0 + curvature_variance)
        
        return float(smoothness)
    
    def _calculate_dice_score(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate Dice similarity coefficient"""
        mask1_binary = (mask1 > 128).astype(bool)
        mask2_binary = (mask2 > 128).astype(bool)
        
        intersection = np.logical_and(mask1_binary, mask2_binary)
        dice = 2.0 * intersection.sum() / (mask1_binary.sum() + mask2_binary.sum() + 1e-8)
        
        return float(dice)
    
    def _calculate_iou_score(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate Intersection over Union score"""
        mask1_binary = (mask1 > 128).astype(bool)
        mask2_binary = (mask2 > 128).astype(bool)
        
        intersection = np.logical_and(mask1_binary, mask2_binary)
        union = np.logical_or(mask1_binary, mask2_binary)
        
        iou = intersection.sum() / (union.sum() + 1e-8)
        return float(iou)
    
    def _calculate_hausdorff_distance(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate Hausdorff distance between mask boundaries"""
        from scipy.spatial.distance import directed_hausdorff
        
        # Extract boundaries
        edges1 = cv2.Canny((mask1 > 128).astype(np.uint8) * 255, 50, 150)
        edges2 = cv2.Canny((mask2 > 128).astype(np.uint8) * 255, 50, 150)
        
        points1 = np.column_stack(np.where(edges1 > 0))
        points2 = np.column_stack(np.where(edges2 > 0))
        
        if len(points1) == 0 or len(points2) == 0:
            return float('inf')
        
        # Calculate directed Hausdorff distances
        d1 = directed_hausdorff(points1, points2)[0]
        d2 = directed_hausdorff(points2, points1)[0]
        
        return float(max(d1, d2))
    
    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            'mask_coherence': 0.2,
            'edge_quality': 0.25,
            'gradient_consistency': 0.2,
            'boundary_smoothness': 0.15,
            'dice_score': 0.2  # If available
        }
        
        quality_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                quality_score += metrics[metric] * weight
                total_weight += weight
        
        return quality_score / total_weight if total_weight > 0 else 0.0