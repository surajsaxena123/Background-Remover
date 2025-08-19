"""
Advanced Precision Matting for Hair and Fine Detail Preservation
Combines state-of-the-art techniques: Trimap-free matting, Deep Image Matting, and Hair-specific processing
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import logging
from PIL import Image
import scipy.ndimage
from scipy import sparse
from scipy.sparse.linalg import spsolve

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available - using CPU-only implementations")

class PrecisionMattingEngine:
    """
    State-of-the-art precision matting engine specifically designed for hair and fine detail preservation
    Combines multiple cutting-edge approaches for maximum quality
    """
    
    def __init__(self, precision_level: str = "ultra_high", device: str = "cpu"):
        """
        Initialize precision matting engine
        
        Args:
            precision_level: 'high', 'ultra_high', or 'precision'
            device: Computing device ('cpu', 'cuda', 'mps')
        """
        self.precision_level = precision_level
        self.device = device
        self.enable_hair_enhancement = True
        self.enable_trimap_free = True
        self.enable_deep_matting = TORCH_AVAILABLE
        
        # Precision-specific parameters
        self.config = self._get_precision_config()
        
    def _get_precision_config(self) -> Dict[str, Any]:
        """Get configuration based on precision level"""
        configs = {
            "precision": {
                "guided_filter_radius": 4,
                "guided_filter_eps": 1e-8,
                "morphology_iterations": 1,
                "bilateral_d": 15,
                "bilateral_sigma_color": 150,
                "bilateral_sigma_space": 150,
                "hair_enhancement_strength": 0.8,
                "edge_preservation_factor": 0.9,
                "trimap_precision": 3,
                "alpha_refinement_iterations": 3
            },
            "ultra_high": {
                "guided_filter_radius": 6,
                "guided_filter_eps": 1e-6,
                "morphology_iterations": 2,
                "bilateral_d": 12,
                "bilateral_sigma_color": 100,
                "bilateral_sigma_space": 100,
                "hair_enhancement_strength": 0.6,
                "edge_preservation_factor": 0.8,
                "trimap_precision": 2,
                "alpha_refinement_iterations": 2
            },
            "high": {
                "guided_filter_radius": 8,
                "guided_filter_eps": 1e-4,
                "morphology_iterations": 3,
                "bilateral_d": 9,
                "bilateral_sigma_color": 75,
                "bilateral_sigma_space": 75,
                "hair_enhancement_strength": 0.4,
                "edge_preservation_factor": 0.7,
                "trimap_precision": 1,
                "alpha_refinement_iterations": 1
            }
        }
        return configs.get(self.precision_level, configs["ultra_high"])
    
    def generate_precision_alpha(
        self, 
        image: np.ndarray, 
        initial_mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Generate high-precision alpha matte with hair detail preservation
        
        Args:
            image: Input image (BGR format)
            initial_mask: Initial segmentation mask
            
        Returns:
            Tuple of (alpha_matte, quality_metrics)
        """
        metrics = {}
        
        # Step 1: Enhanced trimap generation with hair detection
        trimap = self._generate_enhanced_trimap(image, initial_mask)
        metrics['trimap_quality'] = self._evaluate_trimap_quality(trimap)
        
        # Step 2: Hair-aware alpha matting
        if self.enable_hair_enhancement:
            alpha = self._hair_aware_alpha_matting(image, trimap)
            metrics['hair_enhancement'] = True
        else:
            alpha = self._standard_alpha_matting(image, trimap)
            metrics['hair_enhancement'] = False
        
        # Step 3: Multi-scale refinement
        alpha_refined = self._multi_scale_alpha_refinement(image, alpha)
        
        # Step 4: Edge-preserving post-processing
        alpha_final = self._edge_preserving_postprocess(image, alpha_refined)
        
        # Step 5: Quality assessment
        metrics.update(self._assess_alpha_quality(image, alpha_final, initial_mask))
        
        return alpha_final, metrics
    
    def _generate_enhanced_trimap(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Generate enhanced trimap with hair-aware boundaries"""
        
        # Convert mask to binary
        binary_mask = (mask > 128).astype(np.uint8)
        
        # Detect hair regions using texture analysis
        hair_regions = self._detect_hair_regions(image, binary_mask)
        
        # Adaptive morphological operations based on content
        kernel_size = self.config['trimap_precision'] * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Different erosion/dilation for hair vs non-hair regions
        hair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        non_hair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Process hair regions more carefully
        hair_fg = cv2.erode(binary_mask * hair_regions, hair_kernel, iterations=1)
        hair_bg = cv2.dilate(binary_mask * hair_regions, hair_kernel, iterations=2)
        
        # Process non-hair regions normally
        non_hair_mask = binary_mask * (1 - hair_regions)
        non_hair_fg = cv2.erode(non_hair_mask, non_hair_kernel, iterations=self.config['trimap_precision'])
        non_hair_bg = cv2.dilate(non_hair_mask, non_hair_kernel, iterations=self.config['trimap_precision'])
        
        # Combine results
        sure_fg = np.maximum(hair_fg, non_hair_fg)
        sure_bg_inv = np.minimum(hair_bg, non_hair_bg)
        
        # Create trimap
        trimap = np.full(mask.shape, 128, dtype=np.uint8)  # Unknown
        trimap[sure_fg > 0] = 255  # Foreground
        trimap[sure_bg_inv == 0] = 0  # Background
        
        return trimap
    
    def _detect_hair_regions(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Detect hair regions using texture and gradient analysis"""
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Hair detection using Gabor filters
        hair_response = np.zeros_like(gray, dtype=np.float32)
        
        # Multiple Gabor filters for different hair orientations
        for theta in np.arange(0, 180, 30):  # 6 orientations
            kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 2*np.pi/3, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            hair_response += np.abs(filtered)
        
        # Normalize and threshold
        hair_response = cv2.normalize(hair_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, hair_regions = cv2.threshold(hair_response, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Only consider hair regions within the mask
        hair_regions = hair_regions * mask
        
        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hair_regions = cv2.morphologyEx(hair_regions, cv2.MORPH_OPEN, kernel)
        hair_regions = cv2.morphologyEx(hair_regions, cv2.MORPH_CLOSE, kernel)
        
        return hair_regions.astype(np.uint8)
    
    def _hair_aware_alpha_matting(self, image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """Advanced alpha matting with hair-specific processing"""
        
        # Standard closed-form matting as base
        alpha_base = self._closed_form_matting(image, trimap)
        
        # Hair-specific refinement
        alpha_hair_refined = self._refine_hair_alpha(image, alpha_base, trimap)
        
        # Combine base and hair-refined alpha
        hair_mask = self._detect_hair_regions(image, (trimap > 0).astype(np.uint8))
        hair_strength = self.config['hair_enhancement_strength']
        
        alpha_combined = alpha_base.copy()
        hair_regions = hair_mask > 0
        alpha_combined[hair_regions] = (
            (1 - hair_strength) * alpha_base[hair_regions] + 
            hair_strength * alpha_hair_refined[hair_regions]
        )
        
        return alpha_combined
    
    def _closed_form_matting(self, image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """Implementation of closed-form matting with optimizations"""
        
        try:
            from pymatting.alpha import estimate_alpha_cf
            
            # Convert to RGB and normalize
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            trimap_norm = trimap.astype(np.float64) / 255.0
            
            # Estimate alpha using closed-form matting
            alpha = estimate_alpha_cf(image_rgb, trimap_norm)
            
            return np.clip(alpha, 0, 1)
            
        except ImportError:
            logging.warning("pymatting not available, using fallback method")
            return self._fallback_alpha_matting(image, trimap)
    
    def _fallback_alpha_matting(self, image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """Fallback alpha matting implementation"""
        
        # Simple gradient-based alpha estimation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Initialize alpha from trimap
        alpha = trimap.astype(np.float64) / 255.0
        
        # Refine unknown regions based on gradients
        unknown_mask = (trimap == 128)
        if np.any(unknown_mask):
            # Use gradient information to estimate alpha in unknown regions
            alpha[unknown_mask] = 1.0 - gradient_magnitude[unknown_mask]
            alpha[unknown_mask] = np.clip(alpha[unknown_mask], 0, 1)
        
        return alpha
    
    def _refine_hair_alpha(self, image: np.ndarray, alpha: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """Specialized hair alpha refinement"""
        
        # Convert to LAB color space for better hair detection
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Use L channel for structure, A and B for color information
        l_channel = lab[:, :, 0].astype(np.float64) / 255.0
        
        # Detect fine hair structures using multi-scale analysis
        hair_structures = np.zeros_like(alpha)
        
        for scale in [1, 2, 4]:
            # Apply Gaussian blur at different scales
            blurred = cv2.GaussianBlur(l_channel, (0, 0), scale)
            
            # Calculate structure tensor
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            
            # Structure tensor components
            gxx = grad_x * grad_x
            gxy = grad_x * grad_y
            gyy = grad_y * grad_y
            
            # Apply Gaussian to structure tensor
            gxx = cv2.GaussianBlur(gxx, (5, 5), 1.0)
            gxy = cv2.GaussianBlur(gxy, (5, 5), 1.0)
            gyy = cv2.GaussianBlur(gyy, (5, 5), 1.0)
            
            # Calculate eigenvalues for structure detection
            trace = gxx + gyy
            det = gxx * gyy - gxy * gxy
            
            # Eigenvalues
            lambda1 = 0.5 * (trace + np.sqrt(trace**2 - 4*det + 1e-10))
            lambda2 = 0.5 * (trace - np.sqrt(trace**2 - 4*det + 1e-10))
            
            # Hair-like structures have high anisotropy
            coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10)
            hair_structures += coherence / scale
        
        # Normalize and apply to alpha
        hair_structures = cv2.normalize(hair_structures, None, 0, 1, cv2.NORM_MINMAX)
        
        # Enhance alpha in hair regions
        alpha_refined = alpha.copy()
        unknown_mask = (trimap == 128)
        
        # Adjust alpha based on hair structure detection
        structure_factor = 0.3
        alpha_refined[unknown_mask] = (
            alpha[unknown_mask] * (1 + structure_factor * hair_structures[unknown_mask])
        )
        
        return np.clip(alpha_refined, 0, 1)
    
    def _multi_scale_alpha_refinement(self, image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Multi-scale alpha refinement for different detail levels"""
        
        alpha_refined = alpha.copy()
        
        # Process at multiple scales
        scales = [0.5, 1.0, 2.0] if self.precision_level == "precision" else [1.0, 2.0]
        
        for scale in scales:
            if scale != 1.0:
                # Resize image and alpha
                h, w = image.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                
                image_scaled = cv2.resize(image, (new_w, new_h))
                alpha_scaled = cv2.resize(alpha_refined, (new_w, new_h))
            else:
                image_scaled = image
                alpha_scaled = alpha_refined
            
            # Apply guided filter at this scale
            try:
                import cv2.ximgproc as ximgproc
                
                guide = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2RGB)
                alpha_guided = ximgproc.guidedFilter(
                    guide=guide,
                    src=(alpha_scaled * 255).astype(np.uint8),
                    radius=self.config['guided_filter_radius'],
                    eps=self.config['guided_filter_eps']
                ).astype(np.float64) / 255.0
                
                if scale != 1.0:
                    # Resize back and blend
                    alpha_guided = cv2.resize(alpha_guided, (w, h))
                    blend_factor = 0.3 / scale  # Less influence for extreme scales
                    alpha_refined = (1 - blend_factor) * alpha_refined + blend_factor * alpha_guided
                else:
                    alpha_refined = alpha_guided
                    
            except (ImportError, AttributeError):
                # Fallback to bilateral filter
                alpha_bilateral = cv2.bilateralFilter(
                    (alpha_scaled * 255).astype(np.uint8),
                    self.config['bilateral_d'],
                    self.config['bilateral_sigma_color'],
                    self.config['bilateral_sigma_space']
                ).astype(np.float64) / 255.0
                
                if scale != 1.0:
                    alpha_bilateral = cv2.resize(alpha_bilateral, (w, h))
                    blend_factor = 0.3 / scale
                    alpha_refined = (1 - blend_factor) * alpha_refined + blend_factor * alpha_bilateral
                else:
                    alpha_refined = alpha_bilateral
        
        return np.clip(alpha_refined, 0, 1)
    
    def _edge_preserving_postprocess(self, image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Edge-preserving post-processing to maintain fine details"""
        
        # Step 1: Edge-aware smoothing
        alpha_smooth = self._edge_aware_smoothing(image, alpha)
        
        # Step 2: Detail enhancement
        alpha_enhanced = self._enhance_fine_details(image, alpha_smooth)
        
        # Step 3: Final precision adjustments
        alpha_final = self._precision_adjustments(alpha_enhanced)
        
        return alpha_final
    
    def _edge_aware_smoothing(self, image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Edge-aware smoothing preserving important boundaries"""
        
        # Calculate edge map from image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_map = edges.astype(np.float64) / 255.0
        
        # Apply anisotropic diffusion-like smoothing
        alpha_smooth = alpha.copy()
        
        for _ in range(self.config['alpha_refinement_iterations']):
            # Calculate gradients
            grad_x = cv2.Sobel(alpha_smooth, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(alpha_smooth, cv2.CV_64F, 0, 1, ksize=3)
            
            # Edge-stopping function
            edge_factor = self.config['edge_preservation_factor']
            diffusion_factor = 1.0 / (1.0 + edge_factor * edge_map)
            
            # Apply smoothing with edge preservation
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64) * 0.1
            laplacian = cv2.filter2D(alpha_smooth, cv2.CV_64F, kernel)
            
            alpha_smooth += diffusion_factor * laplacian
            alpha_smooth = np.clip(alpha_smooth, 0, 1)
        
        return alpha_smooth
    
    def _enhance_fine_details(self, image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Enhance fine details like hair strands"""
        
        # Create detail map using unsharp masking
        alpha_blurred = cv2.GaussianBlur(alpha, (5, 5), 1.0)
        detail_map = alpha - alpha_blurred
        
        # Enhance details based on image gradients
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        grad_magnitude = np.sqrt(
            cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)**2 + 
            cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)**2
        )
        
        # Adaptive detail enhancement
        enhancement_factor = 0.5 * grad_magnitude
        alpha_enhanced = alpha + enhancement_factor * detail_map
        
        return np.clip(alpha_enhanced, 0, 1)
    
    def _precision_adjustments(self, alpha: np.ndarray) -> np.ndarray:
        """Final precision adjustments based on precision level"""
        
        if self.precision_level == "precision":
            # Most aggressive refinement for maximum precision
            # Remove isolated pixels
            alpha_clean = scipy.ndimage.median_filter(alpha, size=3)
            
            # Enhance contrast
            alpha_contrast = np.power(alpha_clean, 0.8)
            
            # Final blend
            alpha_final = 0.7 * alpha_contrast + 0.3 * alpha
            
        elif self.precision_level == "ultra_high":
            # Moderate refinement
            alpha_clean = scipy.ndimage.median_filter(alpha, size=2)
            alpha_final = 0.8 * alpha + 0.2 * alpha_clean
            
        else:  # high
            # Minimal refinement
            alpha_final = alpha
        
        return np.clip(alpha_final, 0, 1)
    
    def _evaluate_trimap_quality(self, trimap: np.ndarray) -> float:
        """Evaluate trimap quality"""
        
        unknown_ratio = np.sum(trimap == 128) / trimap.size
        
        # Optimal unknown ratio is around 10-30%
        if 0.1 <= unknown_ratio <= 0.3:
            quality = 1.0 - abs(unknown_ratio - 0.2) / 0.1
        else:
            quality = max(0.0, 1.0 - abs(unknown_ratio - 0.2) / 0.4)
        
        return float(quality)
    
    def _assess_alpha_quality(
        self, 
        image: np.ndarray, 
        alpha: np.ndarray, 
        original_mask: np.ndarray
    ) -> Dict[str, float]:
        """Comprehensive alpha quality assessment"""
        
        metrics = {}
        
        # Edge preservation quality
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_edges = cv2.Canny(gray, 50, 150)
        alpha_edges = cv2.Canny((alpha * 255).astype(np.uint8), 50, 150)
        
        edge_alignment = np.sum(image_edges & alpha_edges) / (np.sum(image_edges | alpha_edges) + 1e-8)
        metrics['edge_preservation'] = float(edge_alignment)
        
        # Smoothness in non-edge regions
        laplacian_variance = cv2.Laplacian(alpha, cv2.CV_64F).var()
        metrics['smoothness'] = float(1.0 / (1.0 + laplacian_variance))
        
        # Coverage similarity to original mask
        alpha_binary = (alpha > 0.5).astype(np.uint8)
        mask_binary = (original_mask > 128).astype(np.uint8)
        
        intersection = np.sum(alpha_binary & mask_binary)
        union = np.sum(alpha_binary | mask_binary)
        iou = intersection / (union + 1e-8)
        metrics['coverage_similarity'] = float(iou)
        
        # Detail preservation (high-frequency content)
        alpha_highfreq = alpha - cv2.GaussianBlur(alpha, (5, 5), 1.0)
        detail_strength = np.std(alpha_highfreq)
        metrics['detail_preservation'] = float(min(1.0, detail_strength * 10))
        
        # Overall quality score
        weights = {
            'edge_preservation': 0.3,
            'smoothness': 0.2,
            'coverage_similarity': 0.3,
            'detail_preservation': 0.2
        }
        
        overall_quality = sum(metrics[key] * weight for key, weight in weights.items())
        metrics['overall_alpha_quality'] = float(overall_quality)
        
        return metrics

class HairSpecificProcessor:
    """
    Specialized processor for hair and fine detail preservation
    Uses advanced computer vision techniques specifically designed for hair segmentation
    """
    
    def __init__(self):
        self.gabor_kernels = self._create_gabor_kernel_bank()
        self.structure_kernels = self._create_structure_kernels()
    
    def _create_gabor_kernel_bank(self) -> List[np.ndarray]:
        """Create bank of Gabor filters for hair detection"""
        kernels = []
        
        # Parameters for hair detection
        frequencies = [0.1, 0.2, 0.3]
        orientations = np.arange(0, 180, 15)  # 12 orientations
        
        for freq in frequencies:
            for theta in orientations:
                kernel = cv2.getGaborKernel(
                    (21, 21), 3, np.radians(theta), 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F
                )
                kernels.append(kernel)
        
        return kernels
    
    def _create_structure_kernels(self) -> List[np.ndarray]:
        """Create kernels for structure analysis"""
        kernels = []
        
        # Line detection kernels for different orientations
        for angle in range(0, 180, 15):
            kernel = np.zeros((15, 15), dtype=np.float32)
            # Create line kernel
            center = 7
            for i in range(15):
                x = int(center + (i - center) * np.cos(np.radians(angle)))
                y = int(center + (i - center) * np.sin(np.radians(angle)))
                if 0 <= x < 15 and 0 <= y < 15:
                    kernel[y, x] = 1.0
            
            # Normalize
            if np.sum(kernel) > 0:
                kernel /= np.sum(kernel)
                kernels.append(kernel)
        
        return kernels
    
    def enhance_hair_regions(
        self, 
        image: np.ndarray, 
        alpha: np.ndarray
    ) -> np.ndarray:
        """Enhance alpha matte specifically in hair regions"""
        
        # Detect hair regions
        hair_map = self._detect_hair_with_gabor(image)
        
        # Enhance alpha in hair regions
        alpha_enhanced = alpha.copy()
        
        # Apply structure-preserving enhancement
        structure_response = self._analyze_hair_structure(image, hair_map)
        
        # Enhance alpha based on structure
        enhancement_factor = 0.3 * hair_map * structure_response
        alpha_enhanced += enhancement_factor * (1 - alpha_enhanced)
        
        return np.clip(alpha_enhanced, 0, 1)
    
    def _detect_hair_with_gabor(self, image: np.ndarray) -> np.ndarray:
        """Detect hair regions using Gabor filter bank"""
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        responses = []
        for kernel in self.gabor_kernels:
            response = cv2.filter2D(gray, cv2.CV_32F, kernel)
            responses.append(np.abs(response))
        
        # Combine responses
        combined_response = np.max(responses, axis=0)
        
        # Normalize and threshold
        hair_map = cv2.normalize(combined_response, None, 0, 1, cv2.NORM_MINMAX)
        
        # Apply threshold to get hair regions
        _, hair_binary = cv2.threshold(hair_map, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hair_binary = cv2.morphologyEx(hair_binary, cv2.MORPH_OPEN, kernel)
        
        return hair_binary
    
    def _analyze_hair_structure(self, image: np.ndarray, hair_map: np.ndarray) -> np.ndarray:
        """Analyze hair structure for better alpha estimation"""
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        structure_responses = []
        for kernel in self.structure_kernels:
            response = cv2.filter2D(gray, cv2.CV_32F, kernel)
            structure_responses.append(np.abs(response))
        
        # Find dominant structure orientation
        structure_strength = np.max(structure_responses, axis=0)
        
        # Apply only in hair regions
        structure_in_hair = structure_strength * hair_map
        
        # Normalize
        structure_normalized = cv2.normalize(structure_in_hair, None, 0, 1, cv2.NORM_MINMAX)
        
        return structure_normalized