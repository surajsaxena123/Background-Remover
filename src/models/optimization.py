"""
Performance Optimization and Precision Enhancement Module
Advanced techniques for precision-grade background removal optimization
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import time
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available - some optimization features disabled")
except Exception as e:
    SKLEARN_AVAILABLE = False
    logging.warning(f"scikit-learn import failed: {e} - using fallback clustering")

try:
    import torch.jit
    TORCH_JIT_AVAILABLE = True
except ImportError:
    TORCH_JIT_AVAILABLE = False
    logging.warning("TorchScript not available - model optimization disabled")

class PrecisionOptimizer:
    """
    Advanced optimization techniques for precision-grade quality
    """
    
    def __init__(
        self,
        target_precision: str = "ultra_high",
        optimization_level: str = "aggressive",
        enable_gpu_acceleration: bool = True
    ):
        """
        Initialize precision optimizer
        
        Args:
            target_precision: Target precision level
            optimization_level: Optimization aggressiveness
            enable_gpu_acceleration: Use GPU acceleration if available
        """
        self.target_precision = target_precision
        self.optimization_level = optimization_level
        self.enable_gpu_acceleration = enable_gpu_acceleration
        
        self.device = self._get_optimal_device()
        self.precision_thresholds = self._get_precision_thresholds()
        
    def _get_optimal_device(self) -> str:
        """Get optimal device for processing"""
        if self.enable_gpu_acceleration:
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        return "cpu"
    
    def _get_precision_thresholds(self) -> Dict[str, float]:
        """Get precision thresholds based on target precision"""
        thresholds = {
            "medical": {
                "edge_quality": 0.95,
                "mask_coherence": 0.98,
                "gradient_consistency": 0.92,
                "boundary_smoothness": 0.90
            },
            "ultra_high": {
                "edge_quality": 0.90,
                "mask_coherence": 0.95,
                "gradient_consistency": 0.88,
                "boundary_smoothness": 0.85
            },
            "high": {
                "edge_quality": 0.85,
                "mask_coherence": 0.90,
                "gradient_consistency": 0.80,
                "boundary_smoothness": 0.75
            }
        }
        return thresholds.get(self.target_precision, thresholds["ultra_high"])
    
    def optimize_mask_precision(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        quality_metrics: Dict[str, float]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Optimize mask precision using advanced techniques
        
        Args:
            image: Original image
            mask: Input mask
            quality_metrics: Current quality metrics
            
        Returns:
            Tuple of (optimized_mask, updated_metrics)
        """
        optimized_mask = mask.copy()
        optimization_log = []
        
        # 1. Adaptive edge refinement
        if quality_metrics.get('edge_quality', 0) < self.precision_thresholds['edge_quality']:
            optimized_mask = self._adaptive_edge_refinement(image, optimized_mask)
            optimization_log.append("Applied adaptive edge refinement")
        
        # 2. Boundary smoothing optimization
        if quality_metrics.get('boundary_smoothness', 0) < self.precision_thresholds['boundary_smoothness']:
            optimized_mask = self._optimize_boundary_smoothness(optimized_mask)
            optimization_log.append("Applied boundary smoothing optimization")
        
        # 3. Gradient-guided refinement
        if quality_metrics.get('gradient_consistency', 0) < self.precision_thresholds['gradient_consistency']:
            optimized_mask = self._gradient_guided_refinement(image, optimized_mask)
            optimization_log.append("Applied gradient-guided refinement")
        
        # 4. Multi-scale coherence enhancement
        if quality_metrics.get('mask_coherence', 0) < self.precision_thresholds['mask_coherence']:
            optimized_mask = self._multi_scale_coherence_enhancement(optimized_mask)
            optimization_log.append("Applied multi-scale coherence enhancement")
        
        # 5. Precision-aware morphological operations
        optimized_mask = self._precision_morphological_operations(optimized_mask)
        optimization_log.append("Applied precision morphological operations")
        
        # Update quality metrics
        updated_metrics = quality_metrics.copy()
        updated_metrics['optimization_log'] = optimization_log
        updated_metrics['optimization_applied'] = len(optimization_log)
        
        return optimized_mask, updated_metrics
    
    def _adaptive_edge_refinement(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Adaptive edge refinement based on image gradients"""
        
        # Calculate multi-scale gradients
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gradients at different scales
        gradients = []
        for sigma in [0.5, 1.0, 2.0]:
            blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
            gradients.append(gradient_mag)
        
        # Combine gradients
        combined_gradient = np.mean(gradients, axis=0)
        gradient_norm = cv2.normalize(combined_gradient, None, 0, 1, cv2.NORM_MINMAX)
        
        # Adaptive threshold based on gradient strength
        edge_threshold = np.percentile(gradient_norm, 85)
        strong_edges = (gradient_norm > edge_threshold).astype(np.float32)
        
        # Refine mask using edge information
        mask_float = mask.astype(np.float32) / 255.0
        
        # Increase confidence near strong edges
        edge_enhanced = mask_float + 0.2 * strong_edges
        edge_enhanced = np.clip(edge_enhanced, 0, 1)
        
        # Apply guided filter for smooth refinement
        try:
            import cv2.ximgproc as ximgproc
            refined = ximgproc.guidedFilter(
                guide=gray, 
                src=(edge_enhanced * 255).astype(np.uint8), 
                radius=4, 
                eps=1e-8
            )
        except (ImportError, AttributeError):
            refined = cv2.bilateralFilter(
                (edge_enhanced * 255).astype(np.uint8), 9, 50, 50
            )
        
        return refined
    
    def _optimize_boundary_smoothness(self, mask: np.ndarray) -> np.ndarray:
        """Optimize boundary smoothness while preserving detail"""
        
        # Extract contours
        binary_mask = (mask > 128).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return mask
        
        # Process largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Apply adaptive smoothing
        epsilon = 0.001 * cv2.arcLength(largest_contour, True)
        smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Create smoothed mask
        smoothed_mask = np.zeros_like(mask)
        cv2.fillPoly(smoothed_mask, [smoothed_contour], 255)
        
        # Blend with original mask to preserve detail
        alpha = 0.7 if self.target_precision == "medical" else 0.5
        blended = cv2.addWeighted(mask, 1 - alpha, smoothed_mask, alpha, 0)
        
        return blended
    
    def _gradient_guided_refinement(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Gradient-guided mask refinement"""
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradient orientation
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        # Create direction-aware refinement
        mask_edges = cv2.Canny(mask, 50, 150)
        edge_pixels = np.where(mask_edges > 0)
        
        refined_mask = mask.copy().astype(np.float32)
        
        if len(edge_pixels[0]) > 0:
            # Align mask edges with image gradients
            for i, j in zip(edge_pixels[0], edge_pixels[1]):
                local_gradient = gradient_magnitude[max(0, i-2):min(mask.shape[0], i+3),
                                                   max(0, j-2):min(mask.shape[1], j+3)]
                
                if local_gradient.size > 0:
                    avg_gradient = np.mean(local_gradient)
                    # Adjust mask value based on gradient strength
                    adjustment = avg_gradient / (gradient_magnitude.max() + 1e-8)
                    refined_mask[i, j] *= (1 + 0.1 * adjustment)
        
        refined_mask = np.clip(refined_mask, 0, 255)
        return refined_mask.astype(np.uint8)
    
    def _multi_scale_coherence_enhancement(self, mask: np.ndarray) -> np.ndarray:
        """Multi-scale coherence enhancement"""
        
        # Process at multiple scales
        scales = [1.0, 0.5, 0.25] if self.target_precision == "medical" else [1.0, 0.5]
        scale_results = []
        
        for scale in scales:
            if scale != 1.0:
                # Resize mask
                new_size = (int(mask.shape[1] * scale), int(mask.shape[0] * scale))
                scaled_mask = cv2.resize(mask, new_size)
            else:
                scaled_mask = mask.copy()
            
            # Apply coherence operations
            binary_mask = (scaled_mask > 128).astype(np.uint8)
            
            # Remove small components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
            
            if num_labels > 1:
                # Keep only largest component
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                coherent_mask = (labels == largest_label).astype(np.uint8) * 255
            else:
                coherent_mask = scaled_mask
            
            # Resize back if needed
            if scale != 1.0:
                coherent_mask = cv2.resize(coherent_mask, (mask.shape[1], mask.shape[0]))
            
            scale_results.append(coherent_mask.astype(np.float32))
        
        # Combine multi-scale results
        if len(scale_results) > 1:
            weights = [0.6, 0.3, 0.1] if len(scale_results) == 3 else [0.7, 0.3]
            combined = np.zeros_like(scale_results[0])
            for result, weight in zip(scale_results, weights):
                combined += result * weight
        else:
            combined = scale_results[0]
        
        return np.clip(combined, 0, 255).astype(np.uint8)
    
    def _precision_morphological_operations(self, mask: np.ndarray) -> np.ndarray:
        """Precision-aware morphological operations"""
        
        # Adaptive kernel size based on precision level
        if self.target_precision == "medical":
            kernel_sizes = [(3, 3), (5, 5)]
            iterations = [1, 1]
        elif self.target_precision == "ultra_high":
            kernel_sizes = [(3, 3), (5, 5)]
            iterations = [2, 1]
        else:
            kernel_sizes = [(5, 5)]
            iterations = [2]
        
        refined_mask = mask.copy()
        
        for kernel_size, iter_count in zip(kernel_sizes, iterations):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            
            # Close operation to fill gaps
            refined_mask = cv2.morphologyEx(
                refined_mask, cv2.MORPH_CLOSE, kernel, iterations=iter_count
            )
            
            # Open operation to remove noise
            refined_mask = cv2.morphologyEx(
                refined_mask, cv2.MORPH_OPEN, kernel, iterations=max(1, iter_count // 2)
            )
        
        return refined_mask

class PerformanceAccelerator:
    """
    Performance acceleration for precision-grade processing
    """
    
    def __init__(
        self,
        enable_multiprocessing: bool = True,
        enable_gpu_acceleration: bool = True,
        batch_processing: bool = False
    ):
        """
        Initialize performance accelerator
        
        Args:
            enable_multiprocessing: Use multiprocessing for CPU operations
            enable_gpu_acceleration: Use GPU acceleration
            batch_processing: Enable batch processing for multiple images
        """
        self.enable_multiprocessing = enable_multiprocessing
        self.enable_gpu_acceleration = enable_gpu_acceleration
        self.batch_processing = batch_processing
        
        self.device = self._get_optimal_device()
        self.num_workers = min(mp.cpu_count(), 8) if enable_multiprocessing else 1
        
    def _get_optimal_device(self) -> str:
        """Get optimal device for processing"""
        if self.enable_gpu_acceleration:
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        return "cpu"
    
    def accelerate_processing(
        self,
        processing_func,
        inputs: Union[np.ndarray, List[np.ndarray]],
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Accelerate processing using available optimization techniques
        
        Args:
            processing_func: Function to accelerate
            inputs: Input data (single image or list of images)
            **kwargs: Additional arguments for processing function
            
        Returns:
            Processed results
        """
        
        if isinstance(inputs, list) and self.batch_processing:
            return self._batch_process(processing_func, inputs, **kwargs)
        elif isinstance(inputs, list) and self.enable_multiprocessing:
            return self._parallel_process(processing_func, inputs, **kwargs)
        else:
            # Single image processing with optimizations
            return self._optimized_single_process(processing_func, inputs, **kwargs)
    
    def _batch_process(
        self,
        processing_func,
        image_list: List[np.ndarray],
        **kwargs
    ) -> List[np.ndarray]:
        """Batch processing for multiple images"""
        
        # Group images by similar characteristics for efficient batching
        batched_groups = self._group_similar_images(image_list)
        
        results = []
        for group in batched_groups:
            # Process group with optimized parameters
            group_results = []
            for image in group:
                result = self._optimized_single_process(processing_func, image, **kwargs)
                group_results.append(result)
            results.extend(group_results)
        
        return results
    
    def _parallel_process(
        self,
        processing_func,
        image_list: List[np.ndarray],
        **kwargs
    ) -> List[np.ndarray]:
        """Parallel processing using multiple threads"""
        
        results = [None] * len(image_list)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._optimized_single_process, processing_func, img, **kwargs): i
                for i, img in enumerate(image_list)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logging.error(f"Processing failed for image {index}: {e}")
                    results[index] = None
        
        return [r for r in results if r is not None]
    
    def _optimized_single_process(
        self,
        processing_func,
        image: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Optimized single image processing"""
        
        # Pre-processing optimizations
        if self.enable_gpu_acceleration and self.device != "cpu":
            # Move to GPU if beneficial
            try:
                # Convert to tensor for GPU processing
                image_tensor = torch.from_numpy(image).to(self.device)
                # Process with GPU acceleration
                result = processing_func(image, **kwargs)
                return result
            except Exception as e:
                logging.warning(f"GPU processing failed, falling back to CPU: {e}")
        
        # CPU processing with optimizations
        return processing_func(image, **kwargs)
    
    def _group_similar_images(
        self,
        image_list: List[np.ndarray]
    ) -> List[List[np.ndarray]]:
        """Group similar images for efficient batch processing"""
        
        if not SKLEARN_AVAILABLE or len(image_list) < 3:
            return [image_list]  # Return single group if clustering not available
        
        # Extract features for clustering
        features = []
        for img in image_list:
            # Simple features: size, brightness, contrast
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            feature_vector = [
                h, w, h*w,  # Size features
                np.mean(gray),  # Brightness
                np.std(gray),   # Contrast
                np.percentile(gray, 95) - np.percentile(gray, 5)  # Dynamic range
            ]
            features.append(feature_vector)
        
        features_array = np.array(features)
        
        # Determine optimal number of clusters
        max_clusters = min(len(image_list) // 2, 5)
        if max_clusters < 2:
            return [image_list]
        
        best_score = -1
        best_k = 2
        
        for k in range(2, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features_array)
                score = silhouette_score(features_array, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue
        
        # Final clustering
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_array)
        
        # Group images by cluster
        groups = [[] for _ in range(best_k)]
        for i, label in enumerate(labels):
            groups[label].append(image_list[i])
        
        return [group for group in groups if group]  # Remove empty groups

class MemoryOptimizer:
    """
    Memory optimization for large-scale medical image processing
    """
    
    def __init__(
        self,
        max_image_size: Tuple[int, int] = (2048, 2048),
        enable_tiling: bool = True,
        memory_threshold: float = 0.8
    ):
        """
        Initialize memory optimizer
        
        Args:
            max_image_size: Maximum image size before tiling
            enable_tiling: Enable image tiling for large images
            memory_threshold: Memory usage threshold for optimization
        """
        self.max_image_size = max_image_size
        self.enable_tiling = enable_tiling
        self.memory_threshold = memory_threshold
        
    def optimize_image_processing(
        self,
        image: np.ndarray,
        processing_func,
        **kwargs
    ) -> np.ndarray:
        """
        Optimize image processing based on memory constraints
        
        Args:
            image: Input image
            processing_func: Processing function to apply
            **kwargs: Additional arguments
            
        Returns:
            Processed image
        """
        
        h, w = image.shape[:2]
        
        # Check if image needs tiling
        if (self.enable_tiling and 
            (h > self.max_image_size[0] or w > self.max_image_size[1])):
            return self._process_with_tiling(image, processing_func, **kwargs)
        else:
            return processing_func(image, **kwargs)
    
    def _process_with_tiling(
        self,
        image: np.ndarray,
        processing_func,
        overlap: int = 64,
        **kwargs
    ) -> np.ndarray:
        """Process large image using tiling approach"""
        
        h, w = image.shape[:2]
        tile_h, tile_w = self.max_image_size
        
        # Calculate number of tiles
        n_tiles_h = max(1, (h + tile_h - 1) // tile_h)
        n_tiles_w = max(1, (w + tile_w - 1) // tile_w)
        
        # Process tiles
        result = np.zeros_like(image)
        
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate tile boundaries with overlap
                start_h = max(0, i * tile_h - overlap)
                end_h = min(h, (i + 1) * tile_h + overlap)
                start_w = max(0, j * tile_w - overlap)
                end_w = min(w, (j + 1) * tile_w + overlap)
                
                # Extract tile
                tile = image[start_h:end_h, start_w:end_w]
                
                # Process tile
                processed_tile = processing_func(tile, **kwargs)
                
                # Calculate insertion boundaries (removing overlap)
                insert_start_h = start_h if i == 0 else start_h + overlap
                insert_end_h = end_h if i == n_tiles_h - 1 else end_h - overlap
                insert_start_w = start_w if j == 0 else start_w + overlap
                insert_end_w = end_w if j == n_tiles_w - 1 else end_w - overlap
                
                # Calculate corresponding region in processed tile
                tile_start_h = insert_start_h - start_h
                tile_end_h = tile_start_h + (insert_end_h - insert_start_h)
                tile_start_w = insert_start_w - start_w
                tile_end_w = tile_start_w + (insert_end_w - insert_start_w)
                
                # Insert processed tile
                result[insert_start_h:insert_end_h, insert_start_w:insert_end_w] = \
                    processed_tile[tile_start_h:tile_end_h, tile_start_w:tile_end_w]
        
        return result

class QualityAssurance:
    """
    Quality assurance and validation for precision-grade results
    """
    
    def __init__(
        self,
        min_quality_threshold: float = 0.9,
        enable_reference_checking: bool = True
    ):
        """
        Initialize quality assurance
        
        Args:
            min_quality_threshold: Minimum acceptable quality score
            enable_reference_checking: Enable reference image checking
        """
        self.min_quality_threshold = min_quality_threshold
        self.enable_reference_checking = enable_reference_checking
        
    def validate_result_quality(
        self,
        original_image: np.ndarray,
        result_image: np.ndarray,
        mask: np.ndarray,
        quality_metrics: Dict[str, float]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Comprehensive quality validation
        
        Args:
            original_image: Original input image
            result_image: Processed result image
            mask: Generated mask
            quality_metrics: Quality metrics from processing
            
        Returns:
            Tuple of (passes_quality_check, validation_report)
        """
        
        validation_report = {
            "timestamp": time.time(),
            "overall_pass": False,
            "individual_checks": {},
            "recommendations": []
        }
        
        # Individual quality checks
        checks = [
            ("overall_quality", self._check_overall_quality),
            ("edge_preservation", self._check_edge_preservation),
            ("artifact_detection", self._check_artifacts),
            ("mask_validity", self._check_mask_validity),
            ("color_consistency", self._check_color_consistency)
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            try:
                passed, details = check_func(
                    original_image, result_image, mask, quality_metrics
                )
                validation_report["individual_checks"][check_name] = {
                    "passed": passed,
                    "details": details
                }
                
                if not passed:
                    all_passed = False
                    
            except Exception as e:
                logging.error(f"Quality check {check_name} failed: {e}")
                validation_report["individual_checks"][check_name] = {
                    "passed": False,
                    "details": f"Check failed with error: {e}"
                }
                all_passed = False
        
        validation_report["overall_pass"] = all_passed
        
        # Generate recommendations if quality is insufficient
        if not all_passed:
            validation_report["recommendations"] = self._generate_recommendations(
                validation_report["individual_checks"]
            )
        
        return all_passed, validation_report
    
    def _check_overall_quality(
        self,
        original_image: np.ndarray,
        result_image: np.ndarray,
        mask: np.ndarray,
        quality_metrics: Dict[str, float]
    ) -> Tuple[bool, str]:
        """Check overall quality score"""
        
        overall_quality = quality_metrics.get("overall_quality", 0.0)
        passed = overall_quality >= self.min_quality_threshold
        
        details = f"Overall quality: {overall_quality:.3f} (threshold: {self.min_quality_threshold:.3f})"
        
        return passed, details
    
    def _check_edge_preservation(
        self,
        original_image: np.ndarray,
        result_image: np.ndarray,
        mask: np.ndarray,
        quality_metrics: Dict[str, float]
    ) -> Tuple[bool, str]:
        """Check edge preservation quality"""
        
        edge_alignment = quality_metrics.get("edge_alignment", 0.0)
        passed = edge_alignment >= 0.8  # Threshold for edge quality
        
        details = f"Edge alignment: {edge_alignment:.3f}"
        
        return passed, details
    
    def _check_artifacts(
        self,
        original_image: np.ndarray,
        result_image: np.ndarray,
        mask: np.ndarray,
        quality_metrics: Dict[str, float]
    ) -> Tuple[bool, str]:
        """Check for visual artifacts"""
        
        # Extract alpha channel
        if result_image.shape[2] == 4:
            alpha = result_image[:, :, 3]
        else:
            alpha = mask
        
        # Check for common artifacts
        artifacts_detected = []
        
        # 1. Check for jagged edges
        edges = cv2.Canny(alpha, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_boundary = cv2.arcLength(cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], True) if cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] else 0
        
        if total_boundary > 0:
            jaggedness = edge_pixels / total_boundary
            if jaggedness > 2.0:  # Threshold for jaggedness
                artifacts_detected.append(f"Jagged edges detected (score: {jaggedness:.2f})")
        
        # 2. Check for holes in the mask
        num_labels, labels = cv2.connectedComponents(alpha)
        if num_labels > 2:  # Background + more than one foreground component
            artifacts_detected.append(f"Multiple disconnected regions detected ({num_labels - 1} regions)")
        
        # 3. Check for halo effects
        # Dilate mask and check for sudden intensity changes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(alpha, kernel, iterations=1)
        boundary_region = dilated - alpha
        
        if result_image.shape[2] >= 3:
            rgb_result = result_image[:, :, :3]
            boundary_pixels = np.where(boundary_region > 0)
            
            if len(boundary_pixels[0]) > 0:
                boundary_values = rgb_result[boundary_pixels]
                if np.std(boundary_values) > 50:  # High variance indicates potential halo
                    artifacts_detected.append("Potential halo effect detected")
        
        passed = len(artifacts_detected) == 0
        details = "; ".join(artifacts_detected) if artifacts_detected else "No artifacts detected"
        
        return passed, details
    
    def _check_mask_validity(
        self,
        original_image: np.ndarray,
        result_image: np.ndarray,
        mask: np.ndarray,
        quality_metrics: Dict[str, float]
    ) -> Tuple[bool, str]:
        """Check mask validity"""
        
        issues = []
        
        # Check mask coverage
        coverage = np.sum(mask > 128) / mask.size
        if coverage < 0.05:
            issues.append("Mask coverage too low (< 5%)")
        elif coverage > 0.95:
            issues.append("Mask coverage too high (> 95%)")
        
        # Check mask coherence
        coherence = quality_metrics.get("mask_coherence", 0.0)
        if coherence < 0.8:
            issues.append(f"Low mask coherence ({coherence:.3f})")
        
        passed = len(issues) == 0
        details = "; ".join(issues) if issues else "Mask is valid"
        
        return passed, details
    
    def _check_color_consistency(
        self,
        original_image: np.ndarray,
        result_image: np.ndarray,
        mask: np.ndarray,
        quality_metrics: Dict[str, float]
    ) -> Tuple[bool, str]:
        """Check color consistency"""
        
        if result_image.shape[2] < 3:
            return True, "Color consistency check skipped (grayscale image)"
        
        # Extract foreground regions
        foreground_mask = mask > 128
        
        if np.sum(foreground_mask) == 0:
            return False, "No foreground detected for color consistency check"
        
        # Compare original and result colors in foreground
        orig_fg = original_image[foreground_mask]
        result_fg = result_image[foreground_mask][:, :3]  # RGB channels only
        
        # Calculate color difference
        color_diff = np.mean(np.abs(orig_fg.astype(float) - result_fg.astype(float)))
        
        passed = color_diff < 10  # Threshold for color consistency
        details = f"Average color difference: {color_diff:.2f}"
        
        return passed, details
    
    def _generate_recommendations(
        self,
        check_results: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on failed checks"""
        
        recommendations = []
        
        for check_name, result in check_results.items():
            if not result["passed"]:
                if check_name == "overall_quality":
                    recommendations.append("Consider using higher precision mode or ensemble methods")
                elif check_name == "edge_preservation":
                    recommendations.append("Apply additional edge refinement or use gradient-guided processing")
                elif check_name == "artifact_detection":
                    recommendations.append("Apply post-processing to reduce artifacts and smooth boundaries")
                elif check_name == "mask_validity":
                    recommendations.append("Review segmentation parameters or use alternative model")
                elif check_name == "color_consistency":
                    recommendations.append("Check alpha blending or color space conversion")
        
        return recommendations