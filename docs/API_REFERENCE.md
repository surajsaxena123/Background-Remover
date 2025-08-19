# API Reference - Precision Background Remover

## Overview

The Precision Background Remover provides a comprehensive API for state-of-the-art background removal with precision-grade quality. This document covers all available functions, classes, and their usage.

## Core Module (`src.core`)

### Background Removal Functions

#### `remove_background_precision_grade()`

The main function for precision-grade background removal.

```python
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
) -> Tuple[np.ndarray, Dict[str, Any]]
```

**Parameters:**
- `image` (np.ndarray): Input image in BGR format
- `precision_mode` (str): Quality level ('high', 'ultra_high', 'precision')
- `use_sam2` (bool): Enable Precision SAM2 segmentation
- `use_enhanced_birefnet` (bool): Enable Enhanced BiRefNet processing
- `quality_validation` (bool): Enable comprehensive quality metrics
- `session` (Optional[object]): Reusable rembg session for efficiency
- `model` (str): Model variant ('birefnet-general-hd', 'birefnet-portrait', etc.)
- `alpha_matting` (bool): Enable alpha matting for better edges
- `post_processing` (bool): Enable post-processing refinement
- `enable_hair_enhancement` (bool): Enable specialized hair processing

**Returns:**
- `Tuple[np.ndarray, Dict[str, Any]]`: Processed image and quality metrics

**Example:**
```python
import cv2
from src.core import remove_background_precision_grade

image = cv2.imread('input.jpg')
result, metrics = remove_background_precision_grade(
    image,
    precision_mode='ultra_high',
    enable_hair_enhancement=True
)

print(f"Quality Score: {metrics['quality_score']:.3f}")
cv2.imwrite('output.png', result)
```

#### `remove_background()`

Standard background removal function for basic usage.

```python
def remove_background(
    image: np.ndarray,
    session: Optional[object] = None,
    model: str = "birefnet-general",
) -> np.ndarray
```

**Parameters:**
- `image` (np.ndarray): Input image in BGR format
- `session` (Optional[object]): Reusable rembg session
- `model` (str): Model variant to use

**Returns:**
- `np.ndarray`: Processed image with background removed

#### `analyze_image_quality()`

Analyze input image quality for optimal processing parameters.

```python
def analyze_image_quality(image: np.ndarray) -> Dict[str, float]
```

**Parameters:**
- `image` (np.ndarray): Input image in BGR format

**Returns:**
- `Dict[str, float]`: Quality metrics including:
  - `sharpness`: Laplacian variance indicating image sharpness
  - `contrast`: Standard deviation of pixel intensities
  - `brightness`: Mean pixel intensity
  - `edge_density`: Proportion of edge pixels
  - `noise_level`: Estimated noise level

**Example:**
```python
metrics = analyze_image_quality(image)
print(f"Image sharpness: {metrics['sharpness']:.2f}")
print(f"Edge density: {metrics['edge_density']:.3f}")
```

#### `optimize_processing_parameters()`

Automatically optimize processing parameters based on image characteristics.

```python
def optimize_processing_parameters(
    image: np.ndarray, 
    target_precision: str = "ultra_high"
) -> Dict[str, Any]
```

**Parameters:**
- `image` (np.ndarray): Input image for analysis
- `target_precision` (str): Target precision level

**Returns:**
- `Dict[str, Any]`: Optimized parameters for processing

### Advanced Processing Functions

#### `apply_precision_alpha_matting()`

Apply precision-grade alpha matting with hair enhancement.

```python
def apply_precision_alpha_matting(
    image: np.ndarray, 
    mask: np.ndarray, 
    precision_mode: str,
    enable_hair_enhancement: bool = True
) -> np.ndarray
```

#### `combine_masks_advanced()`

Advanced mask combination using weighted ensemble.

```python
def combine_masks_advanced(
    masks: List[np.ndarray], 
    confidences: List[float], 
    precision_mode: str
) -> np.ndarray
```

## Precision SAM2 Module (`src.core.precision_sam2`)

### PrecisionSAM2Segmentor

Advanced segmentation using SAM2 with precision-grade quality.

```python
class PrecisionSAM2Segmentor:
    def __init__(
        self,
        model_type: str = "sam2_hiera_large",
        device: Optional[str] = None,
        confidence_threshold: float = 0.95,
        edge_refinement: bool = True
    )
```

**Methods:**

#### `generate_high_precision_mask()`

```python
def generate_high_precision_mask(
    self,
    image: np.ndarray,
    prompts: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, float]
```

**Parameters:**
- `image` (np.ndarray): Input image in BGR format
- `prompts` (Optional[Dict]): Optional prompts for guided segmentation

**Returns:**
- `Tuple[np.ndarray, float]`: Generated mask and confidence score

### PrecisionQualityValidator

Comprehensive quality validation for precision-grade results.

```python
class PrecisionQualityValidator:
    def __init__(
        self, 
        min_dice_score: float = 0.95, 
        min_edge_quality: float = 0.90
    )
```

**Methods:**

#### `validate_segmentation_quality()`

```python
def validate_segmentation_quality(
    self,
    image: np.ndarray,
    mask: np.ndarray,
    reference_mask: Optional[np.ndarray] = None
) -> Dict[str, float]
```

**Returns:**
- `Dict[str, float]`: Comprehensive quality metrics including:
  - `mask_coverage`: Proportion of mask coverage
  - `mask_coherence`: Connected component analysis
  - `edge_quality`: Gradient alignment quality
  - `gradient_consistency`: Boundary gradient consistency
  - `boundary_smoothness`: Contour smoothness analysis
  - `overall_quality`: Weighted composite score

## Precision Matting Module (`src.core.precision_matting`)

### PrecisionMattingEngine

State-of-the-art precision matting engine for hair and fine detail preservation.

```python
class PrecisionMattingEngine:
    def __init__(
        self, 
        precision_level: str = "ultra_high", 
        device: str = "cpu"
    )
```

**Methods:**

#### `generate_precision_alpha()`

```python
def generate_precision_alpha(
    self, 
    image: np.ndarray, 
    initial_mask: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]
```

**Parameters:**
- `image` (np.ndarray): Input image in BGR format
- `initial_mask` (np.ndarray): Initial segmentation mask

**Returns:**
- `Tuple[np.ndarray, Dict[str, float]]`: Precision alpha matte and quality metrics

### HairSpecificProcessor

Specialized processor for hair and fine detail preservation.

```python
class HairSpecificProcessor:
    def __init__(self)
```

**Methods:**

#### `enhance_hair_regions()`

```python
def enhance_hair_regions(
    self, 
    image: np.ndarray, 
    alpha: np.ndarray
) -> np.ndarray
```

**Parameters:**
- `image` (np.ndarray): Input image
- `alpha` (np.ndarray): Initial alpha matte

**Returns:**
- `np.ndarray`: Enhanced alpha matte with improved hair details

## Model Implementations (`src.models`)

### EnhancedBiRefNet

Enhanced BiRefNet implementation with ensemble methods and precision optimizations.

```python
class EnhancedBiRefNet:
    def __init__(
        self,
        model_variant: str = "birefnet-general-hd",
        precision_mode: str = "ultra_high",
        device: Optional[str] = None,
        use_ensemble: bool = True,
        enable_tta: bool = True
    )
```

**Methods:**

#### `generate_precision_mask()`

```python
def generate_precision_mask(
    self,
    image: np.ndarray,
    alpha_matting: bool = True,
    post_process: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]
```

### PrecisionAlphaMatting

Advanced alpha matting implementation for precision-grade edge refinement.

```python
class PrecisionAlphaMatting:
    def __init__(self, precision_level: str = "ultra_high")
```

## Utility Functions (`src.utils`)

### Mac Optimization

Functions for Mac-specific optimizations.

```python
def detect_mac_architecture() -> Dict[str, Any]
def initialize_mac_optimizations() -> Dict[str, Any]
def get_optimal_device() -> str
def optimize_for_mac_processing(image_size: Tuple[int, int]) -> Dict[str, Any]
```

## Command Line Interface

### Main Entry Point

The `main.py` script provides a comprehensive command-line interface:

```bash
python main.py input.jpg output.png [OPTIONS]
```

**Common Options:**
- `--precision-mode {high,ultra_high,precision}`: Set precision level
- `--model {birefnet-general,birefnet-portrait,birefnet-hd}`: Choose model
- `--use-sam2`: Enable Precision SAM2
- `--enable-hair-enhancement`: Enable hair processing
- `--analyze-only`: Only analyze image quality
- `--batch`: Process directory of images
- `--auto-optimize`: Automatically optimize parameters

**Examples:**
```bash
# Basic processing
python main.py input.jpg output.png

# High precision with hair enhancement
python main.py input.jpg output.png --precision-mode precision --enable-hair-enhancement

# Batch processing
python main.py input_dir/ output_dir/ --batch

# Quality analysis only
python main.py input.jpg --analyze-only
```

## Error Handling

All functions include comprehensive error handling and will:
- Log warnings for missing optional dependencies
- Gracefully fallback to available methods
- Provide detailed error messages for troubleshooting

## Performance Considerations

- Use `session` parameter to reuse model instances for batch processing
- Enable `auto_optimize` for best quality/speed trade-off
- Consider `precision_mode='high'` for faster processing
- Use GPU acceleration when available (CUDA/MPS)

## Quality Metrics

The system provides comprehensive quality metrics:

- **Quality Score (0.0-1.0)**: Overall quality assessment
- **Edge Preservation**: How well edges are maintained
- **Coverage Similarity**: Accuracy compared to ground truth
- **Detail Preservation**: Fine detail retention
- **Processing Time**: Performance metrics
- **Model Confidence**: Individual model confidence scores