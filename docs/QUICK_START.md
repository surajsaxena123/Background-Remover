# Quick Start Guide

## Installation

```bash
# Install basic dependencies
pip install -r requirements.txt

# Optional: For enhanced functionality
pip install ultralytics segment-anything-hq
```

## Basic Usage

### Command Line
```bash
# Process an image
python main.py input.jpg output.png

# Test with sample image (verify path exists first)
python main.py "data/input/DKDR03423_16_KA4C41N8_BLK_Alt1 (2).jpg" result.png
```

### Web Interface
```bash
# Launch web interface
streamlit run app.py
# Open browser to: http://localhost:8501
```

## Available Options

```bash
# Different precision modes
python main.py input.jpg output.png --precision-mode ultra_high

# Available models: birefnet-general, birefnet-portrait, birefnet-hd, u2net
python main.py input.jpg output.png --model birefnet-general

# Quality analysis only
python main.py input.jpg --analyze-only

# Batch processing
python main.py input_folder/ output_folder/ --batch
```

## Python API
```python
from src.core.processing import remove_background_precision_grade
import cv2

image = cv2.imread('input.jpg')
result, metrics = remove_background_precision_grade(image)
cv2.imwrite('output.png', result)
```

## Notes

- **First run**: Downloads models automatically (requires internet)
- **Enhanced features**: Install optional packages for SAM2 and additional models
- **Memory**: Large images may require significant RAM

For complete documentation, see other guides in the `docs/` folder.