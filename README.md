# Background Remover

This repository demonstrates a complete pipeline for removing image backgrounds.  
The default segmentation is produced with [rembg](https://github.com/danielgatis/rembg) and then refined using a set of morphological operations and an optional guided filter to eliminate residual artefacts around hair and other fine details.

## Features
- Automatic foreground mask generation.
- Post-processing to clean edges and remove background traces.
- Streamlit application for interactive use and downloading results.
- Example data set with input images, raw model masks and the expected final output.
- GitHub Actions workflow that installs dependencies and runs a sample processing task.

## Project structure
```
├── app.py                 # Streamlit interface
├── src/
│   └── processing.py      # Mask generation and post‑processing logic
├── data/                  # Example input, model masks and expected outputs
├── requirements.txt       # Python dependencies
└── .github/workflows/
    └── deploy.yml         # Deployment pipeline
```

## Getting started
1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the Streamlit application
   ```bash
   streamlit run app.py
   ```
3. Upload an image to obtain a transparent PNG with the background removed.

The processing pipeline can also be used from the command line:
```bash
python -m src.processing path/to/input.jpg path/to/output.png
```

## Deployment
The workflow defined in `.github/workflows/deploy.yml` installs all dependencies and executes a sample background‑removal task. This file can serve as a starting point for automating deployment of the Streamlit application to your chosen hosting platform.

## License
This project is released under the MIT License.
