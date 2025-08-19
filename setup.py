"""Setup script for Precision Background Remover."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
try:
    with open('requirements.txt') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    pass

setup(
    name="precision-background-remover",
    version="1.0.0",
    author="Assignment Team",
    author_email="",
    description="State-of-the-art background removal with precision-grade quality",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/assignment/precision-background-remover",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "gpu": [
            "torch>=1.12.0",
            "torchvision>=0.13.0",
            "segment-anything",
            "ultralytics",
        ],
        "advanced": [
            "pymatting>=1.1.8",
            "albumentations>=1.3.0",
            "opencv-contrib-python>=4.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "precision-bg-remover=main:main",
            "pbr=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["**/*.py"],
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "background-removal",
        "image-processing",
        "computer-vision",
        "ai",
        "machine-learning",
        "segmentation",
        "alpha-matting",
        "precision",
    ],
    project_urls={
        "Bug Reports": "https://github.com/assignment/precision-background-remover/issues",
        "Source": "https://github.com/assignment/precision-background-remover",
        "Documentation": "https://github.com/assignment/precision-background-remover/blob/main/docs/",
    },
)