import io
import json
import time
from typing import Dict, Any, Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from rembg import new_session

# Import with fallback handling
try:
    from src.core import (
        remove_background, 
        remove_background_precision_grade,
        analyze_image_quality,
        optimize_processing_parameters
    )
    ADVANCED_FEATURES = True
except Exception as e:
    st.error(f"Advanced features not available: {e}")
    try:
        from src.core import remove_background, analyze_image_quality
        ADVANCED_FEATURES = False
    except Exception:
        st.error("Core functionality not available. Please check installation.")
        ADVANCED_FEATURES = False


@st.cache_resource
def get_session(model: str):
    return new_session(model)


def display_quality_metrics(metrics: Dict[str, Any]) -> None:
    """Display comprehensive quality metrics in the Streamlit sidebar.
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        Quality metrics dictionary from precision-grade processing containing
        scores for overall quality, edge alignment, mask confidence, etc.
    """
    if not metrics:
        return
    
    st.sidebar.subheader("Quality Metrics")
    
    # Overall quality score
    if 'overall_quality' in metrics:
        quality_score = metrics['overall_quality']
        if quality_score >= 0.9:
            quality_color = "Green"
        elif quality_score >= 0.7:
            quality_color = "Yellow"
        else:
            quality_color = "Red"
        st.sidebar.metric("Overall Quality", f"{quality_score:.3f}", delta=None)
        st.sidebar.write(f"{quality_color} Quality Level")
    
    # Individual metrics
    if 'edge_alignment' in metrics:
        st.sidebar.metric("Edge Alignment", f"{metrics['edge_alignment']:.3f}")
    
    if 'mask_confidence' in metrics:
        st.sidebar.metric("Mask Confidence", f"{metrics['mask_confidence']:.3f}")
    
    if 'sam2_confidence' in metrics:
        st.sidebar.metric("SAM2 Confidence", f"{metrics['sam2_confidence']:.3f}")
    
    if 'ensemble_consistency' in metrics:
        st.sidebar.metric("Ensemble Consistency", f"{metrics['ensemble_consistency']:.3f}")
    
    # Expandable detailed metrics
    with st.sidebar.expander("📋 Detailed Metrics"):
        st.json(metrics)

def display_image_analysis(image_metrics: Dict[str, float]) -> None:
    """Display image quality analysis results with processing recommendations.
    
    Parameters
    ----------
    image_metrics : Dict[str, float]
        Image quality metrics including sharpness, contrast, brightness,
        edge density, and noise level from analyze_image_quality().
    """
    st.sidebar.subheader("🔍 Image Analysis")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.metric("Sharpness", f"{image_metrics.get('sharpness', 0):.1f}")
        st.metric("Contrast", f"{image_metrics.get('contrast', 0):.1f}")
    
    with col2:
        st.metric("Brightness", f"{image_metrics.get('brightness', 0):.1f}")
        st.metric("Edge Density", f"{image_metrics.get('edge_density', 0):.3f}")
    
    # Recommendations
    recommendations = []
    if image_metrics.get('sharpness', 0) < 100:
        recommendations.append("📌 Image appears blurry - using enhanced processing")
    if image_metrics.get('noise_level', 0) > 50:
        recommendations.append("📌 High noise detected - applying noise reduction")
    if image_metrics.get('edge_density', 0) > 0.1:
        recommendations.append("📌 High detail image - using precision processing")
    
    if recommendations:
        st.sidebar.write("**Recommendations:**")
        for rec in recommendations:
            st.sidebar.write(rec)

def main() -> None:
    """Main Streamlit application entry point for precision-grade background removal.
    
    Provides an interactive web interface with the following features:
    - Upload and process images with precision-grade quality
    - Real-time image quality analysis and recommendations
    - Configurable processing modes (Medical Grade vs Standard)
    - Advanced parameter tuning and automatic optimization
    - Comprehensive quality metrics display
    - Multiple download format options
    """
    st.set_page_config(
        page_title="Precision Background Remover",
        layout="wide"
    )
    
    st.title("Precision Background Remover")
    st.markdown("""
    **State-of-the-art background removal with precision-grade quality**  
    Powered by Precision SAM2, Enhanced BiRefNet, and advanced post-processing
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Processing mode selection
    processing_mode = st.sidebar.selectbox(
        "Processing Mode",
        ["Precision Grade", "Standard"],
        help="Precision Grade uses advanced AI models for maximum precision"
    )
    
    if processing_mode == "Precision Grade":
        precision_level = st.sidebar.selectbox(
            "Precision Level",
            ["Precision", "Ultra High", "High"],
            help="Higher precision requires more processing time"
        )
        
        # Advanced options
        with st.sidebar.expander("🔧 Advanced Options"):
            use_sam2 = st.checkbox("Use Precision SAM2", value=True, help="Latest segmentation AI")
            use_enhanced_birefnet = st.checkbox("Use Enhanced BiRefNet", value=True, help="Ensemble of multiple models")
            quality_validation = st.checkbox("Quality Validation", value=True, help="Comprehensive quality checks")
            auto_optimize = st.checkbox("Auto-optimize Parameters", value=True, help="Automatically adjust based on image")
    else:
        # Standard mode options
        model = st.sidebar.selectbox(
            "Model", 
            ["birefnet-general", "birefnet-general-lite", "u2net"], 
            index=0
        )
    
    # File upload
    uploaded = st.file_uploader(
        "Upload an image", 
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
    )
    
    if uploaded is not None:
        # Load and display input image
        input_image = Image.open(uploaded).convert("RGB")
        image_array = np.array(input_image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Input Image")
            st.image(input_image, caption="Original Image", use_container_width=True)
        
        # Analyze image quality
        with st.spinner("Analyzing image quality..."):
            image_metrics = analyze_image_quality(image_bgr)
        
        display_image_analysis(image_metrics)
        
        # Process image
        if st.button("🚀 Remove Background", type="primary"):
            with st.spinner("Processing image with precision-grade quality..."):
                start_time = time.time()
                
                if processing_mode == "Precision Grade" and ADVANCED_FEATURES:
                    # Auto-optimize parameters if enabled
                    if auto_optimize:
                        params = optimize_processing_parameters(
                            image_bgr, 
                            target_precision=precision_level.lower().replace(" ", "_")
                        )
                    else:
                        params = {
                            'precision_mode': precision_level.lower().replace(" ", "_"),
                            'use_sam2': use_sam2,
                            'use_enhanced_birefnet': use_enhanced_birefnet,
                            'quality_validation': quality_validation
                        }
                    
                    # Medical-grade processing
                    result_bgra, quality_metrics = remove_background_precision_grade(
                        image_bgr,
                        **params
                    )
                    
                    # Display quality metrics
                    display_quality_metrics(quality_metrics)
                    
                else:
                    # Standard processing (fallback for both standard mode and when advanced features unavailable)
                    if processing_mode == "Precision Grade" and not ADVANCED_FEATURES:
                        st.warning("Warning: Precision-grade features not available. Using standard processing.")
                    
                    session = get_session(model)
                    result_bgra = remove_background(image_bgr, session=session, model=model)
                    quality_metrics = {}
                
                processing_time = time.time() - start_time
                
                # Convert result for display
                result_rgba = cv2.cvtColor(result_bgra, cv2.COLOR_BGRA2RGBA)
                
                with col2:
                    st.subheader("📤 Result")
                    st.image(result_rgba, caption="Background Removed", use_container_width=True)
                
                # Processing info
                st.success(f"Processing completed in {processing_time:.2f} seconds")
                
                if processing_mode == "Precision Grade" and quality_metrics.get('overall_quality'):
                    quality_score = quality_metrics['overall_quality']
                    if quality_score >= 0.9:
                        st.success(f"🏆 Excellent quality achieved: {quality_score:.3f}")
                    elif quality_score >= 0.7:
                        st.info(f"👍 Good quality achieved: {quality_score:.3f}")
                    else:
                        st.warning(f"Warning: Quality could be improved: {quality_score:.3f}")
                
                # Download options
                col1_dl, col2_dl, col3_dl = st.columns(3)
                
                with col1_dl:
                    # PNG download
                    buffer_png = io.BytesIO()
                    Image.fromarray(result_rgba).save(buffer_png, format="PNG")
                    st.download_button(
                        "📥 Download PNG",
                        buffer_png.getvalue(),
                        file_name=f"precision_grade_removed_{uploaded.name.split('.')[0]}.png",
                        mime="image/png"
                    )
                
                with col2_dl:
                    # High-quality PNG download
                    buffer_hq = io.BytesIO()
                    result_pil = Image.fromarray(result_rgba)
                    result_pil.save(buffer_hq, format="PNG", compress_level=0)  # No compression
                    st.download_button(
                        "📥 Download HQ PNG",
                        buffer_hq.getvalue(),
                        file_name=f"precision_grade_hq_{uploaded.name.split('.')[0]}.png",
                        mime="image/png"
                    )
                
                with col3_dl:
                    # Quality metrics download
                    if quality_metrics:
                        metrics_json = json.dumps(quality_metrics, indent=2)
                        st.download_button(
                            "Download Metrics",
                            metrics_json,
                            file_name=f"quality_metrics_{uploaded.name.split('.')[0]}.json",
                            mime="application/json"
                        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>🔬 Precision-Grade Background Removal | Powered by Precision SAM2 & Enhanced BiRefNet</p>
    <p>📈 State-of-the-art AI models for maximum precision and quality</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
