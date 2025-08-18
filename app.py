import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.processing import remove_background


def main() -> None:
    st.title("Background Remover")
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        input_image = Image.open(uploaded).convert("RGB")
        image_array = np.array(input_image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        result_bgra = remove_background(image_bgr)
        result_rgba = cv2.cvtColor(result_bgra, cv2.COLOR_BGRA2RGBA)
        st.image(result_rgba, caption="Processed Image", use_container_width=True)

        buffer = io.BytesIO()
        Image.fromarray(result_rgba).save(buffer, format="PNG")
        st.download_button(
            "Download", buffer.getvalue(), file_name="background_removed.png", mime="image/png"
        )


if __name__ == "__main__":
    main()
