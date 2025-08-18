from typing import Optional

import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session
from pymatting.alpha import estimate_alpha_cf


def generate_mask(image: np.ndarray, session: Optional[object] = None) -> np.ndarray:
    """Generate a foreground mask for the given image using rembg."""
    if session is None:
        session = new_session()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    mask_image = remove(
        pil_image,
        session=session,
        only_mask=True,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
    )
    # rembg returns a PIL Image when given a PIL Image input
    mask = np.array(mask_image.convert("L"))
    return mask


def refine_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Refine the raw mask with morphological operations and optional guided filter."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)
    refined = cv2.GaussianBlur(refined, (5, 5), 0)

    # Attempt to use guided filter if available for better edge quality
    try:
        import cv2.ximgproc as ximgproc
        guide = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        refined = ximgproc.guidedFilter(guide=guide, src=refined, radius=8, eps=1e-6)
    except Exception:
        pass

    refined = cv2.normalize(refined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, refined = cv2.threshold(refined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return refined


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply closed-form alpha matting for high-precision edges."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = cv2.erode(mask, kernel, iterations=5)
    bg = cv2.dilate(mask, kernel, iterations=5)
    trimap = np.full(mask.shape, 128, dtype=np.uint8)
    trimap[fg > 200] = 255
    trimap[bg < 55] = 0
    alpha = estimate_alpha_cf(image_rgb, trimap / 255.0)
    comp = image_rgb * alpha[..., None]
    rgba = np.dstack((comp, alpha))
    return cv2.cvtColor((rgba * 255).astype(np.uint8), cv2.COLOR_RGBA2BGRA)


def remove_background(image: np.ndarray, session: Optional[object] = None) -> np.ndarray:
    """Full pipeline to remove background from an image."""
    mask = generate_mask(image, session=session)
    mask = refine_mask(image, mask)
    return apply_mask(image, mask)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Remove image background with post-processing")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("output_image", help="Path to save the processed image")
    args = parser.parse_args()

    image = cv2.imread(args.input_image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.input_image)
    result = remove_background(image)
    cv2.imwrite(args.output_image, result)


if __name__ == "__main__":
    main()
