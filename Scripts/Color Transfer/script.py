import cv2
import numpy as np
import sys
import os

def ensure_bgr(img):
    """Ensure the image has 3 BGR channels."""
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3]
    raise ValueError(f"Unsupported number of channels: {img.shape}")

def extract_alpha(img):
    """Return alpha channel (float32 0–1) or None."""
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, 3].astype(np.float32) / 255.0
    return None

def color_transfer(source_path, target_path, output_path):
    # Load images
    source_raw = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
    target_raw = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)

    if source_raw is None or target_raw is None:
        print("Error: Could not load source or target image.")
        sys.exit(1)

    # Extract alpha (if present)
    target_alpha = extract_alpha(target_raw)

    # Convert to BGR (without premultiplying)
    source_bgr = ensure_bgr(source_raw)
    target_bgr = ensure_bgr(target_raw)

    # Convert to LAB (uint8 expected)
    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Mask: only color-transfer opaque areas
    if target_alpha is not None:
        mask = target_alpha > 0.1  # Adjust this threshold as needed
    else:
        mask = np.ones(target_lab.shape[:2], dtype=bool)

    # Compute means and stds
    s_mean, s_std = [], []
    t_mean, t_std = [], []

    for i in range(3):
        s_vals = source_lab[:, :, i].ravel()
        t_vals = target_lab[:, :, i]
        t_masked = t_vals[mask]
        s_mean.append(s_vals.mean())
        s_std.append(s_vals.std())
        t_mean.append(t_masked.mean())
        t_std.append(t_masked.std())

    s_mean = np.array(s_mean)
    s_std = np.array(s_std)
    t_mean = np.array(t_mean)
    t_std = np.array(t_std)

    # Reinhard color transfer result_lab
    result_lab = target_lab.copy()
    for i in range(3):
        s_std_i = max(s_std[i], 1e-6)
        t_std_i = max(t_std[i], 1e-6)
        result_lab[..., i] = (result_lab[..., i] - t_mean[i]) * (s_std_i / t_std_i) + s_mean[i]

    # Convert back to BGR
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    # Reattach alpha cleanly (without premultiply)
    if target_alpha is not None:
        alpha_out = (target_alpha * 255).astype(np.uint8)
        result = cv2.merge([result_bgr, alpha_out])
    else:
        result = result_bgr

    # Save final PNG
    cv2.imwrite(output_path, result)
    print(f"Finished color transfer! Your result is in: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py source.png target.png output.png")
        sys.exit(1)

    source_path, target_path, output_path = sys.argv[1:4]

    outdir = os.path.dirname(output_path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    color_transfer(source_path, target_path, output_path)
