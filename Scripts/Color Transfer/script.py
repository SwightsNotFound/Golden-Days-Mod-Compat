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

    # Load input images
    source_raw = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
    target_raw = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)

    if source_raw is None or target_raw is None:
        print("Error: Could not load source or target image.")
        sys.exit(1)

    # Extract alpha
    source_alpha = extract_alpha(source_raw)
    target_alpha = extract_alpha(target_raw)

    # Convert to 3-channel BGR
    source_bgr = ensure_bgr(source_raw)
    target_bgr = ensure_bgr(target_raw)

    # Convert to LAB
    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Build masks of valid (opaque) pixels
    # We now mask BOTH source and target to remove missing pixels from BOTH
    if source_alpha is not None:
        source_mask = source_alpha > 0.01
    else:
        source_mask = np.ones(source_lab.shape[:2], dtype=bool)

    if target_alpha is not None:
        target_mask = target_alpha > 0.01
    else:
        target_mask = np.ones(target_lab.shape[:2], dtype=bool)

    # --- Compute statistics only on masked pixels ---
    s_mean, s_std, t_mean, t_std = [], [], [], []

    for i in range(3):
        # Source & target now both use masks
        s_vals = source_lab[:, :, i][source_mask]
        t_vals = target_lab[:, :, i][target_mask]

        # Avoid empty-mask errors
        if len(s_vals) == 0:
            s_vals = np.array([0], dtype=np.float32)
        if len(t_vals) == 0:
            t_vals = np.array([0], dtype=np.float32)

        s_mean.append(s_vals.mean())
        s_std.append(s_vals.std())
        t_mean.append(t_vals.mean())
        t_std.append(t_vals.std())

    s_mean = np.array(s_mean)
    s_std  = np.array(s_std)
    t_mean = np.array(t_mean)
    t_std  = np.array(t_std)

    # --- Apply Reinhard color transfer only to masked pixels ---
    result_lab = target_lab.copy()

    for i in range(3):
        s_std_i = max(s_std[i], 1e-6)
        t_std_i = max(t_std[i], 1e-6)

        # Only adjust masked pixels
        channel = result_lab[:, :, i]
        masked_vals = channel[target_mask]

        masked_vals = (masked_vals - t_mean[i]) * (s_std_i / t_std_i) + s_mean[i]
        result_lab[:, :, i][target_mask] = masked_vals

    # Convert back to BGR
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    # Preserve transparency exactly like before
    if target_alpha is not None:
        fully_transparent = target_alpha == 0
        result_bgr[fully_transparent] = target_bgr[fully_transparent]

        alpha_out = (target_alpha * 255).astype(np.uint8)
        result = cv2.merge([result_bgr, alpha_out])
    else:
        result = result_bgr

    cv2.imwrite(output_path, result)
    print(f"Color transfer complete! Output saved to:\n{output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py source.png target.png output.png")
        sys.exit(1)

    source_path, target_path, output_path = sys.argv[1:]

    outdir = os.path.dirname(output_path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    color_transfer(source_path, target_path, output_path)

