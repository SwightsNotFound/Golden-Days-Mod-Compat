import os
import cv2
import numpy as np
from collections import Counter

def color_transfer(source_path, target_path, output_path):
    source = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
    target = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)

    if source is None or target is None:
        print(f"❌ Error: Could not load one or both images ({source_path}, {target_path})")
        return

    if source.ndim == 3 and source.shape[2] == 4:
        source_rgb = source[:, :, :3].astype(np.float32)
        source_alpha = source[:, :, 3].astype(np.float32) / 255.0
    else:
        source_rgb, source_alpha = source.astype(np.float32), None

    if target.ndim == 3 and target.shape[2] == 4:
        target_rgb = target[:, :, :3].astype(np.float32)
        target_alpha = target[:, :, 3].astype(np.float32) / 255.0
    else:
        target_rgb, target_alpha = target.astype(np.float32), None

    if source_alpha is not None:
        source_rgb *= source_alpha[:, :, None]
    if target_alpha is not None:
        target_rgb *= target_alpha[:, :, None]

    source_lab = cv2.cvtColor(source_rgb.astype(np.uint8), cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_rgb.astype(np.uint8), cv2.COLOR_BGR2LAB)

    if target_alpha is not None:
        mask = target_alpha > 0.1
    else:
        mask = np.ones(target_lab.shape[:2], dtype=bool)

    s_mean, s_std, t_mean, t_std = [], [], [], []
    for i in range(3):
        s_vals = source_lab[:, :, i].astype(np.float32)
        t_vals = target_lab[:, :, i].astype(np.float32)

        s_mean.append(np.mean(s_vals))
        s_std.append(np.std(s_vals))
        t_mean.append(np.mean(t_vals[mask]))
        t_std.append(np.std(t_vals[mask]))

    s_mean, s_std = np.array(s_mean), np.array(s_std)
    t_mean, t_std = np.array(t_mean), np.array(t_std)

    result_lab = target_lab.copy().astype(np.float32)
    for i in range(3):
        t_std[i] = max(t_std[i], 1e-6)
        s_std[i] = max(s_std[i], 1e-6)
        result_lab[..., i] = ((target_lab[..., i] - t_mean[i]) *
                              (s_std[i] / t_std[i])) + s_mean[i]

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR).astype(np.float32)

    if target_alpha is not None:
        safe_alpha = np.clip(target_alpha, 1e-6, 1.0)
        result_bgr /= safe_alpha[:, :, None]
        result_bgr = np.clip(result_bgr, 0, 255)

    if target_alpha is not None:
        result = cv2.merge([
            result_bgr.astype(np.uint8),
            (target_alpha * 255).astype(np.uint8)
        ])
    else:
        result = result_bgr.astype(np.uint8)

    cv2.imwrite(output_path, result)
    print(f"✅ Color transfer complete! Saved as: {output_path}")

def process_images(input_folder, example_folder, output_folder):
    input_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for input_file in input_files:
        example_file = input_file.replace("_log", "_planks")
        example_image_path = os.path.join(example_folder, example_file)

        if not os.path.exists(example_image_path):
            print(f"⚠️  Warning: Example file {example_file} not found for {input_file}. Skipping.")
            continue

        input_image_path = os.path.join(input_folder, input_file)
        output_filename = f"stripped_{input_file}"
        output_image_path = os.path.join(output_folder, output_filename)
        color_transfer(example_image_path, input_image_path, output_image_path)

    print(f"\n🎨 Processing complete! Modified images saved to '{output_folder}'.")

if __name__ == "__main__":
    input_folder = 'input'
    example_folder = 'example'
    output_folder = 'output'

    os.makedirs(output_folder, exist_ok=True)
    process_images(input_folder, example_folder, output_folder)


