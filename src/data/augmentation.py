"""
Military Tactical Data Augmentation Engine for Camouflaged Object Detection (COD)
Simulates realistic tactical military surveillance scenarios:
- Atmospheric Disturbances: Fog, smoke screens, dust storms, rain streaks.
- Optical & Sensor Aberrations: Thermal IR sensor grain, high-ISO noise, motion blur, heat defocus.
- Camouflage Specific Occlusions: Mesh net grids, foliage/branch cutout, shadow patterns.
- Photometric & Lighting Variations: Extreme contrast drop, dusk/dawn illumination, seasonal foliage color shifts.
"""

import random
from typing import Tuple, Union
import cv2
import numpy as np
import tensorflow as tf


# ==============================================================================
# Pure NumPy / OpenCV Implementations (for Offline Dataset Generator & Custom Pipelines)
# ==============================================================================

def apply_fog_smoke(image_uint8: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """
    Simulates military smoke screens and atmospheric fog/haze.
    Blends a smooth perlin-like / Gaussian noise haze layer with gray/whitish illumination.
    """
    h, w, c = image_uint8.shape
    scale = random.uniform(0.3, severity)

    # Generate low-frequency smooth fog pattern via downscaled blur
    fog_raw = np.random.uniform(180, 240, (h // 8, w // 8, 3)).astype(np.float32)
    fog_map = cv2.resize(fog_raw, (w, h), interpolation=cv2.INTER_CUBIC)
    fog_map = cv2.GaussianBlur(fog_map, (31, 31), 0)

    image_float = image_uint8.astype(np.float32)
    blended = image_float * (1.0 - scale) + fog_map * scale
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_rain_dust_streaks(image_uint8: np.ndarray, intensity: float = 0.4) -> np.ndarray:
    """
    Simulates rain streaks, desert sandstorm particles, or airborne debris.
    Generates oriented directional linear streaks across the image.
    """
    h, w, c = image_uint8.shape
    num_streaks = int(random.uniform(50, 300) * intensity)

    streak_layer = np.zeros((h, w, 3), dtype=np.uint8)
    angle = random.uniform(-30, 30)

    for _ in range(num_streaks):
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        length = random.randint(10, 35)
        rad = np.radians(angle)
        x2 = int(x1 + length * np.sin(rad))
        y2 = int(y1 + length * np.cos(rad))
        color = random.randint(160, 240)
        thickness = random.choice([1, 2])
        cv2.line(streak_layer, (x1, y1), (x2, y2), (color, color, color), thickness)

    streak_layer = cv2.GaussianBlur(streak_layer, (3, 3), 0)
    result = cv2.addWeighted(image_uint8, 0.85, streak_layer, 0.3, 0)
    return result


def apply_thermal_sensor_noise(image_uint8: np.ndarray, mode: str = "mixed") -> np.ndarray:
    """
    Simulates thermal camera sensor noise, high-ISO electro-optical sensor grain,
    and digital transmission noise (Gaussian + Salt & Pepper).
    """
    image_float = image_uint8.astype(np.float32)

    # Gaussian noise (thermal ISO grain)
    sigma = random.uniform(8.0, 25.0)
    gauss = np.random.normal(0, sigma, image_uint8.shape).astype(np.float32)
    noisy = image_float + gauss

    # Salt & Pepper noise (sensor bit drops)
    if mode in ["salt_pepper", "mixed"] and random.random() > 0.4:
        s_vs_p = 0.5
        amount = random.uniform(0.005, 0.02)
        # Salt
        num_salt = np.ceil(amount * image_uint8.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_uint8.shape]
        noisy[tuple(coords)] = 255
        # Pepper
        num_pepper = np.ceil(amount * image_uint8.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_uint8.shape]
        noisy[tuple(coords)] = 0

    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_motion_blur_defocus(image_uint8: np.ndarray) -> np.ndarray:
    """
    Simulates drone motion blur during high-speed reconnaissance or out-of-focus heat distortion.
    """
    if random.random() > 0.5:
        # Directional motion blur kernel
        kernel_size = random.choice([5, 7, 9, 11])
        kernel = np.zeros((kernel_size, kernel_size))
        if random.random() > 0.5:
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        else:
            kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        return cv2.filter2D(image_uint8, -1, kernel)
    else:
        # Gaussian defocus blur
        k_size = random.choice([5, 7, 9])
        sigma = random.uniform(1.0, 3.0)
        return cv2.GaussianBlur(image_uint8, (k_size, k_size), sigma)


def apply_mesh_foliage_cutout(image_uint8: np.ndarray, mask_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates camouflage net mesh, foliage/branch occlusions, and grid shadow patterns.
    Applies cutout patches or grid overlays to the image.
    (Leaves mask unchanged so model learns to detect camouflaged targets through foliage/nets).
    """
    h, w, c = image_uint8.shape
    augmented_img = image_uint8.copy()

    mode = random.choice(["grid_mesh", "random_cutout", "foliage_stripes"])

    if mode == "grid_mesh":
        # Camouflage net mesh pattern
        grid_size = random.randint(15, 30)
        thickness = random.randint(1, 3)
        mesh_color = random.choice([(40, 50, 30), (30, 30, 30), (80, 70, 50)])
        for x in range(0, w, grid_size):
            cv2.line(augmented_img, (x, 0), (x, h), mesh_color, thickness)
        for y in range(0, h, grid_size):
            cv2.line(augmented_img, (0, y), (w, y), mesh_color, thickness)

    elif mode == "random_cutout":
        # Block cutout patches simulating foliage coverage
        num_patches = random.randint(2, 5)
        for _ in range(num_patches):
            pw = random.randint(int(w * 0.08), int(w * 0.25))
            ph = random.randint(int(h * 0.08), int(h * 0.25))
            x1 = random.randint(0, w - pw)
            y1 = random.randint(0, h - ph)
            patch_color = random.choice([(35, 55, 25), (45, 40, 30), (0, 0, 0)])
            augmented_img[y1:y1 + ph, x1:x1 + pw] = patch_color

    elif mode == "foliage_stripes":
        # Branch shadow stripes
        num_stripes = random.randint(3, 8)
        for _ in range(num_stripes):
            x1 = random.randint(0, w)
            y1 = 0
            x2 = random.randint(0, w)
            y2 = h
            thickness = random.randint(5, 15)
            cv2.line(augmented_img, (x1, y1), (x2, y2), (20, 30, 15), thickness)

    return augmented_img, mask_uint8


def apply_tactical_lighting_shift(image_uint8: np.ndarray) -> np.ndarray:
    """
    Simulates extreme lighting conditions:
    - Dusk / Low-Light Surveillance Drop
    - Harsh Sunlight / Lens Flare Glare
    - Camouflage Spectrum Tint Shift (jungle green to autumn brown to desert tan)
    """
    img_hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)

    mode = random.choice(["dusk_shadows", "sunlight_glare", "color_tint"])

    if mode == "dusk_shadows":
        # Drop brightness & boost contrast (dusk/low-light)
        img_hsv[:, :, 2] *= random.uniform(0.5, 0.85)
        img_hsv[:, :, 1] *= random.uniform(0.8, 1.2)

    elif mode == "sunlight_glare":
        # High glare / overexposure
        img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * random.uniform(1.15, 1.45), 0, 255)
        img_hsv[:, :, 1] *= random.uniform(0.7, 0.9)

    elif mode == "color_tint":
        # Shift hue slightly (simulating seasonal foliage shifts)
        hue_shift = random.uniform(-15, 15)
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_shift) % 180

    img_hsv = np.clip(img_hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)


def apply_full_military_tactical_pipeline(
    image_uint8: np.ndarray,
    mask_uint8: np.ndarray,
    augmentation_prob: float = 0.85
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete Military Tactical Augmentation Pipeline for an RGB image and GT mask pair.
    Randomly combines multiple tactical scenario distortions.
    """
    if random.random() > augmentation_prob:
        return image_uint8, mask_uint8

    img = image_uint8.copy()
    mask = mask_uint8.copy()

    # 1. Atmospheric Disturbance (Fog/Smoke/Rain) - 40% probability
    if random.random() < 0.4:
        if random.random() > 0.5:
            img = apply_fog_smoke(img, severity=random.uniform(0.2, 0.5))
        else:
            img = apply_rain_dust_streaks(img, intensity=random.uniform(0.3, 0.6))

    # 2. Lighting & Spectrum Shifts - 50% probability
    if random.random() < 0.5:
        img = apply_tactical_lighting_shift(img)

    # 3. Foliage & Net Occlusion - 40% probability
    if random.random() < 0.4:
        img, mask = apply_mesh_foliage_cutout(img, mask)

    # 4. Motion Blur or Defocus - 40% probability
    if random.random() < 0.4:
        img = apply_motion_blur_defocus(img)

    # 5. Thermal / IR Sensor Noise - 35% probability
    if random.random() < 0.35:
        img = apply_thermal_sensor_noise(img)

    return img, mask


# ==============================================================================
# TensorFlow Native Implementations (for Online tf.data pipeline)
# ==============================================================================

def tf_random_military_fog_smoke(img_float: tf.Tensor) -> tf.Tensor:
    """TensorFlow native fog/smoke haze simulation on float [0, 1] tensor."""
    h = tf.shape(img_float)[0]
    w = tf.shape(img_float)[1]
    scale = tf.random.uniform((), minval=0.15, maxval=0.45)

    fog_raw = tf.random.uniform([h // 8, w // 8, 3], minval=0.7, maxval=0.95)
    fog_map = tf.image.resize(fog_raw, [h, w], method="bicubic")

    blended = img_float * (1.0 - scale) + fog_map * scale
    return tf.clip_by_value(blended, 0.0, 1.0)


def tf_random_thermal_noise(img_float: tf.Tensor) -> tf.Tensor:
    """TensorFlow native thermal high-ISO noise on float [0, 1] tensor."""
    std = tf.random.uniform((), minval=0.02, maxval=0.08)
    noise = tf.random.normal(tf.shape(img_float), mean=0.0, stddev=std)
    return tf.clip_by_value(img_float + noise, 0.0, 1.0)
