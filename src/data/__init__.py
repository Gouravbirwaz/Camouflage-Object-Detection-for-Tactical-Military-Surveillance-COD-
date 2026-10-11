"""
Data processing, augmentation, conversion, and tf.data pipelines for COD.
"""

from .augmentation import (
    apply_fog_smoke,
    apply_thermal_sensor_noise,
    apply_tactical_lighting_shift,
    apply_motion_blur_defocus,
    apply_mesh_foliage_cutout,
    apply_rain_dust_streaks,
    apply_full_military_tactical_pipeline,
    tf_random_military_fog_smoke,
    tf_random_thermal_noise,
)
from .dataset import (
    discover_dataset_pairs,
    compute_object_gradient_np,
    parse_sample,
    create_tf_dataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from .mask_converter import (
    convert_single_image,
    process_directory,
    find_jpg_files,
    run_gt_conversion,
)
from .offline_augmentation import (
    discover_raw_training_pairs,
    generate_augmented_dataset,
)

__all__ = [
    "apply_fog_smoke",
    "apply_thermal_sensor_noise",
    "apply_tactical_lighting_shift",
    "apply_motion_blur_defocus",
    "apply_mesh_foliage_cutout",
    "apply_rain_dust_streaks",
    "apply_full_military_tactical_pipeline",
    "tf_random_military_fog_smoke",
    "tf_random_thermal_noise",
    "discover_dataset_pairs",
    "compute_object_gradient_np",
    "parse_sample",
    "create_tf_dataset",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "convert_single_image",
    "process_directory",
    "find_jpg_files",
    "run_gt_conversion",
    "discover_raw_training_pairs",
    "generate_augmented_dataset",
]
