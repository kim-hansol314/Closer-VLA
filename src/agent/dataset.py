import logging

import tensorflow as tf

from src.data.dataset import make_interleaved_dataset
from src.data.dataset_torch import TorchRLDSDataset
from src.data.oxe import make_oxe_dataset_kwargs_and_weights
from src.utils.monitor import log_execution_time

tf.config.set_visible_devices([], "GPU")
log = logging.getLogger(__name__)


class TorchRLDSInterleavedDataset:
    @log_execution_time(log)
    def __init__(self, config, train=True):
        dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
            config.dataset_mix,
            config.data_path,
            load_proprio=config.load_proprio,
            load_camera_views=("primary",),
            load_depth=config.load_depth,
        )
        if config.get("target_bbox_path", None):
            for dataset_kwargs in dataset_kwargs_list:
                dataset_kwargs["target_bbox_path"] = config.target_bbox_path
        phase_label_kwargs = config.get("phase_labels", None)
        if phase_label_kwargs is not None:
            phase_label_kwargs = dict(phase_label_kwargs)
        dataset = make_interleaved_dataset(
            dataset_kwargs_list,
            sample_weights,
            train=train,
            split=config.get("split", None),
            shuffle_buffer_size=config.shuffle_buffer_size,
            batch_size=None,  # batching will be handles in PyTorch Dataloader object
            balance_weights=True,
            traj_transform_kwargs=dict(
                # goal_relabeling_strategy="uniform",   # no neeed for goal relabeling
                window_size=config.window_size,
                action_horizon=config.action_horizon,
                subsample_length=100,
                skip_unlabeled=config.skip_unlabeled,  # skip ones without language annotation
                phase_label_kwargs=phase_label_kwargs,
            ),
            frame_transform_kwargs=dict(
                image_augment_kwargs={
                    "primary": dict(
                        random_brightness=[0.1],
                        random_contrast=[0.9, 1.1],
                        random_saturation=[0.9, 1.1],
                        random_hue=[0.05],
                        augment_order=[
                            "random_brightness",
                            "random_contrast",
                            "random_saturation",
                            "random_hue",
                        ],
                    ),
                    "wrist": dict(
                        random_brightness=[0.1],
                        random_contrast=[0.9, 1.1],
                        random_saturation=[0.9, 1.1],
                        random_hue=[0.05],
                        augment_order=[
                            "random_brightness",
                            "random_contrast",
                            "random_saturation",
                            "random_hue",
                        ],
                    ),
                },
                resize_size=dict(
                    primary=(224, 224),
                    wrist=(224, 224),
                ),
                num_parallel_calls=config.num_parallel_calls,
            ),
            traj_transform_threads=config.traj_transform_threads,
            traj_read_threads=config.traj_read_threads,
        )

        # convert for torch
        self.dataset = TorchRLDSDataset(dataset, train=train)
