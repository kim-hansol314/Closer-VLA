"""
Per-episode target (placement location) bounding boxes for the magnify expert.

Boxes are produced offline by scripts/data/annotate_target_bbox.py and stored in a json file keyed by a hash of
the episode's first primary image (the encoded bytes as stored in the RLDS dataset), so no dataset rewrite is
needed. The key must be computed on the same dataset (e.g., the resize_224 version) used for training.

json format:
{
    "dataset": "libero_dataset",
    "episodes": {"<episode key>": {"bbox": [x0, y0, x1, y1], "instruction": ..., "phrase": ..., "score": ...}},
}
Boxes are normalized to [0, 1]; episodes without a detection are omitted.
"""

import json
from typing import Tuple

import tensorflow as tf


def episode_key(first_image: tf.Tensor) -> tf.Tensor:
    """int64 hash of an episode's first image (encoded bytes or raw uint8)"""
    if first_image.dtype != tf.string:
        first_image = tf.io.serialize_tensor(first_image)
    return tf.strings.to_hash_bucket_fast(first_image, 2**62)


def load_target_bbox_table(
    path: str,
) -> Tuple[tf.lookup.StaticHashTable, tf.Tensor]:
    with tf.io.gfile.GFile(path, "r") as f:
        episodes = json.load(f)["episodes"]
    keys = [int(k) for k in episodes]
    boxes = [v["bbox"] for v in episodes.values()]
    if not keys:
        raise ValueError(f"No target boxes found in {path}")
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(keys, dtype=tf.int64),
            tf.range(len(keys), dtype=tf.int32),
        ),
        default_value=-1,
    )
    return table, tf.constant(boxes, dtype=tf.float32)


def lookup_target_bbox(
    table: tf.lookup.StaticHashTable,
    boxes: tf.Tensor,
    first_image: tf.Tensor,
    traj_len: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Returns the box [traj_len, 4] (full image if missing) and validity [traj_len] for one trajectory"""
    idx = table.lookup(episode_key(first_image))
    valid = idx >= 0
    box = tf.cond(
        valid,
        lambda: tf.gather(boxes, tf.maximum(idx, 0)),
        lambda: tf.constant([0.0, 0.0, 1.0, 1.0]),
    )
    return tf.repeat(box[None], traj_len, axis=0), tf.repeat(valid, traj_len)
