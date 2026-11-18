"""
Annotate per-episode target (placement location) boxes for the magnify expert.

Step 1 --- dump the instruction -> target phrase mapping, then review / edit the json by hand (null = no target):
    python scripts/data/annotate_target_bbox.py phrases \
        --data_path $VLA_DATA_DIR/resize_224 --dataset libero_dataset \
        --phrases_path config/target_bbox/libero_phrases.json

Step 2 --- detect the phrase in the first frame of each episode with Grounding DINO:
    python scripts/data/annotate_target_bbox.py detect \
        --data_path $VLA_DATA_DIR/resize_224 --dataset libero_dataset \
        --phrases_path config/target_bbox/libero_phrases.json \
        --out_path config/target_bbox/libero_bbox.json --vis_dir $VLA_LOG_DIR/target_bbox_vis

Use the same data_path as training, since episodes are keyed by the encoded bytes of their first image.
The same phrases json and detector settings should be used at evaluation (scripts/eval_libero.py).
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image, ImageDraw

sys.path.append(os.path.expanduser("./"))

import src.data.dlimp as dl
from src.agent.target_locator import TargetLocator, parse_target_phrase
from src.data.oxe.oxe_dataset_configs import OXE_DATASET_CONFIGS
from src.data.oxe.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS
from src.data.utils.target_bbox import episode_key

tf.config.set_visible_devices([], "GPU")


def iterate_episodes(data_path: str, dataset: str):
    """Yields (episode key, first image bytes, instruction), matching `make_dataset_from_rlds.restructure`"""
    builder = tfds.builder(dataset, data_dir=data_path)
    image_key = OXE_DATASET_CONFIGS[dataset]["image_obs_keys"]["primary"]
    standardize_fn = OXE_STANDARDIZATION_TRANSFORMS[dataset]

    def first_frame(traj):
        traj = standardize_fn(traj)
        first_image = traj["observation"][image_key][0]
        return {
            "key": episode_key(first_image),
            "image": first_image,
            "instruction": traj["language_instruction"][0],
        }

    episodes = dl.DLataset.from_rlds(builder, split="all", shuffle=False).traj_map(
        first_frame
    )
    for episode in episodes.as_numpy_iterator():
        yield (
            str(episode["key"]),
            episode["image"],
            episode["instruction"].decode("utf-8"),
        )


def dump_phrases(args):
    instructions = Counter(
        instruction for _, _, instruction in iterate_episodes(args.data_path, args.dataset)
    )
    phrases = {
        instruction: parse_target_phrase(instruction)
        for instruction in sorted(instructions)
    }
    os.makedirs(os.path.dirname(args.phrases_path) or ".", exist_ok=True)
    with open(args.phrases_path, "w") as f:
        json.dump(phrases, f, indent=2)
    num_missing = sum(phrase is None for phrase in phrases.values())
    print(
        f"Saved {len(phrases)} instructions ({num_missing} without a phrase) to {args.phrases_path}. Review before detection."
    )


def draw_box(image: np.ndarray, bbox, text: str) -> Image.Image:
    pil_image = Image.fromarray(image)
    height, width = image.shape[:2]
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle(
        [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height],
        outline=(255, 0, 0),
        width=2,
    )
    draw.text((2, 2), text, fill=(255, 0, 0))
    return pil_image


def detect(args):
    locator = TargetLocator(
        model_id=args.model_id,
        phrases_path=args.phrases_path,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    annotations = {}
    cnt_total, cnt_no_phrase, cnt_no_detection = 0, 0, 0
    for key, image_bytes, instruction in iterate_episodes(args.data_path, args.dataset):
        cnt_total += 1
        if locator.get_phrase(instruction) is None:
            cnt_no_phrase += 1
            continue
        image = tf.io.decode_image(image_bytes, expand_animations=False).numpy()
        detection = locator.locate(image, instruction)
        if detection is None:
            cnt_no_detection += 1
            continue
        annotations[key] = {"instruction": instruction, **detection}
        if args.vis_dir and cnt_total % args.vis_every == 0:
            draw_box(
                image, detection["bbox"], f"{detection['phrase']} {detection['score']:.2f}"
            ).save(os.path.join(args.vis_dir, f"{cnt_total:06d}.png"))
        if cnt_total % 500 == 0:
            print(f"{cnt_total} episodes, {len(annotations)} annotated")

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "data_path": args.data_path,
                "model_id": args.model_id,
                "episodes": annotations,
            },
            f,
        )
    print(
        f"Annotated {len(annotations)} / {cnt_total} episodes (no phrase: {cnt_no_phrase}, no detection: {cnt_no_detection}). Saved to {args.out_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["phrases", "detect"])
    parser.add_argument(
        "--data_path", type=str, default=f"{os.environ.get('VLA_DATA_DIR', '')}/resize_224"
    )
    parser.add_argument("--dataset", type=str, default="libero_dataset")
    parser.add_argument(
        "--phrases_path", type=str, default="config/target_bbox/libero_phrases.json"
    )
    parser.add_argument(
        "--out_path", type=str, default="config/target_bbox/libero_bbox.json"
    )
    parser.add_argument(
        "--model_id", type=str, default="IDEA-Research/grounding-dino-base"
    )
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--vis_dir", type=str, default=None)
    parser.add_argument("--vis_every", type=int, default=50)
    args = parser.parse_args()

    if args.mode == "phrases":
        dump_phrases(args)
    else:
        detect(args)
