import collections
import dataclasses
import logging
import math
import pathlib
from typing import Optional

import cv2
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import torch
import tqdm
import tyro
import hydra

import sys
import os

# 添加路径到 sys.path
sys.path.append(os.path.expanduser("./"))

from src.model.vla.pizero import PiZeroInference
from scripts.utils.preprocess import convert_to_uint8, resize_with_pad
from scripts.utils.load_model import load_checkpoint
from omegaconf import OmegaConf, DictConfig

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model parameters
    #################################################################################################################

    resize_size: int = 224
    replan_steps: int = 4
    cfg_path: str = "./config/eval/libero.yaml"
    cfg: DictConfig = dataclasses.field(init=False)

    #################################################################################################################
    # Magnify target detection --- keep identical to scripts/data/annotate_target_bbox.py
    #################################################################################################################
    target_phrases_path: Optional[str] = "config/target_bbox/libero_phrases.json"
    detector_model_id: str = "IDEA-Research/grounding-dino-base"
    detector_box_threshold: float = 0.3
    detector_text_threshold: float = 0.25

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = f"/DATA/disk0/lyx/dataset/Libero/libero_whole_datasets/eval_fractal_ft_{task_suite_name}_100epoch"  # For spatial tasks

    seed: int = 7  # Random Seed (for reproducibility)

    def __post_init__(self):
        # Load YAML config into cfg field
        self.cfg: DictConfig = OmegaConf.load(self.cfg_path)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    # elif args.task_suite_name == "libero_90": 
    #     max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    
    # load pi0 model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    client = PiZeroInference(args.cfg, use_ddp=False)
    load_checkpoint(client, args.cfg.checkpoint_path)
    client = client.to(device)
    print(f'model loaded!  device: {device}')

    locator = None
    if client.use_magnify:
        from src.agent.target_locator import TargetLocator

        locator = TargetLocator(
            model_id=args.detector_model_id,
            phrases_path=args.target_phrases_path,
            box_threshold=args.detector_box_threshold,
            text_threshold=args.detector_text_threshold,
            device=device,
        )

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)): # range(num_tasks_in_suite)
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            target = None  # detected once per episode, after objects settle
            phase_pred, magnify_gate = None, None

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    img = convert_to_uint8(
                        resize_with_pad(img, args.resize_size, args.resize_size)
                    )

                    if locator is not None and target is None:
                        target = locator.locate(img, str(task_description)) or {}
                        logging.info(f"Target: {target}")

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        env_adapter = hydra.utils.instantiate(args.cfg.env.adapter)
                        dtype = torch.bfloat16 if args.cfg.get("use_bf16", False) else torch.float32

                        inputs = env_adapter.preprocess(img, obs, str(task_description))
                        causal_mask, position_ids = client.build_causal_mask_and_position_ids(
                            inputs["attention_mask"], dtype=dtype
                        )
                        image_text_proprio_mask, action_mask = (
                            client.split_full_mask_into_submasks(causal_mask)
                        )
                        model_inputs = {
                            "input_ids": inputs["input_ids"],
                            "pixel_values": inputs["pixel_values"].to(dtype),
                            "image_text_proprio_mask": image_text_proprio_mask,
                            "action_mask": action_mask,
                            "proprios": inputs["proprios"].to(dtype),
                            **{f"{name}_position_ids": ids for name, ids in position_ids.items()},
                        }
                        if client.use_magnify:
                            magnify_valid = torch.tensor(["bbox" in target])
                            magnify_inputs = client.prepare_magnify_inputs(
                                torch.as_tensor(img).permute(2, 0, 1)[None],
                                torch.tensor([target.get("bbox", [0.0, 0.0, 1.0, 1.0])]),
                                magnify_valid,
                            )
                            model_inputs["magnify_pixel_values"] = magnify_inputs["magnify_pixel_values"].to(dtype)
                            model_inputs["target_bboxes"] = magnify_inputs["target_bboxes"].to(dtype)
                            model_inputs["magnify_valid"] = magnify_valid
                        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                        # using bf16 shows ~0.001 difference in action inferred when using vs. not using kv cache (infer_action_naive, needs to pass in full causal_mask instead), if starting from the same initial noise. no difference when using float32
                        # https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535
                        with torch.inference_mode():
                            actions, phase_logits, magnify_gate = client.infer_action(
                                **model_inputs, return_phase=True
                            )
                        if phase_logits is not None:
                            phase_pred = int(phase_logits[0].argmax())
                        action_chunk = env_adapter.postprocess(actions[0].float().cpu().numpy())
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    # Save preprocessed image for replay video, with the target box (green if magnify is attended) and phase
                    replay_images.append(_annotate_frame(img, target, phase_pred, magnify_gate))

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout{episode_idx}_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _annotate_frame(img, target, phase_pred, magnify_gate):
    if not target and phase_pred is None:
        return img
    img = img.copy()
    if target and "bbox" in target:
        height, width = img.shape[:2]
        x0, y0, x1, y1 = target["bbox"]
        gate_on = magnify_gate is not None and bool(magnify_gate[0])
        cv2.rectangle(
            img,
            (int(x0 * width), int(y0 * height)),
            (int(x1 * width), int(y1 * height)),
            (0, 255, 0) if gate_on else (255, 0, 0),
            1,
        )
    if phase_pred is not None:
        cv2.putText(img, f"phase {phase_pred}", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
