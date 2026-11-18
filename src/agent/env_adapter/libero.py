import json
import math
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
import torch
from transformers import AutoTokenizer

from src.agent.env_adapter.base import BaseEnvAdapter
from src.model.vla.processing import VLAProcessor
from src.utils.geometry import euler2axangle, mat2euler, quat2mat

class Adapter(BaseEnvAdapter):
    def __init__(
        self,
        dataset_statistics_path: str,
        pretrained_model_path: str,
        tokenizer_padding: str,
        num_image_tokens: int,
        image_size: Tuple[int, int],
        max_seq_len: int,
        action_normalization_type: str = "bound",
        proprio_normalization_type: str = "bound",
    ):
        super().__init__()
        self.image_size = tuple(image_size)
        self.action_normalization_type = action_normalization_type
        self.proprio_normalization_type = proprio_normalization_type
        assert action_normalization_type in ["bound", "gaussian"]
        assert proprio_normalization_type in ["bound", "gaussian"]

        # for normalization
        with tf.io.gfile.GFile(dataset_statistics_path, "r") as f:
            self.dataset_statistics = json.load(f)

        # tokenizer and processer --- assume paligemma for now
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path, padding_side="right"
        )
        self.processor = VLAProcessor(
            self.tokenizer,
            num_image_tokens=num_image_tokens,
            max_seq_len=max_seq_len,
            tokenizer_padding=tokenizer_padding,
        )

    def reset(self):
        pass

    def preprocess(
        self,
        image,
        obs: dict,
        instruction: str,
    ) -> dict:
        """using sxyz convention for euler angles"""
        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_LANCZOS4,
        )
        # no normalization for image before processor
        # always on cpu
        # (224, 224, 3) -> (1,3,224,224)
        images = torch.as_tensor(image, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)
        model_inputs = self.processor(text=[instruction], images=images)

        # process proprio depending on the robot
        raw_proprio = self.preprocess_proprio(obs)

        # normalize proprios - gripper opening is normalized
        if self.proprio_normalization_type == "bound":
            proprio = self.normalize_bound(
                raw_proprio,
                np.array(self.dataset_statistics["proprio"]["p01"]),
                np.array(self.dataset_statistics["proprio"]["p99"]),
                clip_min=-1,
                clip_max=1,
            )
        elif self.proprio_normalization_type == "gaussian":
            proprio = self.normalize_gaussian(
                raw_proprio,
                np.array(self.dataset_statistics["proprio"]["mean"]),
                np.array(self.dataset_statistics["proprio"]["std"]),
            )

        return {
            "input_ids": model_inputs["input_ids"],
            "pixel_values": model_inputs["pixel_values"],
            "attention_mask": model_inputs["attention_mask"],
            "proprios": torch.as_tensor(proprio, dtype=torch.float32)[
                None, None
            ],  # [B, T, dim]
        }

    def postprocess(
        self,
        actions: np.array,
    ) -> List[dict]:
        # gripper action is not normalized in training dataset
        if self.action_normalization_type == "bound":
            raw_actions_except_gripper = self.denormalize_bound(
                actions[:, :-1],
                np.array(self.dataset_statistics["action"]["p01"])[:-1],
                np.array(self.dataset_statistics["action"]["p99"])[:-1],
                clip_min=-1,
                clip_max=1,
            )
        elif self.action_normalization_type == "gaussian":
            raw_actions_except_gripper = self.denormalize_gaussian(
                actions[:, :-1],
                np.array(self.dataset_statistics["action"]["mean"])[:-1],
                np.array(self.dataset_statistics["action"]["std"])[:-1],
            )
        raw_actions = np.concatenate(
            [
                raw_actions_except_gripper,
                actions[:, -1:],
            ],
            axis=1,
        )

        # prepare for simpler env
        actions = np.zeros((len(raw_actions), 7))  # chunk
        for idx, raw_action in enumerate(raw_actions):
            roll, pitch, yaw = raw_action[3:6]
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            action_gripper = self.postprocess_gripper(raw_action[-1])

            actions[idx] = np.concatenate(
                [
                    raw_action[:3],
                    action_rotation_ax * action_rotation_angle,
                    [action_gripper],
                ]
            )
        return actions

    def preprocess_proprio(self, obs: dict) -> np.array:
        raise NotImplementedError

    def postprocess_gripper(self, action: float) -> float:
        raise NotImplementedError

class Libero_Adapter(Adapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self):
        super().reset()

    def preprocess_proprio(self, obs: dict) -> np.array:
        """
        Copied from openpi: https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py
        """
        def quat2axisangle(quat):
            """
            Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

            Converts quaternion to axis-angle format.
            Returns a unit vector direction scaled by its angle in radians.

            Args:
                quat (np.array): (x,y,z,w) vec4 float angles

            Returns:
                np.array: (ax,ay,az) axis-angle exponential coordinates
            """
            # clip quaternion
            if quat[3] > 1.0:
                quat[3] = 1.0
            elif quat[3] < -1.0:
                quat[3] = -1.0

            den = np.sqrt(1.0 - quat[3] * quat[3])
            if math.isclose(den, 0.0):
                # This is (close to) a zero degree rotation, immediately return
                return np.zeros(3)

            return (quat[:3] * 2.0 * math.acos(quat[3])) / den
        
        raw_proprio = np.concatenate(
            (
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
        return raw_proprio

    def postprocess_gripper(self, action: float) -> float:
        """from simpler octo inference: https://github.com/allenzren/SimplerEnv/blob/7d39d8a44e6d5ec02d4cdc9101bb17f5913bcd2a/simpler_env/policies/octo/octo_model.py#L234-L235"""
        # trained with [0, 1], 0 for close, 1 for open
        # convert to -1 close, 1 open for simpler
        action_gripper = 2.0 * (action > 0.5) - 1.0
        return action_gripper
