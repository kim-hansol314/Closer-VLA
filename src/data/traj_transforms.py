"""
Contains trajectory transforms used in the octo data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory
length).
"""

from typing import Optional

import tensorflow as tf


def chunk_act_obs(
    traj: dict,
    window_size: int = 1,
    action_horizon: int = 1,
) -> dict:
    """Chunks actions and observations.

    "observation" keys are given a new history axis, making them of shape [traj_len, window_size, ...],
    containing the observation history at each timestep from `t - window_size + 1` to `t`.

    The "action" key is given two new axes, making it of shape [traj_len, window_size, action_horizon,
    action_dim]. The first two axes are the same as in the observations, i.e., an action chunk `action[t, h]`
    corresponds to an observation `observation[t, h]`. The third axis indexes into the action chunk,
    containing the current action plus `action_horizon - 1` future actions.

    The "action" key can also be pre-chunked coming into this function, meaning it starts with shape
    [traj_len, N, action_dim] instead of [traj_len, action_dim]. In this case, `N` must be larger than or
    equal to `action_horizon`, and only one axis will be added (the history axis). This is useful for
    custom chunking schemes where an action may differ depending on which observation it is paired with.
    """
    traj_len = tf.shape(traj["action"])[0]

    # chunk observations into histories
    history_indices = tf.range(traj_len)[:, None] + tf.range(
        -window_size + 1, 1
    )  # [traj_len, window_size]
    # indicates which observations at the beginning of the trajectory are padding
    timestep_pad_mask = history_indices >= 0
    # repeat the first observation at the beginning of the trajectory rather than going out of bounds
    history_indices = tf.maximum(history_indices, 0)
    # gather
    traj["observation"] = tf.nest.map_structure(
        lambda x: tf.gather(x, history_indices), traj["observation"]
    )  # [traj_len, window_size, ...]
    traj["observation"]["timestep_pad_mask"] = timestep_pad_mask

    # first, chunk actions into `action_horizon` current + future actions
    if len(traj["action"].shape) == 2:
        # actions are not pre-chunked
        action_chunk_indices = tf.range(traj_len)[:, None] + tf.range(
            action_horizon
        )  # [traj_len, action_horizon]
        # repeat the last action at the end of the trajectory rather than going out of bounds
        action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)
        # gather
        traj["action"] = tf.gather(
            traj["action"], action_chunk_indices
        )  # [traj_len, action_horizon, action_dim]
    else:
        # actions are pre-chunked, so we don't add a new axis
        if traj["action"].shape[1] < action_horizon:
            raise ValueError(
                f"action_horizon ({action_horizon}) is greater than the pre-chunked action dimension ({traj['action'].shape[1]})"
            )
        traj["action"] = traj["action"][:, :action_horizon]

    # then, add the history axis to actions
    traj["action"] = tf.gather(
        traj["action"], history_indices
    )  # [traj_len, window_size, action_horizon, action_dim]

    # finally, we deal with marking which actions are past the goal timestep (or final timestep if no goal)
    if "timestep" in traj["task"]:
        goal_timestep = traj["task"]["timestep"]
    else:
        goal_timestep = tf.fill([traj_len], traj_len - 1)
    # computes the number of timesteps away the goal is relative to a particular action
    t, w, h = tf.meshgrid(
        tf.range(traj_len),
        tf.range(window_size),
        tf.range(action_horizon),
        indexing="ij",
    )
    relative_goal_timestep = goal_timestep[:, None, None] - (
        t - (window_size + 1) + w + h
    )  # [traj_len, window_size, action_horizon]
    traj["observation"]["task_completed"] = relative_goal_timestep <= 0

    # broadcast "action_pad_mask" to the new chunked shape, and mark actions past the goal timestep as padding
    traj["action_pad_mask"] = tf.logical_and(
        # [traj_len, 1, 1, action_dim]
        (
            traj["action_pad_mask"][:, None, None, :]
            if len(traj["action_pad_mask"].shape) == 2
            else traj["action_pad_mask"][:, None, :]
        ),
        # [traj_len, window_size, action_horizon, 1]
        tf.logical_not(traj["observation"]["task_completed"])[:, :, :, None],
    )

    return traj


def add_phase_labels(
    traj: dict,
    pre_steps: int = 15,
    post_steps: int = 5,
    num_classes: int = 2,
    gripper_closed_value: float = 1.0,
) -> dict:
    """Heuristic manipulation phase labels from gripper open/close events in the (unchunked) actions.

    Timesteps within [event - pre_steps, event + post_steps] of a gripper event are fine manipulation, the rest
    is coarse motion (approach / transport). Adds traj["observation"]["phase"] of shape [traj_len]:
        num_classes=2: 0 coarse, 1 fine
        num_classes=3: 0 coarse, 1 grasp (around closing), 2 release (around opening)

    Assumes the last action dimension is an unnormalized binary gripper action.
    """
    assert num_classes in (2, 3), f"Unsupported number of phase classes: {num_classes}"
    closed = tf.abs(traj["action"][:, -1] - gripper_closed_value) < 0.5
    traj_len = tf.shape(closed)[0]

    # timesteps where the gripper command changes
    changed = tf.concat([[False], closed[1:] != closed[:-1]], axis=0)
    event_steps = tf.cast(tf.where(changed)[:, 0], tf.int32)  # [num_events]
    event_is_close = tf.gather(closed, event_steps)

    offsets = tf.range(traj_len)[:, None] - event_steps[None, :]  # [traj_len, num_events]
    near_event = (offsets >= -pre_steps) & (offsets <= post_steps)
    near_close = tf.reduce_any(near_event & event_is_close[None, :], axis=1)
    near_open = tf.reduce_any(near_event & ~event_is_close[None, :], axis=1)

    if num_classes == 2:
        phase = tf.cast(near_close | near_open, tf.int32)
    else:  # release wins where the windows overlap
        phase = tf.where(near_open, 2, tf.where(near_close, 1, 0))
    traj["observation"]["phase"] = phase
    return traj


def subsample(traj: dict, subsample_length: int) -> dict:
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)
    return traj


def add_pad_mask_dict(traj: dict) -> dict:
    """Adds a dictionary indicating which elements of the observation/task should be treated as padding.

    traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]
    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            if traj[key][subkey].dtype == tf.string:
                # handles "language_instruction", "image_*", and "depth_*"
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0
            else:
                # all other keys should not be treated as padding
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)
        traj[key]["pad_mask_dict"] = pad_mask_dict
    return traj


def pad_actions_and_proprio(
    traj: dict, max_action_dim: Optional[int], max_proprio_dim: Optional[int]
) -> dict:
    """Pads actions and proprio to a maximum number of dimensions across all datasets.

    Records which action dimensions are padding in "action_pad_mask".
    """
    traj["action_pad_mask"] = tf.ones_like(traj["action"], dtype=tf.bool)
    if max_action_dim is not None:
        action_dim = traj["action"].shape[-1]
        if action_dim > max_action_dim:
            raise ValueError(
                f"action_dim ({action_dim}) is greater than max_action_dim ({max_action_dim})"
            )
        for key in {"action", "action_pad_mask"}:
            traj[key] = tf.pad(
                traj[key],
                [
                    *[[0, 0]] * (len(traj[key].shape) - 1),
                    [0, max_action_dim - action_dim],
                ],
            )

    if max_proprio_dim is not None and "proprio" in traj["observation"]:
        proprio_dim = traj["observation"]["proprio"].shape[-1]
        if proprio_dim > max_proprio_dim:
            raise ValueError(
                f"proprio_dim ({proprio_dim}) is greater than max_proprio_dim ({max_proprio_dim})"
            )
        traj["observation"]["proprio"] = tf.pad(
            traj["observation"]["proprio"], [[0, 0], [0, max_proprio_dim - proprio_dim]]
        )
    return traj
