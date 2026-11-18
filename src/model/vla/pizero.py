"""
Wrapper around the joint model (mixtures). Siglip from PaliGemma, action-time encoder, proprio encoder, action decoder. Flow matching training

Generates causal masking for the mixtures

Optional experts are enabled by adding their block to `cfg.mixture`:
    - depth: quantized depth token prediction (QDepth-VLA)
    - magnify: zoomed-in crop around the target (placement) location, attended by action only when gated on
    - phase: learned query tokens classifying the manipulation phase (e.g., approach vs. fine manipulation)

Token blocks in sequence order: vlm | depth | magnify | phase | proprio | action

"""

import logging
from typing import Dict, Optional, Tuple

import hydra
import torch
from torch import nn
import torch.nn.functional as F

from src.model.kv_cache import KVCache
from src.model.vla.modules import (
    ActionEncoder,
    SinusoidalPosEmb,
    get_depth_tokens,
    DepthDecoder,
    ConvDepthDecoder,
)
from src.utils.decorator import NoSyncBase
from src.utils.monitor import log_execution_time
from src.model.vla.processing import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    crop_to_bboxes,
    jitter_bboxes,
    process_images,
)

log = logging.getLogger(__name__)

OPTIONAL_EXPERTS = ("depth", "magnify", "phase")


class PiZero(nn.Module, NoSyncBase):
    @log_execution_time(log)
    def __init__(self, cfg, use_ddp: bool = False):
        super().__init__()
        self.cfg = cfg
        self.use_ddp = use_ddp  # used in NoSyncBase
        self.vocab_size = cfg.vocab_size
        self.pad_token_id = cfg.pad_token_id
        self.image_token_index = cfg.image_token_index
        self.use_lm_head = cfg.get("use_lm_head", False)

        # optional experts are enabled by their presence in the mixture config
        self.use_depth = "depth" in cfg.mixture
        self.use_magnify = "magnify" in cfg.mixture
        self.use_phase = "phase" in cfg.mixture
        log.info(
            f"Optional experts: {[name for name in OPTIONAL_EXPERTS if name in cfg.mixture]}"
        )

        self.dataset_name = cfg.get("dataset_name", "bridge")  # dataset
        self.depth_loss_mode = cfg.get("depth_loss_mode", "index_decline")
        self.use_dinov2 = self.use_depth and cfg.get("use_dinov2", False)
        if self.use_dinov2:
            self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        if self.use_depth:
            log.info(f"Using depth loss mode: {self.depth_loss_mode}")
            log.info(f"use_dinov2: {self.use_dinov2}")

        self.image_text_hidden_size = cfg.mixture.vlm.hidden_size
        self.proprio_hidden_size = cfg.mixture.proprio.hidden_size
        self.action_hidden_size = cfg.mixture.action.hidden_size
        self.num_image_tokens = cfg.vision.config.num_image_tokens

        self.max_image_text_tokens = cfg.max_image_text_tokens
        self.num_proprio_tokens = cfg.cond_steps
        self.num_action_tokens = cfg.horizon_steps

        # Depth parameterization
        if self.use_depth:
            self.max_depth_tokens = cfg.max_depth_tokens
            self.depth_hidden_size = cfg.mixture.depth.hidden_size
            self.depth_loss_weight = cfg.depth_loss_weight

        # Magnify parameterization
        if self.use_magnify:
            self.magnify_cfg = cfg.magnify
            self.magnify_hidden_size = cfg.mixture.magnify.hidden_size
            self.magnify_pool = cfg.magnify.get("pool", 1)
            self.image_grid_size = int(self.num_image_tokens**0.5)
            assert self.image_grid_size % self.magnify_pool == 0
            self.num_magnify_tokens = (
                self.image_grid_size // self.magnify_pool
            ) ** 2
            self.magnify_gate_mode = cfg.magnify.get("gate_mode", "phase")
            self.magnify_gate_phases = list(cfg.magnify.get("gate_phases", [1]))
            assert self.magnify_gate_mode in ["phase", "always"]
            assert cfg.magnify.get("inference_gate", "phase") in [
                "phase",
                "always",
                "never",
            ]
            if self.magnify_gate_mode == "phase" or (
                cfg.magnify.get("inference_gate", "phase") == "phase"
            ):
                assert self.use_phase, "Phase gating of magnify needs the phase expert"

        # Phase parameterization
        if self.use_phase:
            self.phase_hidden_size = cfg.mixture.phase.hidden_size
            self.num_phase_tokens = cfg.phase.num_tokens
            self.num_phase_classes = cfg.phase.num_classes
            self.phase_loss_weight = cfg.phase.loss_weight

        # token blocks in sequence order --- kv caches follow the same order, and action must be the last
        self.block_lens = {"vlm": self.max_image_text_tokens}
        if self.use_depth:
            self.block_lens["depth"] = self.max_depth_tokens
        if self.use_magnify:
            self.block_lens["magnify"] = self.num_magnify_tokens
        if self.use_phase:
            self.block_lens["phase"] = self.num_phase_tokens
        self.block_lens["proprio"] = self.num_proprio_tokens
        self.block_lens["action"] = self.num_action_tokens
        self.prefix_block_names = [name for name in self.block_lens if name != "action"]
        self.block_slices = {}
        start = 0
        for name, length in self.block_lens.items():
            self.block_slices[name] = slice(start, start + length)
            start += length
        self.total_num_tokens = start

        # outputs not used in training: vlm, proprio (shares weights with action), magnify
        self.train_skip_names = ("vlm", "proprio") + (
            ("magnify",) if self.use_magnify else ()
        )

        # Action parameterization
        self.num_inference_steps = cfg.num_inference_steps
        self.horizon_steps = cfg.horizon_steps
        self.action_dim = cfg.action_dim
        self.proprio_dim = cfg.proprio_dim
        self.final_action_clip_value = cfg.final_action_clip_value
        self.flow_sig_min = cfg.get("flow_sig_min", 0.001) # no "flow_sig_min" in keys use 0.001 instead

        # text input only
        self.embed_tokens = nn.Embedding(
            cfg.vocab_size,
            self.image_text_hidden_size,
            self.pad_token_id,
        )  # 0.527B parameters

        # Vision
        self.vision_tower = hydra.utils.instantiate(cfg.vision)
        self.multi_modal_projector = hydra.utils.instantiate(cfg.vision_projector)

        # Mixtures
        self.joint_model = hydra.utils.instantiate(cfg.joint)

        # Action, proprio, time encoders
        self.action_expert_adaptive_mode = cfg.action_expert_adaptive_mode
        if cfg.action_expert_adaptive_mode:  # adaLN or adaLN-Zero
            self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                time_cond=False,
            )
            self.time_embedding = SinusoidalPosEmb(
                cfg.time_hidden_size, cfg.time_max_period
            )
        else:  # matching pi0
            self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                time_cond=True,
            )
            self.time_embedding = SinusoidalPosEmb(
                self.action_hidden_size, cfg.time_max_period
            )
        self.proprio_encoder = nn.Linear(
            self.proprio_dim,
            self.proprio_hidden_size,
        )

        # Depth encoder and depth image decoder
        if self.use_depth:
            if not self.use_dinov2:
                self.vlm_to_depth = nn.Sequential(
                    nn.Linear(self.image_text_hidden_size, self.depth_hidden_size),
                )
            else:
                self.dinov2_proj = nn.Sequential(
                    nn.Linear(1024, self.depth_hidden_size),
                )
            self.depth_decoder = ConvDepthDecoder() # or DepthDecoder()

        # Magnify encoder --- crop features from the shared siglip, plus the crop location in the full image
        if self.use_magnify:
            self.magnify_proj = nn.Linear(
                self.image_text_hidden_size, self.magnify_hidden_size
            )
            self.magnify_bbox_embed = nn.Linear(4, self.magnify_hidden_size)
            nn.init.zeros_(self.magnify_bbox_embed.weight)  # start as a no-op
            nn.init.zeros_(self.magnify_bbox_embed.bias)

        # Phase queries and classifier
        if self.use_phase:
            self.phase_queries = nn.Parameter(
                torch.randn(self.num_phase_tokens, self.phase_hidden_size) * 0.02
            )
            self.phase_head = nn.Linear(self.phase_hidden_size, self.num_phase_classes)

        # Action decoder
        self.action_decoder = nn.Linear(
            self.action_hidden_size,
            self.action_dim,
        )

        # optional text output
        if self.use_lm_head:
            self.lm_head = nn.Linear(
                self.image_text_hidden_size,
                self.vocab_size,
                bias=False,
            )
            self.lm_head.weight = self.embed_tokens.weight  # tie weights

    @property
    def action_expert_parameters(self):
        """All non-VLM parameters (action/proprio and the optional experts)"""
        params = (
            list(self.action_encoder.parameters())
            + list(self.action_decoder.parameters())
            + list(self.proprio_encoder.parameters())
            + list(self.joint_model.mixtures["action"].parameters())
        )  # note: action and proprio share weights
        if self.use_depth:
            params += list(self.joint_model.mixtures["depth"].parameters())
            params += (
                list(self.vlm_to_depth.parameters())
                if not self.use_dinov2
                else list(self.dinov2_proj.parameters())
            )
            params += list(self.depth_decoder.parameters())
        if self.use_magnify:
            params += [
                param
                for name, param in self.joint_model.mixtures["magnify"].named_parameters()
                if not self._check_final_layer_post_attn_parameter_by_name(name)
            ]
            params += list(self.magnify_proj.parameters())
            params += list(self.magnify_bbox_embed.parameters())
        if self.use_phase:
            params += list(self.joint_model.mixtures["phase"].parameters())
            params += [self.phase_queries]
            params += list(self.phase_head.parameters())
        return params

    @property
    def trainable_vlm_parameters(self):
        return (
            list(self.vision_tower.parameters())
            + list(self.multi_modal_projector.parameters())
            + self.trainable_gemma_parameters
        )

    @property
    def lora_trainable_vlm_parameters(self):
        params = []
        for name, param in self.vision_tower.named_parameters():
            if "lora_" in name:
                params.append(param)
        for name, param in self.multi_modal_projector.named_parameters():
            if "lora_" in name:
                params.append(param)
        params.extend(self.trainable_lora_gemma_parameters)
        return params

    @property
    def trainable_gemma_parameters(self):
        gemma_parameters = []
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                gemma_parameters.append(param)
        return gemma_parameters

    @property
    def trainable_lora_gemma_parameters(self):
        gemma_parameters = []
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                if "lora_" in name:
                    gemma_parameters.append(param)
        return gemma_parameters

    @log_execution_time(log)
    def load_pretrained_weights(self):
        """vision, projector, lm from paligemma"""
        import glob
        import os

        from safetensors import safe_open

        # load tensors from files
        safetensors_files = glob.glob(
            os.path.join(self.cfg.pretrained_model_path, "*.safetensors")
        )
        tensors = {}
        for safetensors_file in safetensors_files:
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

        # load embed tokens
        embed_tokens_state_dict = self.embed_tokens.state_dict()
        for k, v in tensors.items():
            if "embed_tokens" in k:
                new_key = k.replace("language_model.model.embed_tokens.", "")
                embed_tokens_state_dict[new_key] = v
        self.embed_tokens.load_state_dict(embed_tokens_state_dict, strict=True)
        log.info("Loaded pre-trained weights for embed tokens")

        # load vision tower --- "vision_tower.vision_model" -> "vision_model"
        vision_tower_state_dict = self.vision_tower.state_dict()
        for k, v in tensors.items():
            if "vision_tower" in k:
                new_key = k.replace("vision_tower.", "")
                vision_tower_state_dict[new_key] = v
        self.vision_tower.load_state_dict(vision_tower_state_dict, strict=True)
        log.info("Loaded pre-trained weights for vision tower")

        # load projector --- "multi_modal_projector.linear" -> "linear"
        multi_modal_projector_state_dict = self.multi_modal_projector.state_dict()
        for k, v in tensors.items():
            if "multi_modal_projector" in k:
                new_key = k.replace("multi_modal_projector.", "")
                multi_modal_projector_state_dict[new_key] = v
        self.multi_modal_projector.load_state_dict(
            multi_modal_projector_state_dict, strict=True
        )
        log.info("Loaded pre-trained weights for projector")

        # load lm --- do not change any lora weights
        joint_model_state_dict = self.joint_model.state_dict()
        lora_keys = []
        for key in (
            joint_model_state_dict.keys()
        ):  # avoid RuntimeError: OrderedDict mutated during iteration
            if "lora_" in key:
                lora_keys.append(key)
        for key in lora_keys:
            del joint_model_state_dict[key]
        for k, v in tensors.items():
            if "language_model.model" in k:
                new_key = k.replace("language_model.model.", "mixtures.vlm.")
                joint_model_state_dict[new_key] = v
        self.joint_model.load_state_dict(joint_model_state_dict, strict=False)
        log.info("Loaded pre-trained weights for lm part of the joint model")

    def _check_gemma_unused_parameter_by_name(self, name: str) -> bool:
        """no need to train vlm parameters after attention of last layer"""
        last_hidden_layer_index = self.joint_model.num_hidden_layers - 1
        if (
            f"{last_hidden_layer_index}.post" in name
            or f"{last_hidden_layer_index}.mlp" in name
            or f"{last_hidden_layer_index}.self_attn.o_proj" in name
            or f"{last_hidden_layer_index}.self_attn.v_proj" in name
        ):  # final norm is not initialized
            return True
        return False

    def _check_final_layer_post_attn_parameter_by_name(self, name: str) -> bool:
        """for mixtures whose output is never read (magnify), the last layer after attention gets no gradient"""
        last_hidden_layer_index = self.joint_model.num_hidden_layers - 1
        return (
            f"layers.{last_hidden_layer_index}.post" in name
            or f"layers.{last_hidden_layer_index}.final" in name
            or f"layers.{last_hidden_layer_index}.mlp" in name
            or f"layers.{last_hidden_layer_index}.self_attn.o_proj" in name
        )

    def freeze_non_lora_weights_in_vlm(self):
        """Keep all bias frozen"""
        for name, param in self.vision_tower.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in vision tower")

        for name, param in self.multi_modal_projector.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in projector")

        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in lm part of the joint model")

    def freeze_unused_weights(self):
        """text embedding and part of last layer of vlm, including lora"""
        self.embed_tokens.weight.requires_grad = False
        """ freeeze all weights in vision encoder
        for param in self.vision_tower.parameters():
            param.requires_grad = False"""
        if self.use_dinov2:
            for p in self.dinov2.parameters():
                p.requires_grad = False
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if self._check_gemma_unused_parameter_by_name(name):
                param.requires_grad = False
        if self.use_magnify:
            for name, param in self.joint_model.mixtures["magnify"].named_parameters():
                if self._check_final_layer_post_attn_parameter_by_name(name):
                    param.requires_grad = False

    def freeze_all_weights(self):
        for _, param in self.named_parameters():
            param.requires_grad = False

    def tie_action_proprio_weights(self):
        """technically more than just tying weights"""
        self.joint_model.mixtures["proprio"] = self.joint_model.mixtures["action"]

    def build_text_cache(self):
        return KVCache()

    # ---------- Input preparation ----------#

    def build_causal_mask_and_position_ids(
        self,
        attention_mask: torch.Tensor,
        dtype: torch.dtype,
        magnify_gate: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.LongTensor]]:
        """
        block attention --- padding for unused text tokens before

        Q (from) / KV (to)  image/text  depth  magnify  phase  proprio  action
        image/text              ✅        ❌      ❌       ❌      ❌       ❌
        depth                   ✅        ✅      ❌       ❌      ❌       ❌
        magnify                 ✅        ❌      ✅       ❌      ❌       ❌
        phase                   ✅        ❌      ❌       ✅      ✅       ❌
        proprio                 ✅        ❌      ❌       ❌      ✅       ❌
        action                  ✅        ✅      gate     ✅      ✅       ✅

        phase does not see magnify, so the phase prediction can gate magnify for action at inference.
        magnify_gate: [Batch_Size] bool, whether action attends to magnify; None means always.

        Returns the mask [Batch_Size, 1, Total_Len, Total_Len] and the position ids of each block,
        keyed by block name.
        """
        bsz = attention_mask.size(0)
        s = self.block_slices
        allow = torch.zeros(
            (bsz, self.total_num_tokens, self.total_num_tokens), dtype=torch.bool
        )

        def allow_block(q_name, kv_names):
            for kv_name in kv_names:
                allow[:, s[q_name], s[kv_name]] = True

        allow_block("vlm", ["vlm"])
        if self.use_depth:
            allow_block("depth", ["vlm", "depth"])
        if self.use_magnify:
            allow_block("magnify", ["vlm", "magnify"])
        if self.use_phase:
            allow_block("phase", ["vlm", "phase", "proprio"])
        allow_block("proprio", ["vlm", "proprio"])
        allow[:, s["action"], :] = True  # action attends to everything
        if self.use_magnify and magnify_gate is not None:
            allow[:, s["action"], s["magnify"]] &= magnify_gate.bool().cpu()[:, None, None]

        # no one attends to padded text tokens; padded text tokens attend to nothing (uniform)
        text_valid = attention_mask.bool().cpu()
        allow[:, :, s["vlm"]] &= text_valid[:, None, :]
        allow[:, s["vlm"], :] &= text_valid[:, :, None]

        causal_mask = torch.zeros(
            (bsz, self.total_num_tokens, self.total_num_tokens), dtype=dtype
        ).masked_fill(
            ~allow, torch.finfo(dtype).min
        )  # smallest value, avoid using inf for softmax nan issues with padding

        # add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        # position ids for each blocks --- start at 1
        # since proprio and action share the same mixture weights, makes sense to use [1 (proprio), 2 (action), 3 (action), ...] instead of [1 (proprio), 1 (action), 2 (action), ...]
        position_ids = {}
        for name, length in self.block_lens.items():
            start = self.num_proprio_tokens + 1 if name == "action" else 1
            position_ids[name] = torch.arange(start, start + length).repeat(bsz, 1)
        return causal_mask, position_ids

    def split_full_mask_into_submasks(
        self, causal_mask: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """split into ones for the prefix (all blocks but action) and action"""
        prefix_len = self.block_slices["action"].start
        image_text_proprio_mask = causal_mask[..., :prefix_len, :prefix_len]
        action_mask = causal_mask[..., self.block_slices["action"], :]
        return image_text_proprio_mask, action_mask

    def sample_magnify_gate(
        self,
        phase_labels: Optional[torch.LongTensor],
        magnify_valid: torch.BoolTensor,
    ) -> torch.BoolTensor:
        """Training gate: ground-truth phase, randomly flipped so action is robust to phase errors at inference"""
        if self.magnify_gate_mode == "always":
            return magnify_valid.bool()
        gate = torch.isin(
            phase_labels, torch.tensor(self.magnify_gate_phases, device=phase_labels.device)
        )
        drop = torch.rand(gate.shape) < self.magnify_cfg.get("gate_drop_prob", 0.0)
        force = torch.rand(gate.shape) < self.magnify_cfg.get("gate_force_prob", 0.0)
        gate = (gate & ~drop) | (~gate & force)
        return gate & magnify_valid.bool()

    def prepare_magnify_inputs(
        self,
        images: torch.ByteTensor,
        target_bboxes: torch.FloatTensor,
        magnify_valid: torch.BoolTensor,
        jitter: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        images: [Batch_Size, 3, H, W] uint8, the same images fed to siglip
        target_bboxes: [Batch_Size, 4] normalized (x0, y0, x1, y1); ignored where magnify_valid is False
        """
        target_bboxes = torch.where(
            magnify_valid.bool()[:, None],
            target_bboxes.float(),
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
        )
        if jitter:
            target_bboxes = jitter_bboxes(
                target_bboxes, self.magnify_cfg.get("bbox_jitter", 0.0)
            )
        crops, crop_bboxes = crop_to_bboxes(
            images,
            target_bboxes,
            out_size=self.cfg.vision.config.image_size,
            margin=self.magnify_cfg.get("bbox_margin", 0.25),
            min_size=self.magnify_cfg.get("min_crop_size", 0.2),
        )
        magnify_pixel_values = process_images(
            crops,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )
        return {
            "magnify_pixel_values": magnify_pixel_values,
            "target_bboxes": crop_bboxes,
            "magnify_valid": magnify_valid.bool(),
        }

    # ---------- Embeddings ----------#

    def _forward_image_features(
        self, pixel_values: torch.FloatTensor
    ) -> torch.FloatTensor:
        # image features from siglip and projector
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        selected_image_feature = self.vision_tower(pixel_values)
        image_features = self.multi_modal_projector(selected_image_feature)

        # normalize the image features
        return image_features / (self.image_text_hidden_size**0.5)

    def _merge_text_and_image_embedding(
        self,
        input_ids: torch.LongTensor,
        scaled_image_features: torch.FloatTensor,
    ) -> torch.FloatTensor:
        dtype, device = scaled_image_features.dtype, scaled_image_features.device

        # text embedding
        # [Batch_Size, Seq_Len, Hidden_Size]
        inputs_embeds = self.embed_tokens(input_ids)

        # put embedding together - image, text, padding
        _, _, embed_dim = scaled_image_features.shape
        bsz, seq_len = input_ids.shape
        final_embedding = torch.full(
            (bsz, seq_len, embed_dim), self.pad_token_id, dtype=dtype, device=device
        )

        # [Batch_Size, Seq_Len]
        text_mask = (input_ids != self.image_token_index) & (
            input_ids != self.pad_token_id
        )
        image_mask = input_ids == self.image_token_index
        final_embedding[text_mask] = inputs_embeds[text_mask]
        for i in range(bsz):
            image_indices = image_mask[i].nonzero(as_tuple=True)[0]
            num_image_tokens = len(image_indices)
            final_embedding[i, image_indices] = scaled_image_features[
                i, :num_image_tokens
            ]
        return final_embedding

    def _forward_siglip_and_text_embedding(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return self._merge_text_and_image_embedding(
            input_ids, self._forward_image_features(pixel_values)
        )

    def _forward_dinov2_embedding(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """
        encode images into [B, 256, 1024] feature sequence with DINOv2
        """
        # last layer patch tokens from the DINOv2 backbone
        features = self.dinov2.get_intermediate_layers(
            pixel_values,
        )
        patch_tokens = features[0]  # [B, 256, 1024]
        patch_tokens = self.dinov2_proj(patch_tokens)  # project to depth_hidden_size

        return patch_tokens

    def _embed_magnify(
        self,
        scaled_crop_features: torch.FloatTensor,
        target_bboxes: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # [Batch_Size, 256, 2048] -> [Batch_Size, 256 / pool^2, Hidden_Size]
        x = scaled_crop_features
        if self.magnify_pool > 1:
            bsz, _, dim = x.shape
            x = x.transpose(1, 2).reshape(
                bsz, dim, self.image_grid_size, self.image_grid_size
            )
            x = F.avg_pool2d(x, self.magnify_pool)
            x = x.flatten(2).transpose(1, 2)
        x = self.magnify_proj(x)
        return x + self.magnify_bbox_embed(target_bboxes.to(x.dtype))[:, None]

    def _embed_prefix(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        proprios: torch.FloatTensor,
        dino_pixel_values: Optional[torch.FloatTensor] = None,
        magnify_pixel_values: Optional[torch.FloatTensor] = None,
        target_bboxes: Optional[torch.FloatTensor] = None,
    ) -> Dict[str, torch.FloatTensor]:
        """embeddings of all blocks but action, in sequence order"""
        bsz = input_ids.size(0)

        # run the full image and the magnify crop through siglip together
        if self.use_magnify:
            image_features, crop_features = self._forward_image_features(
                torch.cat([pixel_values, magnify_pixel_values])
            ).split(bsz)
        else:
            image_features = self._forward_image_features(pixel_values)

        embeds = {
            "vlm": self._merge_text_and_image_embedding(input_ids, image_features)
        }
        if self.use_depth:
            if self.use_dinov2:
                embeds["depth"] = self._forward_dinov2_embedding(dino_pixel_values)
            else:
                embeds["depth"] = self.vlm_to_depth(image_features)  # [bsz, 256, 1024]
        if self.use_magnify:
            embeds["magnify"] = self._embed_magnify(crop_features, target_bboxes)
        if self.use_phase:
            # repeat instead of expand since the joint model scales embeddings in place
            embeds["phase"] = self.phase_queries[None].repeat(bsz, 1, 1)
        embeds["proprio"] = self.proprio_encoder(proprios)
        return {name: embeds[name] for name in self.prefix_block_names}

    def _gather_position_ids(self, **position_ids) -> Dict[str, torch.LongTensor]:
        return {
            name: position_ids[f"{name}_position_ids"] for name in self.block_lens
        }

    def _inference_magnify_gate(
        self,
        phase_logits: Optional[torch.FloatTensor],
        magnify_valid: Optional[torch.BoolTensor],
        bsz: int,
        device: torch.device,
    ) -> torch.BoolTensor:
        mode = self.magnify_cfg.get("inference_gate", "phase")
        if mode == "always":
            gate = torch.ones(bsz, dtype=torch.bool, device=device)
        elif mode == "never":
            gate = torch.zeros(bsz, dtype=torch.bool, device=device)
        else:
            gate = torch.isin(
                phase_logits.argmax(dim=-1),
                torch.tensor(self.magnify_gate_phases, device=device),
            )
        if magnify_valid is not None:
            gate = gate & magnify_valid.bool()
        return gate

    # ---------- Inference ----------#

    def infer_action(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_text_proprio_mask: torch.FloatTensor,
        action_mask: torch.FloatTensor,
        proprios: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        depth_position_ids: Optional[torch.LongTensor] = None,
        magnify_position_ids: Optional[torch.LongTensor] = None,
        phase_position_ids: Optional[torch.LongTensor] = None,
        dino_pixel_values: Optional[torch.FloatTensor] = None,
        magnify_pixel_values: Optional[torch.FloatTensor] = None,
        target_bboxes: Optional[torch.FloatTensor] = None,
        magnify_valid: Optional[torch.BoolTensor] = None,
        return_phase: bool = False,
        **unused_training_inputs,  # labels, depth images, cnt_update
    ):
        """
        Returns actions [Batch_Size, Horizon_Steps, Action_Dim], plus (phase_logits, magnify_gate) if return_phase
        """
        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = pixel_values.size(0)
        position_ids_all = self._gather_position_ids(
            vlm_position_ids=vlm_position_ids,
            depth_position_ids=depth_position_ids,
            magnify_position_ids=magnify_position_ids,
            phase_position_ids=phase_position_ids,
            proprio_position_ids=proprio_position_ids,
            action_position_ids=action_position_ids,
        )

        # caches in sequence order, which is also the kv order seen by action
        kv_caches = {name: KVCache() for name in self.prefix_block_names}

        prefix_embeds = self._embed_prefix(
            input_ids,
            pixel_values,
            proprios,
            dino_pixel_values=dino_pixel_values,
            magnify_pixel_values=magnify_pixel_values,
            target_bboxes=target_bboxes,
        )

        # forward pass thru the prefix blocks, cache the kv; only phase output is needed
        prefix_outputs, kv_caches = self.joint_model(
            attention_mask=image_text_proprio_mask,
            position_ids_all={
                name: position_ids_all[name] for name in self.prefix_block_names
            },
            embeds_all=prefix_embeds,
            final_layer_post_attn_skip_names=tuple(
                name for name in self.prefix_block_names if name != "phase"
            ),
            kv_caches=kv_caches,
            return_caches=True,
        )

        phase_logits = None
        if self.use_phase:
            phase_logits = self.phase_head(prefix_outputs["phase"].mean(dim=1))

        # gate magnify for action based on the predicted phase
        magnify_gate = None
        if self.use_magnify:
            magnify_gate = self._inference_magnify_gate(
                phase_logits, magnify_valid, bsz, device
            )
            magnify_cols = self.block_slices["magnify"]
            action_mask = action_mask.clone()
            action_mask[..., magnify_cols] = torch.where(
                magnify_gate[:, None, None, None],
                action_mask[..., magnify_cols],
                torch.finfo(action_mask.dtype).min,
            )

        # sample pure action noise
        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )

        # forward euler integration --- using kv caches of the prefix blocks
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for _ in range(self.num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            if self.action_expert_adaptive_mode:
                action_embeds = self.action_encoder(action)
            else:
                action_embeds = self.action_encoder(action, time_cond)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            action_embeds = self.joint_model(
                attention_mask=action_mask,
                position_ids_all={"action": position_ids_all["action"]},
                embeds_all={"action": action_embeds},
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="append_non_active",  # use caches from the prefix blocks
            )["action"]
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            action_vel = self.action_decoder(action_embeds)
            action += delta_t * action_vel
            t += delta_t

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        if return_phase:
            return action, phase_logits, magnify_gate
        return action

    # ---------- Flow matching training ----------#

    def psi_t(
        self,
        x: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Conditional Flow"""
        t = t[:, None, None]  # (B, 1, 1)
        return (1 - (1 - self.flow_sig_min) * t) * x + t * x1

    def _depth_loss_scale(self, cnt_update: torch.FloatTensor):
        if self.depth_loss_mode == "index_decline":
            # (0.9998**(cnt_update)) for bridge & fractal ; (0.999**(cnt_update-33000)) for libero / real robot finetuned from fractal
            if "real" not in self.dataset_name and self.dataset_name != "libero":
                return 0.9998**cnt_update
            return 0.999 ** (cnt_update - 33000)
        elif self.depth_loss_mode == "constant":
            return 1.0
        elif self.depth_loss_mode == "no_depth":
            return 0.0
        raise NotImplementedError(f"Unknown depth loss mode: {self.depth_loss_mode}. ")

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        causal_mask: torch.FloatTensor,
        proprios: torch.FloatTensor,
        actions: torch.FloatTensor,
        t: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        depth_position_ids: Optional[torch.LongTensor] = None,
        magnify_position_ids: Optional[torch.LongTensor] = None,
        phase_position_ids: Optional[torch.LongTensor] = None,
        dino_pixel_values: Optional[torch.FloatTensor] = None,
        depth_images: Optional[torch.FloatTensor] = None,  # depth image (ground truth)
        cnt_update: Optional[torch.FloatTensor] = None,
        magnify_pixel_values: Optional[torch.FloatTensor] = None,
        target_bboxes: Optional[torch.FloatTensor] = None,
        phase_labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """
        flow matching loss for action prediction plus auxiliary expert losses, no use of kv cache
        magnify gating is already applied in causal_mask

        Returns total loss and a dict of detached metrics for logging
        """
        # noisy action
        # [Batch_Size, Horizon_Steps, Action_Dim]
        x0 = torch.randn_like(actions, device=t.device, dtype=t.dtype)
        x1 = actions
        psi_t = self.psi_t(x0, x1, t)

        # image/text, depth, magnify, phase, proprio
        embeds_all = self._embed_prefix(
            input_ids,
            pixel_values,
            proprios,
            dino_pixel_values=dino_pixel_values,
            magnify_pixel_values=magnify_pixel_values,
            target_bboxes=target_bboxes,
        )

        # inference with noisy action
        # [Batch_Size, Embed_Dim]
        time_cond = self.time_embedding(t)
        # [Batch_Size, Horizon_Steps, Embed_Dim]
        if self.action_expert_adaptive_mode:
            embeds_all["action"] = self.action_encoder(psi_t)
        else:
            embeds_all["action"] = self.action_encoder(psi_t, time_cond)

        outputs = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all=self._gather_position_ids(
                vlm_position_ids=vlm_position_ids,
                depth_position_ids=depth_position_ids,
                magnify_position_ids=magnify_position_ids,
                phase_position_ids=phase_position_ids,
                proprio_position_ids=proprio_position_ids,
                action_position_ids=action_position_ids,
            ),
            embeds_all=embeds_all,
            time_cond=time_cond,
            final_layer_post_attn_skip_names=self.train_skip_names,
            kv_caches={},  # no caching during training
        )

        # [Batch_Size, Horizon_Steps, Action_Dim]
        v_psi = self.action_decoder(outputs["action"])

        # compare to true velocity
        d_psi = x1 - (1 - self.flow_sig_min) * x0
        action_loss = torch.mean((v_psi - d_psi) ** 2)
        loss = action_loss
        metrics = {"action_loss": action_loss.detach()}

        if self.use_depth:
            # 1. predict quantized depth tokens [bsz, 256, 160]
            decoded_tokens = self.depth_decoder(outputs["depth"])

            # 2. cross-entropy token matching loss
            # depth_images : shape: (torch.Size([bsz, 1, H, W, 3], uint8)
            depth_token_gt, logits = get_depth_tokens(
                depth_images.squeeze(1).permute(0, 3, 1, 2),
                decoded_tokens,
                checkpoint_path=f"checkpoints/vq_vae_{self.dataset_name}.ckpt",
            )  # [B, 256]
            depth_loss = F.cross_entropy(logits, depth_token_gt.view(-1))
            loss = loss + depth_loss * self.depth_loss_weight * self._depth_loss_scale(
                cnt_update
            )
            metrics["depth_loss"] = depth_loss.detach()

        if self.use_phase:
            phase_logits = self.phase_head(outputs["phase"].mean(dim=1)).float()
            phase_loss = F.cross_entropy(phase_logits, phase_labels)
            loss = loss + phase_loss * self.phase_loss_weight
            metrics["phase_loss"] = phase_loss.detach()
            metrics["phase_acc"] = (
                (phase_logits.argmax(dim=-1) == phase_labels).float().mean().detach()
            )

        if self.use_magnify:
            magnify_gate = (
                causal_mask[:, 0, self.block_slices["action"].start, self.block_slices["magnify"].start]
                == 0
            )
            metrics["magnify_gate_ratio"] = magnify_gate.float().mean().detach()

        return loss, metrics


class PiZeroInference(PiZero):
    def forward(self, **kwargs) -> torch.FloatTensor:
        return super().infer_action(**kwargs)
