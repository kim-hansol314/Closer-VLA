"""
Locates the target (placement) location of an instruction in an image with an open-vocabulary detector.

Shared by the offline annotation (scripts/data/annotate_target_bbox.py) and evaluation, so the boxes seen in
training and at test time come from the same procedure.

"""

import json
import logging
import re
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)

# order matters: longer prepositions first
_PLACE_PREPOSITIONS = r"(?:on top of|onto|on|into|inside|in|under|to)"
_PLACE_PATTERN = re.compile(rf"\s{_PLACE_PREPOSITIONS}\s+(?:the\s+)?(.+)$")
_VERB_PATTERN = re.compile(r"^(?:open|close|turn on|turn off)\s+(?:the\s+)?(.+)$")
_PRONOUNS = {"it", "them", "there"}


def parse_target_phrase(instruction: str) -> Optional[str]:
    """
    Heuristic target location phrase from an instruction (LIBERO style), e.g.,
        "put the black bowl on the plate" -> "plate"
        "pick up the book and place it in the back compartment of the caddy" -> "back compartment of the caddy"
        "turn on the stove and put the moka pot on it" -> "stove"
        "open the top drawer and put the bowl inside" -> "top drawer"
    The last clause is used; pronouns or a missing object fall back to earlier clauses.
    Returns None if nothing is found. Review the outputs before annotation.
    """
    text = " ".join(instruction.lower().strip().rstrip(".").split())
    clauses = [clause.strip() for clause in text.split(" and ")]
    for clause in reversed(clauses):
        for pattern in (_VERB_PATTERN, _PLACE_PATTERN):
            match = pattern.search(clause)
            if match and match.group(1) not in _PRONOUNS:
                return match.group(1)
    return None


class TargetLocator:
    """Grounding DINO based detector of the target location phrase"""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        phrases_path: Optional[str] = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        device: str = "cuda",
    ):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            .to(device)
            .eval()
        )
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # reviewed instruction -> phrase mapping; null means no target
        self.phrases: Dict[str, Optional[str]] = {}
        if phrases_path is not None:
            with open(phrases_path, "r") as f:
                self.phrases = json.load(f)

    def get_phrase(self, instruction: str) -> Optional[str]:
        if instruction in self.phrases:
            return self.phrases[instruction]
        return parse_target_phrase(instruction)

    @torch.no_grad()
    def detect(self, image: np.ndarray, phrase: str) -> Optional[dict]:
        """image: [H, W, 3] uint8. Returns the highest scoring box, normalized (x0, y0, x1, y1), and its score"""
        pil_image = Image.fromarray(image)
        inputs = self.processor(
            images=pil_image, text=f"{phrase}.", return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]
        if len(result["scores"]) == 0:
            return None
        best = int(result["scores"].argmax())
        height, width = image.shape[:2]
        x0, y0, x1, y1 = result["boxes"][best].tolist()
        bbox = [
            float(np.clip(x0 / width, 0, 1)),
            float(np.clip(y0 / height, 0, 1)),
            float(np.clip(x1 / width, 0, 1)),
            float(np.clip(y1 / height, 0, 1)),
        ]
        return {"bbox": bbox, "score": float(result["scores"][best])}

    def locate(self, image: np.ndarray, instruction: str) -> Optional[dict]:
        phrase = self.get_phrase(instruction)
        if phrase is None:
            return None
        detection = self.detect(image, phrase)
        if detection is not None:
            detection["phrase"] = phrase
        return detection
