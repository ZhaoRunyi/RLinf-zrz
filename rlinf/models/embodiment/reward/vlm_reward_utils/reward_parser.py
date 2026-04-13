import re
import json
import torch
from dataclasses import dataclass
from typing import Protocol, Dict, Any


REWARD_PARSER_REGISTRY: Dict[str, type] = {}

def register_reward_parser(name: str):
    def decorator(cls: type):
        REWARD_PARSER_REGISTRY[name.lower()] = cls
        return cls 
    return decorator

def get_reward_parser(name: str) -> type:
    name_lower = name.lower()
    if name_lower not in REWARD_PARSER_REGISTRY:
        raise ValueError(f"RewardParser '{name}' not registered")
    return REWARD_PARSER_REGISTRY[name_lower]


@register_reward_parser("base_reward_parser")
class BaseRewardParser(Protocol):
    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:  # pragma: no cover - tiny wrapper
        pass


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        json_text_chunk = text[start : end + 1]
        try:
            obj = json.loads(json_text_chunk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None

def _parse_robochallenge_output(text: str, use_confidence: bool) -> float | None:
    obj = _extract_json_object(text)
    if obj is not None:
        judgement = obj.get("judgement", None)
        if judgement is None:
            for key in ("answer", "label"):
                judgement = obj.get(key, None)
                if judgement is not None:
                    break

        if judgement is not None:
            j = str(judgement).strip().lower()
            if j in ("positive", "negative"):
                if not use_confidence:
                    return 1.0 if j == "positive" else -1.0

                confidence = obj.get("confidence", None)
                if confidence is None:
                    return None
                try:
                    conf = float(confidence)
                except Exception:
                    return None
                return conf if j == "positive" else 1.0 - conf

    if use_confidence:
        return None

    matches = re.findall(r"\b(positive|negative)\b", str(text).strip().lower())
    if not matches:
        return None
    judgement = matches[-1]
    if judgement == "positive":
        return 1.0
    if judgement == "negative":
        return 0.0
    return None

@register_reward_parser("robochallenge_reward_parser")
class RoboChallengeRewardParser(BaseRewardParser):
    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:
        rewards: list[float] = []
        for output in outputs:
            reward = _parse_robochallenge_output(output, use_confidence=True)
            rewards.append(0.0 if reward is None else float(reward))
        rewards = torch.tensor(rewards, dtype=torch.float32).clamp(0.0, 1.0)
        return rewards


@register_reward_parser("simple_robochallenge_reward_parser")
class SimpleRobochallengeRewardParser(BaseRewardParser):
    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:
        rewards: list[float] = []
        for output in outputs:
            reward = _parse_robochallenge_output(output, use_confidence=False)
            rewards.append(0.0 if reward is None else float(reward))
        rewards = torch.tensor(rewards, dtype=torch.float32).clamp(0.0, 1.0)
        return rewards
