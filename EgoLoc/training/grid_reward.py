#!/usr/bin/env python3
"""Shared parsing and cell-based rewards for grid SFT evaluation and GRPO."""

import math
import re


GRID_CELLS = 9
DEFAULT_SIGMA = 1.0

# Match exactly one integer inside points and use the final matching JSON block.
POINTS_RE = re.compile(
    r'\{[^{}]*"points"\s*:\s*\[\s*(-?\d+)\s*\][^{}]*\}'
)


def completion_to_text(completion):
    """Normalize TRL standard or conversational completions to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content", "")
        return completion_to_text(content)
    if isinstance(completion, (list, tuple)):
        parts = []
        for item in completion:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(completion_to_text(item))
        return "".join(parts)
    return str(completion)


def parse_pred_cell(completion):
    """Parse the final ``{"points": [N]}`` block from a completion."""
    matches = POINTS_RE.findall(completion_to_text(completion))
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def build_cell_rewards(
    gt_cell, num_cells=GRID_CELLS, sigma=DEFAULT_SIGMA
):
    """Build the Gaussian reward lookup for predicted cells 1..num_cells."""
    gt_cell = int(gt_cell)
    if not 1 <= gt_cell <= num_cells:
        raise ValueError(
            f"gt_cell must be in [1, {num_cells}], got {gt_cell}"
        )
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    denominator = 2.0 * sigma * sigma
    return [
        math.exp(-((cell - gt_cell) ** 2) / denominator)
        for cell in range(1, num_cells + 1)
    ]


def gaussian_reward_for_cell(
    pred_cell,
    gt_cell,
    num_cells=GRID_CELLS,
    sigma=DEFAULT_SIGMA,
    invalid_reward=0.0,
):
    """Compute a Gaussian cell reward, returning invalid_reward when needed."""
    if pred_cell is None:
        return float(invalid_reward)
    pred_cell = int(pred_cell)
    gt_cell = int(gt_cell)
    if not 1 <= pred_cell <= num_cells:
        return float(invalid_reward)
    if not 1 <= gt_cell <= num_cells:
        raise ValueError(
            f"gt_cell must be in [1, {num_cells}], got {gt_cell}"
        )
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return math.exp(
        -((pred_cell - gt_cell) ** 2) / (2.0 * sigma * sigma)
    )


def lookup_cell_reward(pred_cell, cell_rewards, invalid_reward=0.0):
    """Look up a parsed 1-based cell in a dataset-provided reward vector."""
    if pred_cell is None:
        return float(invalid_reward)
    pred_cell = int(pred_cell)
    if not 1 <= pred_cell <= len(cell_rewards):
        return float(invalid_reward)
    return float(cell_rewards[pred_cell - 1])


def _batch_values(values, count, default, name):
    if values is None:
        return [default] * count
    if len(values) != count:
        raise ValueError(
            f"{name} has {len(values)} values for {count} completions"
        )
    return values


def gaussian_cell_reward(
    completions,
    cell_rewards,
    invalid_reward=None,
    **kwargs,
):
    """TRL reward callback using each row's stored ``cell_rewards`` column."""
    del kwargs
    count = len(completions)
    if len(cell_rewards) != count:
        raise ValueError(
            "cell_rewards must contain one lookup vector per completion"
        )
    invalid_rewards = _batch_values(
        invalid_reward, count, 0.0, "invalid_reward"
    )
    return [
        lookup_cell_reward(
            parse_pred_cell(completion),
            reward_table,
            invalid,
        )
        for completion, reward_table, invalid in zip(
            completions, cell_rewards, invalid_rewards
        )
    ]


def parse_success_metric(completions, **kwargs):
    """Zero-weight GRPO diagnostic: whether trailing JSON was parseable."""
    del kwargs
    return [
        1.0 if parse_pred_cell(completion) is not None else 0.0
        for completion in completions
    ]


def exact_cell_metric(completions, gt_cell, **kwargs):
    """Zero-weight GRPO diagnostic: exact parsed-cell accuracy."""
    del kwargs
    if len(gt_cell) != len(completions):
        raise ValueError(
            "gt_cell must contain one value per completion"
        )
    return [
        1.0 if parse_pred_cell(completion) == int(expected) else 0.0
        for completion, expected in zip(completions, gt_cell)
    ]
