"""
Evaluation profiles matching "LoRA Learns Less and Forgets Less" paper settings.

This module defines standardized evaluation configurations for reproducible benchmarks.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


@dataclass
class EvalProfile:
    """Base evaluation profile configuration."""
    name: str
    task: str  # "gsm8k" or "humaneval"
    description: str

    # Dataset settings
    split: str = "test"
    num_fewshot: int = 0
    max_samples: int = -1  # -1 means all samples

    # Generation settings
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    n_generations: int = 1  # Number of generations per problem

    # Metric settings
    metric: str = "accuracy"  # "accuracy" or "pass@k"
    pass_k: int = 1  # For pass@k metric

    # Stop sequences (for code generation)
    stop_sequences: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Remove None values
        return {k: v for k, v in d.items() if v is not None}

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary for config JSON output."""
        return {
            "profile": self.name,
            "task": self.task,
            "split": self.split,
            "num_fewshot": self.num_fewshot,
            "n_generations": self.n_generations,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "metric": self.metric if self.metric != "pass@k" else f"pass@{self.pass_k}",
        }


# =============================================================================
# Paper Profiles: "LoRA Learns Less and Forgets Less"
# =============================================================================

PAPER_MATH_PROFILE = EvalProfile(
    name="paper_math",
    task="gsm8k",
    description="GSM8K evaluation matching 'LoRA Learns Less and Forgets Less' paper settings",
    split="test",
    num_fewshot=5,
    max_samples=-1,  # All 1319 test samples
    temperature=0.0,  # Greedy decoding
    top_p=1.0,
    max_tokens=512,
    n_generations=1,  # Single generation (greedy)
    metric="accuracy",  # Strict match accuracy (pass@1 under greedy)
)

PAPER_CODE_MAIN_PROFILE = EvalProfile(
    name="paper_code_main",
    task="humaneval",
    description="HumanEval evaluation matching 'LoRA Learns Less and Forgets Less' paper settings",
    split="test",
    num_fewshot=0,  # 0-shot
    max_samples=-1,  # All 164 problems
    temperature=0.2,  # Sampling temperature
    top_p=0.95,  # Nucleus sampling
    max_tokens=512,
    n_generations=50,  # 50 generations per problem for pass@1
    metric="pass@k",
    pass_k=1,  # pass@1 computed from 50 samples
    stop_sequences=["\ndef ", "\nclass ", "\nif __name__", "\n#", "\nprint"],
)


# =============================================================================
# Legacy/Custom Profiles
# =============================================================================

GREEDY_MATH_PROFILE = EvalProfile(
    name="greedy_math",
    task="gsm8k",
    description="GSM8K with greedy decoding (legacy default)",
    split="test",
    num_fewshot=5,
    temperature=0.0,
    top_p=1.0,
    max_tokens=512,
    n_generations=1,
    metric="accuracy",
)

GREEDY_CODE_PROFILE = EvalProfile(
    name="greedy_code",
    task="humaneval",
    description="HumanEval with greedy decoding (legacy default)",
    split="test",
    num_fewshot=0,
    temperature=0.0,
    top_p=1.0,
    max_tokens=512,
    n_generations=1,
    metric="accuracy",  # Greedy accuracy, not pass@k
    stop_sequences=["\ndef ", "\nclass ", "\nif __name__", "\n#", "\nprint"],
)


# =============================================================================
# Profile Registry
# =============================================================================

EVAL_PROFILES: Dict[str, EvalProfile] = {
    # Paper profiles (main)
    "paper_math": PAPER_MATH_PROFILE,
    "paper_code_main": PAPER_CODE_MAIN_PROFILE,

    # Legacy/custom profiles
    "greedy_math": GREEDY_MATH_PROFILE,
    "greedy_code": GREEDY_CODE_PROFILE,
}


def get_profile(name: str) -> EvalProfile:
    """Get an evaluation profile by name.

    Args:
        name: Profile name (e.g., "paper_math", "paper_code_main")

    Returns:
        EvalProfile instance

    Raises:
        ValueError: If profile name is not found
    """
    if name not in EVAL_PROFILES:
        available = ", ".join(sorted(EVAL_PROFILES.keys()))
        raise ValueError(f"Unknown eval profile: {name}. Available: {available}")
    return EVAL_PROFILES[name]


def list_profiles() -> List[str]:
    """List all available profile names."""
    return sorted(EVAL_PROFILES.keys())


def get_profile_for_task(task: str, paper_mode: bool = True) -> EvalProfile:
    """Get the appropriate profile for a task.

    Args:
        task: Task name ("gsm8k" or "humaneval")
        paper_mode: If True, use paper-matching profiles; else use greedy

    Returns:
        EvalProfile instance
    """
    if task == "gsm8k":
        return PAPER_MATH_PROFILE if paper_mode else GREEDY_MATH_PROFILE
    elif task == "humaneval":
        return PAPER_CODE_MAIN_PROFILE if paper_mode else GREEDY_CODE_PROFILE
    else:
        raise ValueError(f"Unknown task: {task}. Supported: gsm8k, humaneval")
