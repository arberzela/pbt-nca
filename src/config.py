import json
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Literal

import torch


@dataclass
class Config:
    """Configuration class for adversarial NCA training experiments.

    This dataclass manages all hyperparameters and system settings for training
    competing Neural Cellular Automata. It includes automatic device detection,
    validation, and seed management for reproducible experiments.
    """

    # Grid
    grid_size: tuple[int, int] = (10, 10)
    n_seeds: int = 1

    # World
    cell_state_dim: int = 4
    cell_hidden_dim: int = 4

    # New world paradigm
    seed_dist: Literal["scatter"] = "scatter"
    # Whether it's set to 1 in everything or random noise
    seed_mode: Literal["solid", "random"] = "random"
    alive_visible: bool = True  # Whether NCAs can see where things are alive

    # Burn-in config
    # NOTE: For burn-in, we're currently updating both the steps_per_update and steps_before_update
    burn_in: bool = False
    burn_in_increment_epochs: int = (
        0  # How many steps before you try to increase the steps per
    )
    burn_in_increment: int = 0  # How many steps to increase by each time, it seems that

    # NCAs
    n_ncas: int = 3
    n_hidden_layers: int = 0
    hidden_dim: int = 32
    model_kernel_size: int = 3
    model_dropout_per: float = 0.0
    per_hid_upd: float = 1.0  # Percentage of hidden channels each model can update

    # Training
    softmax_temp: float = 1.0
    optimizer: Literal["AdamW", "Adam", "RMSProp", "SGD"] = "RMSProp"
    learning_rate: float = 3e-4
    batch_size: int = 32
    pool_size: int = 1024
    epochs: int = 1_000
    log_every: int = 100
    wandb: bool = False

    # Sun
    sun_update_epoch_wait: int = 0

    # Multi-world
    steps_before_update: int = 0
    steps_per_update: int = 1

    # General system
    device: Literal["cpu", "cuda", "mps"] = "cuda"
    seed: int = 42
    mode: Literal["train", "eval", "frozen_eval"] = "train"

    # Population-based training (non-stationary objective)
    pbt_enabled: bool = False
    pbt_population_size: int = 8
    pbt_meta_iterations: int = 250
    pbt_world_horizon: int = 4
    pbt_exploit_interval: int = 1
    pbt_replace_fraction: float = 0.25
    pbt_archive_size: int = 512
    pbt_archive_top_m: int = 2
    pbt_novelty_k: int = 8
    pbt_weight_noise_std: float = 0.0

    pbt_viz_interval: int = 5
    pbt_viz_max_frames: int = 24
    pbt_viz_fps: int = 30
    pbt_viz_rollout_multiplier: int = 5

    # PBT VLM descriptors (optional)
    pbt_vlm_enabled: bool = False
    pbt_vlm_model_id: str = "openai/clip-vit-base-patch32"
    pbt_vlm_frames: int = 8
    pbt_vlm_device: Literal["cpu", "cuda", "mps"] = "cuda"
    pbt_vlm_metric: Literal["cosine", "l2"] = "cosine"
    pbt_vlm_prompts: str = ""
    pbt_vlm_prompt_weight: float = 0.0
    pbt_vlm_only: bool = False
    pbt_use_median_ranking: bool = False
    pbt_use_elo_ranking: bool = False # just DINO
    pbt_use_combined_median: bool = False # combining DINO + novelty 
    pbt_elo_k: float = 32.0

    # PBT evolutionary inheritance mode
    pbt_inheritance_mode: Literal["darwin", "lamarck"] = "darwin"

    # PBT mutation controls
    pbt_archive_reset_interval: int = 0

    def __post_init__(self) -> None:
        """Validate configuration and initialize system settings.

        Performs validation checks on configuration parameters, handles device
        availability fallbacks, and sets random seeds for reproducibility.

        Raises:
            AssertionError: If cell_state_dim is not even or batch_size > pool_size.
        """
        assert self.cell_state_dim % 2 == 0, "[config] cell_state_dim must be even"
        assert self.batch_size <= self.pool_size, "[config] batch_size > pool_size"
        assert self.n_seeds * self.n_ncas <= self.total_grid_size, (
            "[config] n_seeds * n_ncas > self.total_grid_size"
        )
        assert self.softmax_temp > 0, "[config] softmax_temp <= 0"
        assert self.learning_rate > 0, "[config] learning_rate <= 0"

        if self.pbt_enabled:
            assert self.pbt_population_size >= 2, "[config] pbt_population_size < 2"
            assert self.pbt_meta_iterations > 0, "[config] pbt_meta_iterations <= 0"
            assert self.pbt_world_horizon > 0, "[config] pbt_world_horizon <= 0"
            assert self.pbt_exploit_interval > 0, "[config] pbt_exploit_interval <= 0"
            assert 0 < self.pbt_replace_fraction < 1, (
                "[config] pbt_replace_fraction must be in (0, 1)"
            )
            assert self.pbt_archive_size > 0, "[config] pbt_archive_size <= 0"
            assert self.pbt_archive_top_m > 0, "[config] pbt_archive_top_m <= 0"
            assert self.pbt_novelty_k > 0, "[config] pbt_novelty_k <= 0"
            assert self.pbt_viz_interval >= 0, "[config] pbt_viz_interval < 0"
            assert self.pbt_viz_fps > 0, "[config] pbt_viz_fps <= 0"
            assert self.pbt_viz_rollout_multiplier >= 1, (
                "[config] pbt_viz_rollout_multiplier < 1"
            )
            assert self.pbt_inheritance_mode in (
                "darwin",
                "lamarck",
            ), "[config] pbt_inheritance_mode invalid"
            if self.pbt_vlm_enabled:
                assert self.pbt_vlm_frames > 0, "[config] pbt_vlm_frames <= 0"
                assert self.pbt_vlm_metric in (
                    "cosine",
                    "l2",
                ), "[config] pbt_vlm_metric invalid"
                assert self.pbt_vlm_prompt_weight >= 0, (
                    "[config] pbt_vlm_prompt_weight < 0"
                )
                if (
                    self.pbt_vlm_only
                    and not (self.pbt_use_median_ranking or self.pbt_use_elo_ranking)
                ):
                    assert self.pbt_vlm_prompt_weight > 0, (
                        "[config] pbt_vlm_only requires pbt_vlm_prompt_weight > 0"
                    )
            elif self.pbt_vlm_only and not (
                self.pbt_use_median_ranking
                or self.pbt_use_elo_ranking
            ):
                raise AssertionError(
                    "[config] pbt_vlm_only requires pbt_vlm_enabled = True"
                )
            ranking_modes = sum([
                self.pbt_use_median_ranking,
                self.pbt_use_elo_ranking,
                self.pbt_use_combined_median,
            ])
            if ranking_modes > 1:
                raise AssertionError(
                    "[config] pbt_use_median_ranking, pbt_use_elo_ranking, "
                    "and pbt_use_combined_median are mutually exclusive"
                )
            if self.pbt_use_median_ranking:
                assert self.pbt_vlm_only, (
                    "[config] pbt_use_median_ranking requires pbt_vlm_only = True"
                )
            if self.pbt_use_elo_ranking:
                assert self.pbt_vlm_only, (
                    "[config] pbt_use_elo_ranking requires pbt_vlm_only = True"
                )
                assert self.pbt_elo_k > 0, "[config] pbt_elo_k must be > 0"
        # Device availability check
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[warning] CUDA not available, falling back to CPU")
            object.__setattr__(self, "device", "cpu")
        elif self.device == "mps" and not torch.backends.mps.is_available():
            print("[warning] MPS not available, falling back to CPU")
            object.__setattr__(self, "device", "cpu")

        if self.pbt_vlm_enabled or self.pbt_use_median_ranking:
            if self.pbt_vlm_device == "cuda" and not torch.cuda.is_available():
                print("[warning] PBT VLM CUDA not available, falling back to CPU")
                object.__setattr__(self, "pbt_vlm_device", "cpu")
            elif (
                self.pbt_vlm_device == "mps"
                and not torch.backends.mps.is_available()
            ):
                print("[warning] PBT VLM MPS not available, falling back to CPU")
                object.__setattr__(self, "pbt_vlm_device", "cpu")

        self._set_random_seed()

    def _set_random_seed(self) -> None:
        """Set all random seeds for reproducibility"""
        import random

        import numpy as np
        import torch

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if torch.backends.mps.is_available():
            torch.mps.manual_seed(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def validate_search_space(self) -> None:
        """Validate only the PBT-mutable hyperparameters."""
        assert self.batch_size <= self.pool_size, "[config] batch_size > pool_size"
        assert self.learning_rate > 0, "[config] learning_rate <= 0"
        assert self.steps_per_update >= 1, "[config] steps_per_update < 1"
        assert self.steps_before_update >= 0, "[config] steps_before_update < 0"

    @contextmanager
    def seed_context(self):
        """Temporarily set global seeds to this config's seed, restoring after."""
        import random as _random

        import numpy as _np

        py_state = _random.getstate()
        np_state = _np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = (
            [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        )

        self._set_random_seed()
        try:
            yield
        finally:
            _random.setstate(py_state)
            _np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)
            for i, s in enumerate(cuda_states):
                torch.cuda.set_rng_state(s, i)

    @property
    def cell_dim(self) -> int:
        """Total cell dimension including state, hidden, aliveness, and NCA channels.

        Returns:
            Combined dimension of cell_state_dim + cell_hidden_dim + n_ncas + 1.
        """
        return self.cell_state_dim + self.cell_hidden_dim + self.n_ncas + 1

    @property
    def cell_wo_alive_dim(self) -> int:
        """Cell dimension w/o NCA channels (just state and hidden)

        Returns:
            Combined dimension of cell_state_dim + cell_hidden_dim
        """
        return self.cell_state_dim + self.cell_hidden_dim

    @property
    def alive_dim(self) -> int:
        """Dimension for aliveness channels.

        Returns:
            Number of aliveness channels (n_ncas + 1 for sun).
        """
        return self.n_ncas + 1

    @property
    def total_grid_size(self) -> int:
        """Total number of cells in the grid.

        Returns:
            Product of grid dimensions (width * height).
        """
        return self.grid_size[0] * self.grid_size[1]

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load configuration from JSON file.

        Args:
            path: Path to JSON configuration file.

        Returns:
            Config instance with parameters loaded from file.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        with open(path) as f:
            raw_config = json.load(f)

        valid_fields = {field.name for field in fields(cls)}
        unknown_keys = sorted(set(raw_config) - valid_fields)
        if unknown_keys:
            print(
                "[warning] ignoring unknown config keys: "
                + ", ".join(unknown_keys)
            )

        filtered_config = {k: v for k, v in raw_config.items() if k in valid_fields}
        if "grid_size" in filtered_config:
            filtered_config["grid_size"] = tuple(filtered_config["grid_size"])
        return cls(**filtered_config)

    def save(self, path: str) -> None:
        """Save configuration to JSON file.

        Args:
            path: Output path for JSON configuration file.

        Raises:
            IOError: If the file cannot be written.
        """
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                print(f"Found tensor in {key}: {value}")
            elif isinstance(value, torch.device):
                print(f"Found torch.device in {key}: {value}")
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
