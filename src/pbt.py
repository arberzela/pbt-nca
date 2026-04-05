import copy
import datetime
import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import wandb
from PIL import Image

from config import Config
from model import CASunGroup
from viz import capture_snapshot, generate_nca_colors
from world import World

EPS = 1e-8
SEARCH_SPACE_KEYS = (
    "learning_rate",
    "batch_size",
    "steps_per_update",
    "steps_before_update",
)
PBT_CROSSOVER_PROB = 0.5
PBT_MUTATE_PROB = 0.1
PBT_PERTURB_SCALE = 0.2

LR_LOG_MIN = math.log(1e-6)
LR_LOG_MAX = math.log(1.0)
MIN_STEPS_PER_UPDATE = 1
MAX_STEPS_PER_UPDATE = 64
MIN_STEPS_BEFORE_UPDATE = 0
MAX_STEPS_BEFORE_UPDATE = 63


@dataclass
class WorldScore:
    descriptor: torch.Tensor
    novelty: float
    prompt_alignment: float
    fitness: float


class DescriptorArchive:
    """Fixed-size FIFO archive used for novelty search."""

    def __init__(self, max_size: int, metric: str = "l2") -> None:
        self.max_size = max_size
        self.metric = metric
        self._descriptors: deque[torch.Tensor] = deque()

    def __len__(self) -> int:
        return len(self._descriptors)

    def clear(self) -> None:
        self._descriptors.clear()

    def add_many(self, descriptors: list[torch.Tensor]) -> None:
        for descriptor in descriptors:
            self._descriptors.append(descriptor.detach().cpu().float())
        while len(self._descriptors) > self.max_size:
            self._descriptors.popleft()

    def novelty(self, descriptor: torch.Tensor, k: int) -> float:
        if not self._descriptors:
            return 0.0

        archive = torch.stack(tuple(self._descriptors))
        candidate = descriptor.detach().cpu().unsqueeze(0)
        if self.metric == "cosine":
            archive_norm = archive / archive.norm(dim=1, keepdim=True).clamp_min(EPS)
            candidate_norm = candidate / candidate.norm(dim=1, keepdim=True).clamp_min(EPS)
            similarities = torch.matmul(archive_norm, candidate_norm.t()).squeeze(1)
            distances = 1.0 - similarities
        else:
            distances = torch.linalg.vector_norm(archive - candidate, dim=1)
        if distances.numel() == 0:
            return 0.0

        k_eff = min(k, distances.numel())
        nearest, _ = torch.topk(distances, k_eff, largest=False)
        return float(nearest.mean().item())

    def to_numpy(self) -> np.ndarray:
        if not self._descriptors:
            return np.zeros((0, 0), dtype=np.float32)
        return torch.stack(tuple(self._descriptors)).numpy()


class PopulationMember:
    """One world in the PBT population."""

    def __init__(self, member_id: int, config: Config) -> None:
        self.member_id = member_id
        self.config = config
        self.world = World(config)
        self.group = CASunGroup(config)

    def rollout(self, horizon: int) -> tuple[torch.Tensor, dict[str, Any]]:
        grid = self.world.get_seed()
        trajectory: list[torch.Tensor] = []
        last_stats: dict[str, Any] = {}

        for _ in range(horizon):
            last_stats, grid, grids = self.world.step(self.group, grid)
            trajectory.append(grids.detach())

        if not trajectory:
            trajectory.append(grid.unsqueeze(0).detach())
        return torch.cat(trajectory, dim=0), last_stats


def _species_fractions(alive_traj: torch.Tensor) -> torch.Tensor:
    mass = alive_traj.sum(dim=(-1, -2))
    return mass / mass.sum(dim=-1, keepdim=True).clamp_min(EPS)


def extract_behavior_descriptor(trajectory: torch.Tensor, n_ncas: int) -> torch.Tensor:
    """Extract compact descriptor from rollout trajectory."""
    n_entities = n_ncas + 1
    alive = trajectory[:, :, :n_entities].float().clamp_min(0)
    fractions = _species_fractions(alive)

    mean_fraction = fractions.mean(dim=(0, 1))
    std_fraction = fractions.std(dim=(0, 1), unbiased=False)
    if fractions.shape[0] > 1:
        turnover = (fractions[1:] - fractions[:-1]).abs().mean(dim=(0, 1))
        temporal_volatility = (alive[1:] - alive[:-1]).abs().mean()
    else:
        turnover = torch.zeros_like(mean_fraction)
        temporal_volatility = torch.tensor(0.0, device=alive.device)

    winners = torch.argmax(alive, dim=2)
    winner_hist = torch.bincount(winners.reshape(-1), minlength=n_entities).float()
    winner_hist = winner_hist / winner_hist.sum().clamp_min(EPS)
    winner_entropy = -(winner_hist * torch.log(winner_hist.clamp_min(EPS))).sum()
    winner_entropy = winner_entropy / math.log(max(2, n_entities))

    descriptor = torch.cat(
        [
            mean_fraction,
            std_fraction,
            turnover,
            torch.stack((winner_entropy, temporal_volatility)),
        ]
    )
    descriptor = descriptor.float()
    descriptor = descriptor / descriptor.norm().clamp_min(EPS)
    return descriptor.detach().cpu()


class PopulationTrainerBase:
    """Shared population-training machinery for PBT-style experiments."""

    def __init__(self, config: Config, run: Any | None = None) -> None:
        self.config = config
        self.run = run
        self.rng = random.Random(config.seed)
        archive_metric = (
            config.pbt_vlm_metric if config.pbt_vlm_enabled else "l2"
        )
        self.archive = DescriptorArchive(config.pbt_archive_size, metric=archive_metric)
        self.vlm_enabled = config.pbt_vlm_enabled
        self.vlm_model = None
        self.vlm_processor = None
        self.vlm_tokenizer = None
        self.vlm_text_embeds: torch.Tensor | None = None
        self.vlm_prompts: list[str] = []
        self.vlm_device = config.pbt_vlm_device
        self._vlm_text_logged = False
        self.use_median_ranking = config.pbt_use_median_ranking
        self.use_elo_ranking = config.pbt_use_elo_ranking
        self.use_combined_median = config.pbt_use_combined_median
        self.elo_k = float(config.pbt_elo_k)
        self.dino_model = None
        self.dino_processor = None
        if self.vlm_enabled:
            self._init_vlm()
        if self.use_median_ranking or self.use_elo_ranking or self.use_combined_median:
            self._init_dino()
        if self.run is not None:
            self.run.config.update(
                {
                    "vlm/enabled": bool(self.vlm_enabled),
                    "vlm/model_id": self.config.pbt_vlm_model_id,
                    "vlm/frames": self.config.pbt_vlm_frames,
                    "vlm/metric": self.config.pbt_vlm_metric,
                    "vlm/device": self.config.pbt_vlm_device,
                    "vlm/prompts": self.config.pbt_vlm_prompts,
                    "vlm/prompt_weight": self.config.pbt_vlm_prompt_weight,
                    "pbt_vlm_prompts": self.config.pbt_vlm_prompts,
                    "pbt_vlm_prompt_weight": self.config.pbt_vlm_prompt_weight,
                    "pbt_vlm_only": self.config.pbt_vlm_only,
                    "pbt/weight_noise_std": self.config.pbt_weight_noise_std,
                    "pbt_use_median_ranking": self.config.pbt_use_median_ranking,
                    "pbt_use_elo_ranking": self.config.pbt_use_elo_ranking,
                    "pbt_use_combined_median": self.config.pbt_use_combined_median,
                    "pbt_elo_k": self.config.pbt_elo_k,
                },
                allow_val_change=True,
            )
        self.population = self._build_population()

    def _apply_post_copy_weight_noise(self, member: PopulationMember) -> None:
            std = float(getattr(member.config, "pbt_weight_noise_std", 0.0))
            if std <= 0.0:
                return
            with torch.no_grad():
                for param in member.group.models.model.parameters():
                    if param.requires_grad:
                        param.add_(torch.randn_like(param) * std)

    def _build_population(self) -> list[PopulationMember]:
        raise NotImplementedError

    def train(self) -> str:
        output_dir = self._create_output_dir()
        best_member_idx = 0
        final_scores: list[WorldScore] = []

        meta_step = 0
        try:
            for meta_step in range(1, self.config.pbt_meta_iterations + 1):
                scores, best_traj, best_grid = self._evaluate_population()
                final_scores = scores

                ranked = sorted(
                    range(len(scores)),
                    key=lambda idx: scores[idx].fitness,
                    reverse=True,
                )
                best_member_idx = ranked[0]
                best_member_hparams = self._member_hparams(
                    self.population[best_member_idx]
                )

                top_m = min(self.config.pbt_archive_top_m, len(ranked))
                self.archive.add_many(
                    [scores[idx].descriptor for idx in ranked[:top_m]]
                )

                if (
                    self.config.pbt_archive_reset_interval > 0
                    and meta_step % self.config.pbt_archive_reset_interval == 0
                ):
                    self.archive.clear()

                if meta_step % self.config.pbt_exploit_interval == 0:
                    self._exploit_and_explore(scores)

                self._log_step(
                    meta_step,
                    scores,
                    best_traj,
                    best_grid,
                    best_member_idx,
                    best_member_hparams,
                )
        except KeyboardInterrupt:
            print(f"\nPBT interrupted at meta-iteration {meta_step}")

        self._save_population(
            output_dir, best_member_idx, final_scores, meta_step
        )
        return output_dir

    def _evaluate_population(
        self,
    ) -> tuple[list[WorldScore], torch.Tensor | None, torch.Tensor | None]:
        if self.use_combined_median:
            return self._evaluate_population_with_combined_median()
        if self.use_median_ranking:
            return self._evaluate_population_with_median_ranking()
        if self.use_elo_ranking:
            return self._evaluate_population_with_elo_ranking()

        scores: list[WorldScore] = []
        best_fitness = -float("inf")
        best_traj: torch.Tensor | None = None
        best_grid: torch.Tensor | None = None
        for member in self.population:
            trajectory, _ = member.rollout(self.config.pbt_world_horizon)
            frame_embeds = None
            if self.vlm_enabled:
                descriptor, frame_embeds = self._embed_trajectory(trajectory, member.config)
            else:
                descriptor = extract_behavior_descriptor(
                    trajectory, member.config.n_ncas
                )

            novelty = self.archive.novelty(
                descriptor, self.config.pbt_novelty_k
            )
            prompt_alignment = (
                self._compute_prompt_alignment(descriptor)
                if self.vlm_enabled
                else 0.0
            )

            if self.config.pbt_vlm_only:
                fitness = prompt_alignment
            else:
                fitness = novelty + self.config.pbt_vlm_prompt_weight * prompt_alignment

            if fitness > best_fitness:
                best_fitness = fitness
                best_traj = trajectory.detach().cpu()
                best_grid = trajectory[-1].detach()
            scores.append(
                WorldScore(
                    descriptor=descriptor,
                    novelty=novelty,
                    prompt_alignment=prompt_alignment,
                    fitness=fitness,
                )
            )

        return scores, best_traj, best_grid

    def _init_dino(self) -> None:
        from transformers import AutoImageProcessor, AutoModel

        model_id = "facebook/dinov2-base"
        hf_token = os.environ.get("HF_TOKEN")
        self.dino_processor = AutoImageProcessor.from_pretrained(model_id, token=hf_token)
        self.dino_model = AutoModel.from_pretrained(model_id, token=hf_token)
        self.dino_model.to(self.vlm_device)
        self.dino_model.eval()

    def _embed_trajectory_dino(self, trajectory: torch.Tensor) -> torch.Tensor:
        frames = [trajectory[t] for t in range(trajectory.shape[0])]
        images = [self._render_frame(frame) for frame in frames]
        inputs = self.dino_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.vlm_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embeds = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                embeds = outputs.last_hidden_state[:, 0]
            elif isinstance(outputs, (tuple, list)) and outputs:
                embeds = outputs[0]
            else:
                raise TypeError(
                    f"Unsupported DINO output type: {type(outputs)}"
                )
        embeds = embeds.float()
        embeds = embeds / embeds.norm(dim=-1, keepdim=True).clamp_min(EPS)
        return embeds.detach()

    def _median_distance_scores(self, embeds_by_member: list[torch.Tensor]) -> torch.Tensor:
        t_len = min(embeds.shape[0] for embeds in embeds_by_member)
        embeds = torch.stack([e[:t_len] for e in embeds_by_member], dim=0)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True).clamp_min(EPS)
        n_members = embeds.shape[0]
        device = embeds.device
        diag_mask = torch.eye(n_members, device=device, dtype=torch.bool)
        per_timestep_scores = []
        for t in range(t_len):
            frame = embeds[:, t]
            similarities = torch.matmul(frame, frame.t())
            distances = 1.0 - similarities
            distances = distances.masked_fill(diag_mask, float("nan"))
            median_scores = torch.nanmedian(distances, dim=1).values
            per_timestep_scores.append(median_scores)
        return torch.stack(per_timestep_scores, dim=0).mean(dim=0)

    def _elo_ranking_scores(self, embeds_by_member: list[torch.Tensor]) -> torch.Tensor:
        t_len = min(embeds.shape[0] for embeds in embeds_by_member)
        embeds = torch.stack([e[:t_len] for e in embeds_by_member], dim=0)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True).clamp_min(EPS)
        n_members = embeds.shape[0]
        device = embeds.device
        diag_mask = torch.eye(n_members, device=device, dtype=torch.bool)
        ratings = torch.full((n_members,), 1000.0, device=device)

        for t in range(t_len):
            frame = embeds[:, t]
            similarities = torch.matmul(frame, frame.t())
            distances = 1.0 - similarities
            distances = distances.masked_fill(diag_mask, float("nan"))
            median_scores = torch.nanmedian(distances, dim=1).values
            for i in range(n_members):
                for j in range(i + 1, n_members):
                    score_i = 1.0 if median_scores[i] > median_scores[j] else 0.0
                    score_j = 1.0 - score_i
                    expected_i = 1.0 / (1.0 + 10.0 ** ((ratings[j] - ratings[i]) / 400.0))
                    expected_j = 1.0 - expected_i
                    ratings[i] = ratings[i] + self.elo_k * (score_i - expected_i)
                    ratings[j] = ratings[j] + self.elo_k * (score_j - expected_j)
        return ratings

    def _evaluate_population_with_median_ranking(
        self,
    ) -> tuple[list[WorldScore], torch.Tensor | None, torch.Tensor | None]:
        scores: list[WorldScore] = []
        best_idx = -1
        best_fitness = -float("inf")
        best_traj: torch.Tensor | None = None
        best_grid: torch.Tensor | None = None

        trajectories: list[torch.Tensor] = []
        descriptors: list[torch.Tensor] = []
        embeddings: list[torch.Tensor] = []

        for member in self.population:
            trajectory, _ = member.rollout(self.config.pbt_world_horizon)
            trajectories.append(trajectory)
            descriptors.append(
                extract_behavior_descriptor(trajectory, member.config.n_ncas)
            )
            embeddings.append(self._embed_trajectory_dino(trajectory))

        fitness_values = self._median_distance_scores(embeddings)

        for idx, member in enumerate(self.population):
            trajectory = trajectories[idx]
            descriptor = descriptors[idx]
            fitness = float(fitness_values[idx].item())
            if fitness > best_fitness:
                best_fitness = fitness
                best_idx = idx
                best_traj = trajectory.detach().cpu()
                best_grid = trajectory[-1].detach()

            scores.append(
                WorldScore(
                    descriptor=descriptor,
                    novelty=0.0,
                    prompt_alignment=0.0,
                    fitness=fitness,
                )
            )

        return scores, best_traj, best_grid

    def _evaluate_population_with_elo_ranking(
        self,
    ) -> tuple[list[WorldScore], torch.Tensor | None, torch.Tensor | None]:
        scores: list[WorldScore] = []
        best_idx = -1
        best_fitness = -float("inf")
        best_traj: torch.Tensor | None = None
        best_grid: torch.Tensor | None = None

        trajectories: list[torch.Tensor] = []
        descriptors: list[torch.Tensor] = []
        embeddings: list[torch.Tensor] = []

        for member in self.population:
            trajectory, _ = member.rollout(self.config.pbt_world_horizon)
            trajectories.append(trajectory)
            descriptors.append(
                extract_behavior_descriptor(trajectory, member.config.n_ncas)
            )
            embeddings.append(self._embed_trajectory_dino(trajectory))

        fitness_values = self._elo_ranking_scores(embeddings)

        for idx, member in enumerate(self.population):
            trajectory = trajectories[idx]
            descriptor = descriptors[idx]
            fitness = float(fitness_values[idx].item())
            if fitness > best_fitness:
                best_fitness = fitness
                best_idx = idx
                best_traj = trajectory.detach().cpu()
                best_grid = trajectory[-1].detach()

            scores.append(
                WorldScore(
                    descriptor=descriptor,
                    novelty=0.0,
                    prompt_alignment=0.0,
                    fitness=fitness,
                )
            )

        return scores, best_traj, best_grid

    def _evaluate_population_with_combined_median(
        self,
    ) -> tuple[list[WorldScore], torch.Tensor | None, torch.Tensor | None]:
        scores: list[WorldScore] = []
        best_idx = -1
        best_fitness = -float("inf")
        best_traj: torch.Tensor | None = None
        best_grid: torch.Tensor | None = None

        trajectories: list[torch.Tensor] = []
        descriptors: list[torch.Tensor] = []
        embeddings: list[torch.Tensor] = []

        for member in self.population:
            trajectory, _ = member.rollout(self.config.pbt_world_horizon)
            trajectories.append(trajectory)
            descriptors.append(
                extract_behavior_descriptor(trajectory, member.config.n_ncas)
            )
            embeddings.append(self._embed_trajectory_dino(trajectory))

        median_scores = self._median_distance_scores(embeddings)

        for idx, member in enumerate(self.population):
            trajectory = trajectories[idx]
            descriptor = descriptors[idx]
            novelty = self.archive.novelty(
                descriptor, self.config.pbt_novelty_k
            )
            median_fitness = float(median_scores[idx].item())
            fitness = novelty + median_fitness

            if fitness > best_fitness:
                best_fitness = fitness
                best_idx = idx
                best_traj = trajectory.detach().cpu()
                best_grid = trajectory[-1].detach()

            scores.append(
                WorldScore(
                    descriptor=descriptor,
                    novelty=novelty,
                    prompt_alignment=0.0,
                    fitness=fitness,
                )
            )

        return scores, best_traj, best_grid

    def _init_vlm(self) -> None:
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

        model_id = self.config.pbt_vlm_model_id
        hf_token = os.environ.get("HF_TOKEN")
        # Image-only path avoids loading text tokenizers (e.g., sentencepiece),
        # which are unnecessary for descriptor extraction.
        self.vlm_processor = AutoImageProcessor.from_pretrained(model_id, token=hf_token)
        self.vlm_model = AutoModel.from_pretrained(model_id, token=hf_token)
        if not hasattr(self.vlm_model, "get_image_features"):
            raise ValueError(
                f"Model {model_id} does not expose get_image_features for image embeddings."
            )
        self.vlm_model.to(self.vlm_device)
        self.vlm_model.eval()
        self.vlm_prompts = [
            prompt.strip()
            for prompt in self.config.pbt_vlm_prompts.split(";")
            if prompt.strip()
        ]
        if not self.vlm_prompts:
            return
        if not hasattr(self.vlm_model, "get_text_features"):
            if self.config.pbt_vlm_prompt_weight > 0:
                raise ValueError(
                    f"Model {model_id} does not expose get_text_features for prompt guidance."
                )
            return
        self.vlm_tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        text_inputs = self.vlm_tokenizer(
            self.vlm_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_inputs = {k: v.to(self.vlm_device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_embeds = self.vlm_model.get_text_features(**text_inputs)
            if not torch.is_tensor(text_embeds):
                if hasattr(text_embeds, "text_embeds"):
                    text_embeds = text_embeds.text_embeds
                elif hasattr(text_embeds, "pooler_output"):
                    text_embeds = text_embeds.pooler_output
                elif isinstance(text_embeds, (tuple, list)) and text_embeds:
                    text_embeds = text_embeds[0]
                else:
                    raise TypeError(
                        f"Unsupported text embedding output type: {type(text_embeds)}"
                    )
        text_embeds = text_embeds.float()
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True).clamp_min(EPS)
        self.vlm_text_embeds = text_embeds.detach().cpu()

    def _embed_trajectory(
        self, trajectory: torch.Tensor, config: Config
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frames = self._sample_frames(trajectory, config.pbt_vlm_frames)
        images = [self._render_frame(frame) for frame in frames]
        inputs = self.vlm_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.vlm_device) for k, v in inputs.items()}
        with torch.no_grad():
            image_embeds = self.vlm_model.get_image_features(**inputs)
            if not torch.is_tensor(image_embeds):
                if hasattr(image_embeds, "image_embeds"):
                    image_embeds = image_embeds.image_embeds
                elif hasattr(image_embeds, "pooler_output"):
                    image_embeds = image_embeds.pooler_output
                elif isinstance(image_embeds, (tuple, list)) and image_embeds:
                    image_embeds = image_embeds[0]
                else:
                    raise TypeError(
                        f"Unsupported embedding output type: {type(image_embeds)}"
                    )
        image_embeds = image_embeds.float()
        image_embeds = image_embeds / image_embeds.norm(
            dim=-1, keepdim=True
        ).clamp_min(EPS)
        pooled = image_embeds.mean(dim=0)
        pooled = pooled / pooled.norm().clamp_min(EPS)
        return pooled.detach().cpu(), image_embeds.detach().cpu()

    def _compute_prompt_alignment(self, descriptor: torch.Tensor) -> float:
        if self.vlm_text_embeds is None or self.config.pbt_vlm_prompt_weight <= 0:
            return 0.0
        descriptor = descriptor.detach().cpu().float()
        descriptor = descriptor / descriptor.norm().clamp_min(EPS)
        similarities = torch.matmul(self.vlm_text_embeds, descriptor)
        return float(similarities.max().item())

    def _sample_frames(
        self, trajectory: torch.Tensor, n_frames: int
    ) -> list[torch.Tensor]:
        total_steps = trajectory.shape[0]
        if total_steps <= 1 or n_frames <= 1:
            return [trajectory[-1]]
        idx = torch.linspace(0, total_steps - 1, steps=n_frames)
        idx = idx.round().long().clamp(0, total_steps - 1)
        return [trajectory[i] for i in idx]

    def _render_frame(self, grid: torch.Tensor) -> Image.Image:
        nca_colors = generate_nca_colors(self.config.n_ncas)
        rgba = capture_snapshot(grid, nca_colors)
        rgb = rgba[:3].clamp(0.0, 1.0)
        rgb = (rgb.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")

    def _exploit_and_explore(self, scores: list[WorldScore]) -> None:
        raise NotImplementedError


class PBTTrainer(PopulationTrainerBase):
    """Population-based training with archive novelty selection."""

    def _build_population(self) -> list[PopulationMember]:
        members: list[PopulationMember] = []
        for member_id in range(self.config.pbt_population_size):
            member_cfg = copy.deepcopy(self.config)
            member_cfg.seed = self.config.seed + member_id
            with member_cfg.seed_context():
                member = PopulationMember(member_id, member_cfg)
                self._sample_member_from_prior(member)
            members.append(member)

        return members

    def _exploit_and_explore(self, scores: list[WorldScore]) -> None:
        pop_size = len(self.population)
        n_replace = int(round(pop_size * self.config.pbt_replace_fraction))
        n_replace = min(pop_size - 1, max(1, n_replace))

        ranked = sorted(
            range(pop_size), key=lambda idx: scores[idx].fitness, reverse=True
        )
        top_indices = ranked[: pop_size - n_replace]
        bottom_indices = ranked[pop_size - n_replace:]

        for child_idx in bottom_indices:
            parent_idx = self.rng.choice(top_indices)
            child = self.population[child_idx]
            previous_child_config = copy.deepcopy(child.config)

            self._copy_member_state(
                parent=self.population[parent_idx],
                child=child,
            )
            self._crossover_hparams(child, previous_child_config)
            self._mutate_member(child)
            self._apply_post_copy_weight_noise(child)

    def _copy_member_state(
        self, parent: PopulationMember, child: PopulationMember
    ) -> None:
        child.config = copy.deepcopy(parent.config)
        child.world.config = child.config

        if child.config.pbt_inheritance_mode == "lamarck":
            child.group.models.model.load_state_dict(
                copy.deepcopy(parent.group.models.model.state_dict())
            )
            child.group.models.optimizer.load_state_dict(
                copy.deepcopy(parent.group.models.optimizer.state_dict())
            )
            child.group.sun_update.data.copy_(parent.group.sun_update.data)
            child.group.sun_optim.load_state_dict(
                copy.deepcopy(parent.group.sun_optim.state_dict())
            )
            child.group.softmax_temp = parent.group.softmax_temp
            child.group.per_hid_upd = parent.group.per_hid_upd
            child.group.threshold.data.copy_(parent.group.threshold.data)
        elif child.config.pbt_inheritance_mode == "darwin":
            child.group = CASunGroup(child.config)
        else:
            raise ValueError(
                f"Unknown pbt_inheritance_mode: {child.config.pbt_inheritance_mode}"
            )

        child.world.pool.copy_(parent.world.pool)
        if hasattr(parent.world, "seed_update"):
            child.world.seed_update = parent.world.seed_update.clone()
        child.world.state = self._clone_state(parent.world.state)
        child.world.epoch = parent.world.epoch
        child.world.steps_taken = parent.world.steps_taken

    def _clone_state(self, state: dict[str, Any]) -> dict[str, Any]:
        cloned: dict[str, Any] = {}
        for key, value in state.items():
            if torch.is_tensor(value):
                cloned[key] = value.clone()
            else:
                cloned[key] = copy.deepcopy(value)
        return cloned

    def _sync_member_runtime(self, member: PopulationMember) -> None:
        self._enforce_search_space_bounds(member.config)
        member.config.validate_search_space()
        assert (
            member.config.batch_size <= self.config.batch_size
        ), "[pbt] batch_size exceeded base config bound"
        assert (
            member.config.steps_per_update <= self.config.steps_per_update
        ), "[pbt] steps_per_update exceeded base config bound"
        assert (
            member.config.steps_before_update <= self.config.steps_before_update
        ), "[pbt] steps_before_update exceeded base config bound"
        member.world.config = member.config
        member.world.batch_size = member.config.batch_size
        member.group.batch_size = member.config.batch_size
        member.world.state["steps_per_update"] = member.config.steps_per_update
        member.world.state["steps_before_update"] = member.config.steps_before_update
        for feature in getattr(member.world, "features", []):
            if hasattr(feature, "target_steps_per"):
                feature.target_steps_per = member.config.steps_per_update
            if hasattr(feature, "target_steps_before"):
                feature.target_steps_before = member.config.steps_before_update
        for optimizer in (
            member.group.models.optimizer,
            member.group.sun_optim,
        ):
            for param_group in optimizer.param_groups:
                param_group["lr"] = member.config.learning_rate

    def _enforce_search_space_bounds(self, cfg: Config) -> None:
        """Keep mutable search-space values within base-config memory envelope."""
        cfg.batch_size = int(
            np.clip(
                cfg.batch_size,
                1,
                min(cfg.pool_size, self.config.batch_size),
            )
        )
        cfg.steps_per_update = int(
            np.clip(
                cfg.steps_per_update,
                MIN_STEPS_PER_UPDATE,
                self.config.steps_per_update,
            )
        )
        cfg.steps_before_update = int(
            np.clip(
                cfg.steps_before_update,
                MIN_STEPS_BEFORE_UPDATE,
                self.config.steps_before_update,
            )
        )

    def _sample_member_from_prior(self, member: PopulationMember) -> None:
        cfg = member.config
        cfg.learning_rate = float(
            math.exp(self.rng.uniform(LR_LOG_MIN, LR_LOG_MAX))
        )
        cfg.batch_size = self.rng.randint(
            1,
            min(cfg.pool_size, self.config.batch_size),
        )
        cfg.steps_per_update = self.rng.randint(
            MIN_STEPS_PER_UPDATE,
            self.config.steps_per_update,
        )
        cfg.steps_before_update = self.rng.randint(
            MIN_STEPS_BEFORE_UPDATE,
            self.config.steps_before_update,
        )
        self._enforce_search_space_bounds(cfg)
        self._sync_member_runtime(member)

    def _crossover_hparams(
        self, child: PopulationMember, previous_child_config: Config
    ) -> None:
        for key in SEARCH_SPACE_KEYS:
            if self.rng.random() >= PBT_CROSSOVER_PROB:
                setattr(child.config, key, getattr(previous_child_config, key))
        self._sync_member_runtime(child)

    def _mutate_learning_rate(self, value: float) -> float:
        factor = (
            1.0 + PBT_PERTURB_SCALE
            if self.rng.random() < 0.5
            else 1.0 - PBT_PERTURB_SCALE
        )
        return float(np.clip(value * factor, 1e-6, 1.0))

    def _mutate_integer(
        self,
        value: int,
        lower: int,
        upper: int,
    ) -> int:
        base = max(1, value)
        factor = (
            1.0 + PBT_PERTURB_SCALE
            if self.rng.random() < 0.5
            else 1.0 - PBT_PERTURB_SCALE
        )
        mutated = int(round(base * factor))
        return int(np.clip(mutated, lower, upper))

    def _mutate_member(self, member: PopulationMember) -> None:
        cfg = member.config

        if self.rng.random() < PBT_MUTATE_PROB:
            cfg.learning_rate = self._mutate_learning_rate(cfg.learning_rate)

        if self.rng.random() < PBT_MUTATE_PROB:
            cfg.batch_size = self._mutate_integer(
                cfg.batch_size,
                1,
                min(cfg.pool_size, self.config.batch_size),
            )

        if self.rng.random() < PBT_MUTATE_PROB:
            cfg.steps_per_update = self._mutate_integer(
                cfg.steps_per_update,
                MIN_STEPS_PER_UPDATE,
                self.config.steps_per_update,
            )

        if self.rng.random() < PBT_MUTATE_PROB:
            cfg.steps_before_update = self._mutate_integer(
                cfg.steps_before_update,
                MIN_STEPS_BEFORE_UPDATE,
                self.config.steps_before_update,
            )

        self._enforce_search_space_bounds(cfg)
        self._sync_member_runtime(member)

    def _member_hparams(self, member: PopulationMember) -> dict[str, int | float]:
        hparams: dict[str, int | float] = {"member_id": member.member_id}
        for key in SEARCH_SPACE_KEYS:
            hparams[key] = getattr(member.config, key)
        return hparams

    @staticmethod
    def _capture_global_rng_state() -> dict[str, Any]:
        import random

        cuda_states = None
        if torch.cuda.is_available():
            cuda_states = [
                torch.cuda.get_rng_state(i)
                for i in range(torch.cuda.device_count())
            ]
        mps_state = None
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            mps_state = torch.mps.get_rng_state()
        return {
            "py": random.getstate(),
            "np": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": cuda_states,
            "mps": mps_state,
        }

    @staticmethod
    def _restore_global_rng_state(states: dict[str, Any]) -> None:
        import random

        random.setstate(states["py"])
        np.random.set_state(states["np"])
        torch.random.set_rng_state(states["torch"])
        if states["cuda"] is not None:
            for device_idx, cuda_state in enumerate(states["cuda"]):
                torch.cuda.set_rng_state(cuda_state, device_idx)
        if states["mps"] is not None:
            torch.mps.set_rng_state(states["mps"])

    def _extend_best_traj_for_viz(
        self, member_idx: int, best_traj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extend the evaluation trajectory for logging without changing training.

        Reuses the exact prefix from ``best_traj`` (same rollout as fitness), then
        continues from its last frame in ``frozen_eval``. Restores world fields and
        global RNG (torch / numpy / python / CUDA) afterward.
        """
        member = self.population[member_idx]
        prefix = best_traj.detach().cpu()
        horizon = prefix.shape[0]
        mult = max(1, int(self.config.pbt_viz_rollout_multiplier))
        target_len = max(1, int(self.config.pbt_world_horizon * mult))
        extra = max(0, target_len - horizon)

        if extra == 0:
            return prefix, prefix[-1]

        rng_snapshot = self._capture_global_rng_state()
        saved_mode = member.world.state.get("mode", member.config.mode)
        saved_state = self._clone_state(member.world.state)
        saved_pool = member.world.pool.clone()
        saved_epoch = member.world.epoch
        saved_steps_taken = member.world.steps_taken

        try:
            member.world.state["mode"] = "frozen_eval"
            device = member.world.device
            dtype = member.world.dtype
            grid = prefix[-1].to(device=device, dtype=dtype).detach().clone()
            suffix_frames: list[torch.Tensor] = []
            with torch.no_grad():
                for _ in range(extra):
                    _, grid, grids = member.world.step(member.group, grid)
                    suffix_frames.append(grids.detach().cpu())
            suffix = torch.cat(suffix_frames, dim=0)
            full = torch.cat([prefix, suffix], dim=0)
        finally:
            member.world.state = saved_state
            member.world.state["mode"] = saved_mode
            member.world.pool.copy_(saved_pool)
            member.world.epoch = saved_epoch
            member.world.steps_taken = saved_steps_taken
            self._restore_global_rng_state(rng_snapshot)

        return full, full[-1]

    def _log_step(
        self,
        meta_step: int,
        scores: list[WorldScore],
        best_traj: torch.Tensor | None,
        best_grid: torch.Tensor | None,
        best_member_idx: int,
        best_member_hparams: dict[str, int | float],
    ) -> None:
        fitness_values = np.array([s.fitness for s in scores], dtype=np.float64)
        novelty_values = np.array([s.novelty for s in scores], dtype=np.float64)

        best_fitness = float(fitness_values.max()) if fitness_values.size else 0.0
        if fitness_values.size:
            mean_fitness = float(fitness_values.mean())
            mean_novelty = float(novelty_values.mean())
            mean_prompt_alignment = 0.0
        else:
            mean_fitness = 0.0
            mean_novelty = 0.0
            mean_prompt_alignment = 0.0

        hparam_summary = ", ".join(
            f"{key}={value}"
            for key, value in best_member_hparams.items()
        )

        print(
            f"Meta {meta_step:5d} | fitness(best/mean)="
            f"{best_fitness:.3f}/{mean_fitness:.3f} | "
            f"nov={mean_novelty:.3f} | archive={len(self.archive)} | "
            f"best_member={best_member_idx} | {hparam_summary}"
        )

        if self.run:
            metrics = {
                "meta/step": meta_step,
                "meta/best_member_idx": best_member_idx,
                "fitness/best": best_fitness,
                "fitness/mean": mean_fitness,
                "fitness/novelty_mean": mean_novelty,
                "fitness/prompt_alignment_mean": mean_prompt_alignment,
                "fitness/prompt_alignment_best": 0.0,
                "fitness/prompt_guidance_term_mean": self.config.pbt_vlm_prompt_weight
                * mean_prompt_alignment,
                "archive/size": len(self.archive),
            }

            metrics.update(
                {
                    f"hparams/best_member/{key}": value
                    for key, value in best_member_hparams.items()
                }
            )
            
            if self.config.pbt_use_median_ranking:
                metrics["fitness/median_rank_mean"] = mean_fitness
                metrics["fitness/median_rank_best"] = best_fitness
            if self.config.pbt_use_elo_ranking:
                metrics["fitness/elo_mean"] = mean_fitness
                metrics["fitness/elo_best"] = best_fitness

            if self.vlm_enabled and not self._vlm_text_logged:
                metrics["vlm/config_text"] = (
                    f"model_id={self.config.pbt_vlm_model_id} | "
                    f"frames={self.config.pbt_vlm_frames} | "
                    f"metric={self.config.pbt_vlm_metric} | "
                    f"device={self.config.pbt_vlm_device} | "
                    f"prompts={self.config.pbt_vlm_prompts!r} | "
                    f"prompt_weight={self.config.pbt_vlm_prompt_weight} | "
                    f"vlm_only={self.config.pbt_vlm_only}"
                )
                self._vlm_text_logged = True
            if (
                self.config.pbt_viz_interval > 0
                and meta_step % self.config.pbt_viz_interval == 0
                and best_traj is not None
                and best_grid is not None
            ):
                viz_traj = best_traj
                viz_grid = best_grid
                try:
                    viz_traj, viz_grid = self._extend_best_traj_for_viz(
                        best_member_idx, best_traj
                    )
                except Exception as exc:
                    print(
                        "[warning] failed extended frozen viz rollout "
                        f"(falling back to normal trajectory): {exc}"
                    )

                nca_colors = generate_nca_colors(self.config.n_ncas)
                frames = [
                    capture_snapshot(viz_traj[t], nca_colors)
                    for t in range(viz_traj.shape[0])
                ]
                frame_images = [
                    wandb.Image(frame, caption=f"Step {i}")
                    for i, frame in enumerate(frames)
                ]
                metrics["viz/frame_sequence"] = frame_images

                video_frames = (torch.stack(frames) * 255).to(torch.uint8)
                video_array = video_frames.numpy()
                metrics["viz/growth"] = wandb.Video(
                    video_array, format="gif", fps=self.config.pbt_viz_fps
                )

                frame = capture_snapshot(viz_grid, nca_colors)
                metrics["viz/final_territory"] = wandb.Image(
                    frame, caption=f"Meta {meta_step} best member"
                )
            self.run.log(metrics, step=meta_step)

    def _create_output_dir(self) -> str:
        base_name = (
            self.run.name
            if self.run is not None
            else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        base_name = base_name.replace("/", "_")
        suffix = "vlm_siglip" if self.config.pbt_vlm_enabled else "baseline"
        root = f"{base_name}_{suffix}_pbt"

        if not os.path.exists(root):
            os.mkdir(root)
            return root

        suffix = 1
        while True:
            candidate = f"{root}_{suffix:02d}"
            if not os.path.exists(candidate):
                os.mkdir(candidate)
                return candidate
            suffix += 1

    def _save_population(
        self,
        output_dir: str,
        best_member_idx: int,
        scores: list[WorldScore],
        last_step: int,
    ) -> None:
        archive_np = self.archive.to_numpy()
        np.save(f"{output_dir}/archive.npy", archive_np)
        if self.config.pbt_vlm_enabled:
            np.save(f"{output_dir}/archive_vlm.npy", archive_np)

        summary = {
            "last_meta_step": last_step,
            "best_member_idx": best_member_idx,
            "archive_size": len(self.archive),
            "population_size": len(self.population),
            "pbt_inheritance_mode": self.config.pbt_inheritance_mode,
            "pbt_weight_noise_std": self.config.pbt_weight_noise_std,
            "pbt_vlm_enabled": self.config.pbt_vlm_enabled,
            "pbt_vlm_model_id": self.config.pbt_vlm_model_id,
            "pbt_vlm_frames": self.config.pbt_vlm_frames,
            "pbt_vlm_metric": self.config.pbt_vlm_metric,
            "pbt_vlm_prompts": self.config.pbt_vlm_prompts,
            "pbt_vlm_prompt_weight": self.config.pbt_vlm_prompt_weight,
            "pbt_vlm_only": self.config.pbt_vlm_only,
        }
        if scores:
            summary["best_fitness"] = max(s.fitness for s in scores)
            summary["mean_fitness"] = float(np.mean([s.fitness for s in scores]))
            summary["mean_prompt_alignment"] = float(
                np.mean([s.prompt_alignment for s in scores])
            )
            final_descriptors = np.stack(
                [s.descriptor.detach().cpu().float().numpy() for s in scores], axis=0
            )
            np.save(f"{output_dir}/final_population_descriptors.npy", final_descriptors)
            if self.config.pbt_vlm_enabled:
                # Save a VLM-specific copy so the artifact is self-describing.
                np.save(f"{output_dir}/final_population_vlm_descriptors.npy", final_descriptors)

        with open(f"{output_dir}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        if self.config.pbt_vlm_enabled:
            vlm_metadata = {
                "model_id": self.config.pbt_vlm_model_id,
                "frames": self.config.pbt_vlm_frames,
                "metric": self.config.pbt_vlm_metric,
                "prompts": self.config.pbt_vlm_prompts,
                "prompt_weight": self.config.pbt_vlm_prompt_weight,
                "vlm_only": self.config.pbt_vlm_only,
                "archive_path": "archive_vlm.npy",
                "population_descriptor_path": "final_population_vlm_descriptors.npy",
            }
            with open(f"{output_dir}/vlm_metadata.json", "w") as f:
                json.dump(vlm_metadata, f, indent=2)
            if self.run is not None:
                self.run.config.update(
                    {
                        "pbt/inheritance_mode": self.config.pbt_inheritance_mode,
                        "vlm/model_id": vlm_metadata["model_id"],
                        "vlm/frames": vlm_metadata["frames"],
                        "vlm/metric": vlm_metadata["metric"],
                        "vlm/prompts": vlm_metadata["prompts"],
                        "vlm/prompt_weight": vlm_metadata["prompt_weight"],
                        "pbt_vlm_prompts": ";".join(self.vlm_prompts),
                        "pbt_vlm_prompt_weight": vlm_metadata["prompt_weight"],
                        "pbt_vlm_only": vlm_metadata["vlm_only"],
                        "vlm/archive_path": f"{output_dir}/archive_vlm.npy",
                        "vlm/population_descriptor_path": (
                            f"{output_dir}/final_population_vlm_descriptors.npy"
                        ),
                        "vlm/metadata_path": f"{output_dir}/vlm_metadata.json",
                    },
                    allow_val_change=True,
                )

        for member in self.population:
            member_dir = f"{output_dir}/member_{member.member_id:03d}"
            member.group.save(member.config, member_dir)
            member.world.save(member.config, member_dir)

        print(f"Saved PBT outputs to {output_dir}")


def train_non_stationary_pbt(config: Config, run: Any | None = None) -> str:
    trainer = PBTTrainer(config, run)
    return trainer.train()
