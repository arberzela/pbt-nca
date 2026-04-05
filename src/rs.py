"""Random-resampling baseline for PBT comparison.

At every ``K`` meta-iterations (reusing ``pbt_exploit_interval`` as K) every
population member independently resamples its hyperparameters from the prior
distribution.  There is no fitness-based selection, no weight copying, and no
archive-guided decisions.  All evaluation machinery and logging are
identical to
:class:`~pbt.NonStationaryPBTTrainer` so the runs are directly comparable on
every metric.

This is *not* canonical random search.  It is a no-selection hyperparameter
resampling baseline that isolates the role of selective pressure while keeping
the PBT meta-iteration structure intact.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import os
from typing import Any

import numpy as np
import torch
import wandb

from config import Config
from pbt import (
    PBTTrainer,
    PopulationMember,
    WorldScore,
)
from viz import capture_snapshot, generate_nca_colors

# Hyperparameter prior bounds.
# Log-uniform priors for scale hyperparameters; uniform for bounded quantities.
# Bounds match the clamp limits used in PBT's _mutate_member so the two methods
# explore the same search space.
_LR_LOG_MIN: float = math.log(1e-6)
_LR_LOG_MAX: float = math.log(1.0)
_BATCH_SIZE_MIN: int = 1


class RandomSearchTrainer(PBTTrainer):
    """No-selection hyperparameter-resampling baseline."""

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
        del scores
        for member in self.population:
            self._sample_member_from_prior(member)

    def _create_output_dir(self) -> str:
        base_name = (
            self.run.name
            if self.run is not None
            else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        base_name = base_name.replace("/", "_")
        suffix = "vlm_siglip" if self.config.pbt_vlm_enabled else "baseline"
        root = f"{base_name}_{suffix}_random_resampling"

        if not os.path.exists(root):
            os.mkdir(root)
            return root

        counter = 1
        while True:
            candidate = f"{root}_{counter:02d}"
            if not os.path.exists(candidate):
                os.mkdir(candidate)
                return candidate
            counter += 1

    # Logging.

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
        mean_fitness = float(fitness_values.mean()) if fitness_values.size else 0.0
        mean_novelty = float(novelty_values.mean()) if novelty_values.size else 0.0
        hparam_summary = ", ".join(
            f"{key}={value}" for key, value in best_member_hparams.items()
        )

        print(
            f"RRS  {meta_step:5d} | fitness(best/mean)="
            f"{best_fitness:.3f}/{mean_fitness:.3f} | "
            f"nov={mean_novelty:.3f} | "
            f"archive={len(self.archive)} | "
            f"best_member={best_member_idx} | {hparam_summary}"
        )

        if self.run is None:
            return

        metrics: dict[str, Any] = {
            "meta/step": meta_step,
            "meta/best_member_idx": best_member_idx,
            "fitness/best": best_fitness,
            "fitness/mean": mean_fitness,
            "fitness/novelty_mean": mean_novelty,
            "fitness/prompt_alignment_mean": 0.0,
            "fitness/prompt_alignment_best": 0.0,
            "fitness/prompt_guidance_term_mean": 0.0,
            "archive/size": len(self.archive),
        }
        metrics.update(
            {
                f"hparams/best_member/{key}": value
                for key, value in best_member_hparams.items()
            }
        )

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
            nca_colors = generate_nca_colors(self.config.n_ncas)
            frames = [
                capture_snapshot(best_traj[t], nca_colors)
                for t in range(best_traj.shape[0])
            ]
            metrics["viz/frame_sequence"] = [
                wandb.Image(f, caption=f"Step {i}") for i, f in enumerate(frames)
            ]
            video_frames = (torch.stack(frames) * 255).to(torch.uint8)
            metrics["viz/growth"] = wandb.Video(
                video_frames.numpy(), format="gif"
            )
            metrics["viz/final_territory"] = wandb.Image(
                capture_snapshot(best_grid, nca_colors),
                caption=f"Meta {meta_step} best member",
            )

        self.run.log(metrics, step=meta_step)

    # Save.

    def _save_population(
        self,
        output_dir: str,
        best_member_idx: int,
        scores: list[WorldScore],
        last_step: int,
    ) -> None:
        archive_np = self.archive.to_numpy()
        np.save(f"{output_dir}/archive.npy", archive_np)

        summary: dict[str, Any] = {
            "trainer": "random_resampling",
            "last_meta_step": last_step,
            "best_member_idx": best_member_idx,
            "archive_size": len(self.archive),
            "population_size": len(self.population),
            "resample_interval": self.config.pbt_exploit_interval,
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
            summary["mean_fitness"] = float(
                np.mean([s.fitness for s in scores])
            )
            summary["mean_prompt_alignment"] = 0.0
            final_descriptors = np.stack(
                [s.descriptor.detach().cpu().float().numpy() for s in scores],
                axis=0,
            )
            np.save(
                f"{output_dir}/final_population_descriptors.npy",
                final_descriptors,
            )

        with open(f"{output_dir}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        for member in self.population:
            member_dir = f"{output_dir}/member_{member.member_id:03d}"
            member.group.save(member.config, member_dir)
            member.world.save(member.config, member_dir)

        print(f"Saved random-resampling outputs to {output_dir}")


# Public entry point.


def train_random_resampling(config: Config, run: Any | None = None) -> str:
    """Run the random-resampling baseline and return the output directory."""
    trainer = RandomSearchTrainer(config, run)
    return trainer.train()


def train_random_search(config: Config, run: Any | None = None) -> str:
    """Backward-compatible alias for the random-resampling baseline."""
    return train_random_resampling(config, run)


# CLI.


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Random-resampling baseline: population with periodic "
            "hyperparameter "
            "resampling and no selective pressure."
        )
    )
    parser.add_argument(
        "--config", required=True, help="Path to JSON config file"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Override compute device",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable W&B logging"
    )
    parser.add_argument(
        "--meta-iterations",
        type=int,
        help="Number of meta-iterations (overrides config)",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        help="Population size (overrides config)",
    )
    parser.add_argument(
        "--resample-interval",
        type=int,
        help=(
            "Hparam resample interval K "
            "(overrides pbt_exploit_interval in config)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = Config.from_file(args.config)

    if args.device:
        config.device = args.device
    if args.wandb:
        config.wandb = True
    if args.meta_iterations:
        config.pbt_meta_iterations = args.meta_iterations
    if args.population_size:
        config.pbt_population_size = args.population_size
    if args.resample_interval:
        config.pbt_exploit_interval = args.resample_interval

    # rs.py always runs in PBT mode (population-based evaluation).
    config.pbt_enabled = True
    config.__post_init__()

    print(
        f"Starting random-resampling baseline: "
        f"{config.pbt_population_size} members, "
        f"{config.pbt_meta_iterations} meta-iterations, "
        f"resample every {config.pbt_exploit_interval} steps, "
        f"horizon {config.pbt_world_horizon}"
    )

    run = (
        wandb.init(
            project="adversarial-nca",
            config=config.__dict__,
            tags=["random-resampling", "baseline", "no-selection"],
        )
        if config.wandb
        else None
    )

    output_dir = train_random_resampling(config, run)

    if run:
        wandb.finish()

    print(
        f"Random-resampling baseline completed! Outputs saved to {output_dir}"
    )


if __name__ == "__main__":
    main()
