# PBT-NCA

**Website:** [arberzela.github.io/pbt-nca](https://arberzela.github.io/pbt-nca/)

PBT-NCA is a meta-optimization framework for **Petri Dish Neural Cellular Automata (PD-NCAs)** that turns population-based training into an open-ended discovery process. Instead of optimizing a stationary objective, it applies **novelty-driven selection pressure at two timescales** so that populations of competitive worlds keep producing new behaviors and structures over long horizons.

## Method in brief

At each meta-iteration, PBT-NCA:

1. **Rolls out and scores** a population of worlds in which multiple NCA agents compete on a shared grid.
2. **Updates a FIFO archive** of behavioral descriptors and rewards novelty relative to past discoveries.
3. **Adds visual diversity** using frozen DINOv2 features to favor new morphologies, not just handcrafted statistics.
4. **Performs exploit–explore replacement**, where weak worlds are replaced by mutated/crossed-over copies of stronger ones.

This produces emergent phenomena such as gliders, shooters, amoebas, colonies, and other lifelike dynamics without manually specifying target behaviors.

## Explore the project

- Live website: [arberzela.github.io/pbt-nca](https://arberzela.github.io/pbt-nca/)
- Website source: [website/index.html](website/index.html)

## Selected emergent dynamics

- **Amoeba**

  ![Amoeba](website/gifs/amoeba.gif)

- **Glider**

  ![Glider](website/gifs/glider.gif)

- **Ant Colony**

  ![Ant Colony](website/gifs/ants.gif)

- **Spaceship / Motherboard**

  ![Spaceship / Motherboard](website/gifs/spaceship_motherboard.gif)

## Citation

If you use this project, please cite:

```bibtex
@inproceedings{berdica2026pbtnca,
  title     = {Evolving Many Worlds: Towards Open-Ended Discovery in Petri Dish NCA via Population-Based Training},
  author    = {Berdica, Uljad and Foerster, Jakob and Hutter, Frank and Zela, Arber},
  year      = {2026}
}
```
