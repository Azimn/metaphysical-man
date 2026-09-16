# Metaphysical Man

**MADMAN** stands for **Metaphysical Artificial Dynamics for Metabolic Autopoiesis and Noesis**.

This repository is a deliberately strange research toy. It does not start with an agent, a memory system, an emotion model, a planner, or a self. It starts with an artificial substrate whose local transformations cost scarce resources. Patterns have to afford their own continuation. Repeated activity can scar the substrate, residue can make repeated activity expensive, bound reserves can make dormant patterns easier to reconstruct, and future resource gradients can pull present activity without an explicit goal object.

The experiment is not "can we prove consciousness?" The experiment is "what kind of mind-like behavior appears if computation has to live inside an artificial metabolism?"

## The vocabulary is intentionally ridiculous

The names are playful, but the mechanics are exact. The simulation substrate is the **Crucible**. A transient local occurrence is a **Spark**. Immediately usable computational resource is **Aether**. Locally bound reserve is **Ichor**. Structural plasticity capacity is **Azoth**. Inhibitory waste is **Ash**. Durable changes to connectivity are **Scars**. A reconstructed pattern is an **Echo**. A recurrent resource-capture attractor is a **Haunting**. A temporary integrated cluster is a **Constellation**. The locally propagated prospective resource field is **Morrow**. Observer-only instrumentation is the **GhostHunter**. The intentionally disadvantaged verbal reporter is the **Medium**.

These terms are aliases for operational mechanisms, not claims about physics, spirituality, or human consciousness.

## What is implemented

The current runtime contains the full first-pass architecture rather than a staged subset. There is no central allocator and no `Agent` object. Resource moves through a directed substrate. Sparks are assimilated only when local Aether can pay for them. Activity can bind Aether into Ichor, leave Ash, strengthen Scars using Azoth, propagate only through local edges, and fail before becoming a complete event if the local economy cannot support it. Low activity can slowly rebuild Azoth from Aether. Ash dissipates. Ichor decays. Scars persist much longer than activation.

Morrow supplies a deliberately speculative teleological ingredient. Resource sources create a locally relaxed prospective field. Neighboring sites inherit some of that field through existing connections. Activity propagation is slightly biased toward regions with stronger Morrow. Nothing stores a goal or knows a destination, but resource-rich possible continuations can exert present influence.

The Medium can read only designated report surfaces. It cannot inspect Scars, Aether, Ichor, Azoth, Ash, causal ancestry, or the observer trace. It therefore has to report consequences rather than hidden mechanisms. The GhostHunter has the opposite role: it is third-person instrumentation and may inspect the entire substrate, but nothing inside the Crucible can query it.

The runtime also supports deterministic cloning and experimental fusion of compatible histories. This lets us create branch, divergence, recombination, and continuity experiments without granting the simulated substrate any privileged answer about which copy is the "real" one.

## Run it

Python 3.11 or newer is required. The core runtime uses only the Python standard library. Install the project in editable mode with `pip install -e .`. Install the test dependency with `pip install -e ".[dev]"`.

Run the whole experiment battery with:

```bash
madman all --seed 7
```

Machine-readable output is available with:

```bash
madman all --seed 7 --json
```

Individual experiments are named `echo`, `history`, `haunting`, `attention`, `fork`, `morrow`, and `speech`.

## What would count as interesting

The code is allowed to fail spectacularly. A positive result is not "the program seems conscious." A positive result is a traceable phenomenon that was not explicitly represented as the thing it resembles. If repeated experience changes later behavior after activation has disappeared, that is memory-like. If resource concentration causes selective elaboration without an attention controller, that is attention-like. If a recurrent configuration repeatedly captures resources and later reconstitutes itself without a reminder object, that is concern-like or rumination-like. If two identical histories diverge into different resource-flow geometries and later react differently to the same perturbation, that is individuality-like.

The observer should always be able to explain the mechanism afterward. The system itself should not automatically have that privilege.

## Status

This is the "build the whole monster and see whether it twitches" version. If something genuinely interesting appears, the next move is to reduce it, ablate it, and reproduce it under controlled conditions rather than immediately adding more machinery.
