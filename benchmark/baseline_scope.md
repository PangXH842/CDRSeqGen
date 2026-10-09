# Baseline coverage for the framework paper

**Decision guidance, 8 October 2026; no new generator has been implemented or run.**
The current comparison contains scratch MLM/AR/S2S at two seeds each, one frozen
IgLM checkpoint and an unconditioned training-H3 empirical reference. That is
eight system/seed rows and two reference types, with one external learned model.

| Intended claim | Current coverage | Recommendation |
|---|---|---|
| Unified multi-method framework with empirical demonstrations | Three local methods, external adapter, empirical reference and historical ablations | Defensible minimum for a focused abstract/framework demonstration; complete interface/release documentation and final protocol |
| Broad benchmark of available infilling systems | Only one external learned model, one corpus family, two local seeds | Add a compatible second external generator and document task coverage before making this a central claim |
| Architecture superiority or state of the art | Unequal capacities/tuning, unresolved local-versus-IgLM intervals, unknown external exposure | Unsupported; baseline count alone cannot fix the experimental controls |

IgLM directly provides variable-length span infilling using preceding/following
sequence context, which makes it relevant to the current H3 task.
[Official repository](https://github.com/Graylab/IgLM).

**IgCraft is the first candidate for a bounded adapter feasibility check.** It
provides sequence inpainting and public weights using a Bayesian Flow Network,
adding a different generative mechanism. Its inference configuration defaults to
`fix_length: true`; the script controls pad revelation and the sampler using that
flag. The sampler also supports `false`. A matched length-free comparison must
mask every H3 slot, including padding, mask the complete light chain, remove all
structural conditioning, and verify that changes to hidden native/partner strings
do not affect generated outputs under a fixed seed. These compatibility checks
remain pending. Sources: [repository](https://github.com/mgreenig/IgCraft),
[configuration](https://github.com/mgreenig/IgCraft/blob/main/config/inpaint.yaml),
[inpainting script](https://github.com/mgreenig/IgCraft/blob/main/scripts/inpaint.py),
[sampler](https://github.com/mgreenig/IgCraft/blob/main/igcraft/model/model.py).

**AbBFN2 is another candidate if IgCraft's adaptation is unsuitable.** Its released
interface supports conditioning on arbitrary subsets of sequence regions and
metadata, with H3 sequence and H3 length as separate modes. It would need only
the observed heavy regions conditioned, with H3 length/sequence and native light
chain left hidden. This is a feasibility inference from its interface, not an
already verified adapter. [Official repository](https://github.com/instadeepai/AbBFN2).

Keep the same frozen framework IDs, external attempt count, failure accounting
and shared scorer. Report model-specific sampling controls, support limits and
compute (including internal particles/steps); numerically identical temperature
values do not imply equivalent sampling across model families. Do not supply
native length, light chain or coordinates to only one arm in the primary table.

For the 1 November abstract, retain the narrow framework contribution and current
exploratory evidence. For a fuller comparison, aim to add one external generator
if the adapter passes smoke/compatibility checks. In parallel, resume MLM methods
review and document the workflow, extension contract and reproducible examples.
More backbone/LR sweeps are lower priority. Freeze final cohort/settings only
after deciding which external systems belong in that final comparison.
