# run_slot — second slot for image runs (S308, design + implementation, not run on the game)

Goal: two image runs (frame dumps, flood checks) at the same time. Tempo (frames/s, times) is never
measured in parallel.

## Layout
- `prepare --variant _Build/variants/<v> --config <cfg.json>` copies the variant directory (binary,
  libMoltenVK, `_ShaderCache`, `_PipelineCache`, `_SaveData`) to `<v>-slot2` and writes `<cfg>-slot2.json`
  (label `<label>-slot2`, emulator in the copy). Copy at rest: no process may be using the variant.
  `--refresh` recopies (new binary or cache wanted). Refuses below 15 GB free (R-BEZ-9). Cost: a variant with
  warm caches is a few GB; delete the slot when done.
- Each run has its own `KYTY_FRAME_LOG` (kbench puts it in the run's `--out`) and its own
  `KYTY_FRAME_DUMP_DIR` (`run --dump-dir`). Use a different `--out` than the main run.
- The slot is announced by `kbench-results/GPU_SHARE-slot2` (`slot2 <pid> <date> image`), not by `GPU_LOCK`.

## Rule (enforced by `check` / `run`)
1. The slot starts only while `GPU_LOCK` exists, its holder PID is alive and its line ends with the word
   `image` (`rt-agent <pid> <date> image`): the main run declares itself an image run. A tempo lock has no
   such word: no slot.
2. One slot at a time (`GPU_SHARE-slot2` absent).
3. A tempo run takes `GPU_LOCK` only when `GPU_SHARE-slot2` is absent (the game agent checks it next to
   `pgrep -fl kyty_emulator`). Results of a run that overlapped a slot are image-only.
4. `run` removes `GPU_SHARE-slot2` when kbench ends; `release` removes a stale one (only if it is ours).

Usage: `tools/run_slot.py check` (exit 0 = may start) then
`tools/run_slot.py run --config <cfg>-slot2.json --dump-dir <dir> -- --scenario <scen> --runs 1 --warmup 0 --allow-busy-gpu --allow-locked --out <out>`.

## Calibration to do by the game agent (not done here: no game in S308)
Four runs in parallel are not possible with one slot, so calibrate in two steps, N–N each (same variant, same config):
1. Baseline: 2 sequential runs N–N (A, B) of the image scenario at the same flips; `imgcheck` A–B and the
   flood detector (G−B ≥ 100 at 6400/6600, `tools/gate.py`) = reference spread.
2. Parallel: the main run (C) and the slot run (D) at the same time, N–N, same flips; `imgcheck` C–D, plus
   C–A and D–A. Accept the slot when C–D and D–A are within the spread of A–B (mean |Δ| ≤ 2, frac(>16) ≤ 1 %) and
   the flood verdict matches the sequential one in all 4 runs; also compare last flip and pipeline compilations
   (R-POM-12) — a slot with a copied warm cache must show the same counts as the sequential runs.
3. If the parallel pair differs more than A–B: the slot is not valid for that scenario (timing-dependent
   scenes, e.g. the tunnel); record which, and keep slot runs for static frames only.
4. Report the wall-clock saving (two image answers in one run time) and the slowdown of the main run's
   flips/s (for information only; a main tempo run never overlaps).

## Tests
`python3 -I tools/test_run_slot.py` (11 tests), mutants: `tools/mutants_run_slot.sh` (each must exit non-zero).
