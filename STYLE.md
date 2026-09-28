# Code Style & Readability Policy

**Project rule: no legacy or backward compatibility.** The codebase is unreleased, so breaking
changes are free: keep one live implementation per mechanism, and when a format changes update
every producer and consumer together. No format-version fields, no compatibility shims, no
legacy code paths. `git` is the archive.

The conventions every Python file in `src/` and `space/` follows. Agreed
with the maintainer; every new edit — human or agent — should already match it.

**How this file is organized.** Sections 1–7 are the _distinctive_ conventions worth reading in
full — including §7, the minimalism and repo-organization rules. Section 8 is a compact list of
_house rules_. Section 9 records what the **tooling owns** (`ruff`, `basedpyright`, configured
in `pyproject.toml`). Section 10 is project-wide **PyTorch conventions** for an inference-only
library.

When in doubt, match the surrounding code.

---

## 1. Typing — semantic jaxtyping aliases, never bare `Tensor`

Every tensor parameter, return value, and meaningful local is annotated with a jaxtyping alias that
encodes **both shape and nature**. A bare `torch.Tensor` annotation is a smell — replace it with the
alias that says what the tensor _is_.

- **Aliases live in `src/inspatio_world/types.py`.**
- **Two aliases may share a shape but mean different things.** Same shape, different nature →
  different alias (`VideoLatents` vs `VAEFeatures`, `KVCache` vs `CrossAttentionCache`,
  `RenderedFrames` in [-1, 1] vs `ClipArray` uint8 at the file boundary).
- **Granularity is hybrid: distinguish by nature at API boundaries.** Public function signatures use
  the role-distinct alias where the nature matters; inside a function body a single base alias is
  fine.
- **Aliases are documentation, not runtime checks.** We do _not_ decorate with
  `@jaxtyped(typechecker=beartype)`; annotations carry zero runtime cost. `basedpyright` checks them
  statically and the test suite catches real shape bugs.

```python
# types.py
type VideoLatents = Float[Tensor, "B C G H W"]  # normalized latent video (C = 48)
type VAEFeatures = Float[Tensor, "B C G H W"]  # internal causal VAE feature maps
type ClipJointsCamera = Float[np.ndarray, "F S J 3"]  # numpy at the parquet boundary, same rule
```

Bare `Tensor` is acceptable only for a genuinely role-agnostic tensor (rare); prefer an alias.

**Axis glossary.** `types.py` opens with a comment block defining the single-letter axis symbols
its aliases use, so shapes stay legible across files. Need a new axis? Add it to the glossary
first. STYLE.md does not duplicate the table — it lives next to the aliases.

```python
# types.py
# --- axis glossary ---
# B batch (windows)   H height (pixels or latent)   N tokens / sequence length
# C channels          W width  (pixels or latent)   D model / head / embed dim
# F frames (video)    G latent frames               L text tokens
# S hands per frame (2: left, right)                J hand joints (21, OpenPose order)
# ...

type Frames = Float[Tensor, "B 3 F H W"]  # RGB in [0, 1], the latent encoders' input
type JointsCamera = Float[Tensor, "B F S J 3"]  # metres in the camera frame
type Joints2D = Float[Tensor, "B F S J 2"]  # normalized image coordinates in [0, 1]
```

Keep using modern typing: `Literal`, `Protocol`, `type` aliases, PEP 695 generics,
`Self`/`override`. Prefer `Literal[...]` for small closed sets of choices
(`DecoderKind = Literal["wan", "taehv"]`); reserve `Enum` for when you genuinely need iteration,
methods, or stable wire values.

**Generics and `Any`.** A helper that is generic over a type uses PEP 695 syntax
(`def prefetch[T](items: Iterable[T], depth: int) -> Iterator[T]`,
`class Wrapper[M: nn.Module]`), never `TypeVar` boilerplate. Avoid `Any` and `dict[str, Any]`
wherever a frozen dataclass, a `TypedDict` or a concrete type is possible; `Any` is for genuinely
untyped external boundaries (an unpickled object, a JSON blob whose schema we do not own).

## 2. einops — named axes, always

Use `rearrange` / `reduce` / `repeat` / `einops.einsum` with **named axes** for every reshape,
reduction, broadcast, and contraction. Do **not** use `.reshape` / `.view` / `.transpose` /
`.flatten` / `.permute` or `torch.einsum`; the named pattern is the readable, self-checking form,
and einops raises on a shape mismatch for free. (Operations einops cannot express — indexing,
masking, padding — are obviously outside this rule.)

The pattern's axis names mirror the alias dimensions.

**Casing convention (kept deliberately distinct across the three layers):**

- jaxtyping alias dims: **UPPERCASE** — `Float[Tensor, "B N D"]`
- einops patterns: **lowercase** (idiomatic einops) — `"b (g h w) d -> b d g h w"`
- inline shape comments: **UPPERCASE** — `# (B, G*H*W, D) -> (B, D, G, H, W)`

```python
def fold(tokens: Tokens, grid: LatentGrid) -> TapFeatures:
    frames, height, width = grid
    # (B, G*H*W, D) -> (B, D, G, H, W)
    return rearrange(tokens, "b (g h w) d -> b d g h w", g=frames, h=height, w=width)
```

## 3. Inline comments — moderate: step + why

- A short comment labelling each logical **step**, plus a **why** on anything non-obvious
  (math identities, broadcasting tricks, numerical gotchas, coordinate-frame conventions, a
  deliberate deviation from upstream). Not line-by-line.
- Inline shape comments `# (B, C, G, H, W)` only at the gnarly transforms the alias/einops can't
  already convey.
- Sentence-case; no trailing period unless the comment is multiple sentences.
- Markers allowed: `# NOTE: …`, `# TODO(owner): …`. Avoid `# HACK` — fix it or open a TODO.

## 4. Vertical spacing — blank-line grouped steps

Separate logical steps within a function with a single blank line so the body reads as visual
paragraphs. No `# --- banner ---` dividers unless a function is genuinely long and multi-stage.

## 5. Docstrings — tiered

- **Module + public class/function:** a one-line **imperative** summary, a short why/how paragraph,
  and a paper reference where relevant on its own line (`Ref: InSpatio-World arXiv:2604.07209`).
  Module docstrings say what the file is for and how it fits the pipeline — useful to a fresh
  reader, not a changelog of past work.
- **Small private helpers:** one line, or none when the name and signature already say everything.
- Do not add Google-style `Args:`/`Returns:` boilerplate unless a parameter is genuinely
  non-obvious.
- **One blank line between the docstring and the first line of code.**

## 6. `__init__.py`

Re-exports only — `from .x import Y` plus `__all__`. No class/function definitions, constants, or
logic live in an `__init__.py`; that belongs in a named module.

## 7. Minimalism & repo organization

This repo is a port whose reason to exist is a **single clean inference path** for one released
model. The upstream repositories show what accretion looks like: every investigation became a
permanent module and config flags select between half-maintained variants. These rules keep
that from happening here.

### Top-level layout

```
src/inspatio_world/  the package — the inference path, and nothing else (see §8 Package layout)
docs/              canonical formats: pose_format.md, lerobot.md, architecture.md
scripts/           one-off tools: checkpoint conversion, example preparation, the model card
space/             the Hugging Face ZeroGPU Space (app, AoT compilation, the viewer component)
```

### 100% live code

Every module in `src/` is reachable from a current entrypoint (a CLI, the estimator, a test that
guards real behaviour). There are **no dead branches**: no "kept just in case" files, no
commented-out alternatives, no `if use_old_head:` fossils, no unused upstream tensors kept "for
completeness". Deleting code is safe, cheap, and reversible, so delete aggressively. A module
nothing imports is a bug in the tree, not a resource.

### Investigations are guests, not residents

Parity checks, speed measurements and alternative-implementation trials live in scratch space
or on a branch — never as new modules inside `src/`. An
investigation ends in exactly one of two ways: **adopted** — it _replaces_ the code it improves
on (the old path is deleted in the same change) and the memo records the decision; or
**abandoned** — its code is deleted and the memo records why. "Merged but switchable" is not an
outcome. If we are not confident enough to delete the old path, the investigation is not done.

### One path per subsystem — deletion over flags

Each subsystem has exactly one implementation: one DiT, one VAE stream, one splat renderer, one
sampler, one scene format, one camera rig. Changing an approach means
**replacing** it, not adding a selector next to it. A config field must change a _quantity_
(encode width, window length, windows per batch); the moment a field selects between _code
paths_, stop and pick one path. The exceptions are **weight choices** behind one shared
interface, never a branch in the pipeline: `decoder` (`wan` / `taehv`, two `FrameDecoder`s) and
`dit_precision` (`bf16` / `fp8`, the same modules with quantized linear weights).

### File & module budget

A module is one idea, readable top to bottom in one sitting:

- **~300 lines is the soft ceiling** for a module; a file drifting past it is a signal to split by
  _concept_ or — more often — to simplify. Genuinely irreducible cores (the DiT, the VAE encoder,
  the projector) may exceed it, but that is an exception to justify, not a norm.
- **Small fixed package shape** (see §8 _Package layout_): a reader should predict the file list
  before opening the package. New files need a new _concept_, not a new _variant_.
- **Flat over deep.** The sub-packages in §8 _Package layout_ are the whole tree; modules are
  one level below them and never nest further.

### Helpers

A function used in one file stays private there (`_name`). A helper reused across files moves to
the closest shared module — the module that owns the concept, or `utils/` for genuinely generic
plumbing — and is imported from there; it is never duplicated.

### Module dependency direction

Modules import **one way**: `types` and `config` are foundations that import nothing else in the
package; `wan/`, `depth/`, `render/` and `utils/` build on them; `scene.py`, `sampler.py`,
`session.py`, `controls.py`, `world.py` and `data/` compose them; `scripts/` sits on top and is
imported by nothing. No import cycles,
and no module reaches into a sibling for a private helper — if two modules need it, it is public
in the module that owns the concept.

### One canonical form per boundary; the mess lives in the glue

Every external world we touch arrives in as many shapes as it has sources: upstream checkpoints
(a PEFT state dict, a Diffusers VAE, a `taew2_2` file), camera calibrations (inline `fx,fy,cx,cy`,
a `.camera.parquet` sidecar, a 3x3 `.npy`, a LeRobot feature with 4, 6 or 9 values), video
containers, dataset layouts. The rule that keeps that variety out of `src/`: **pick one canonical
internal form, convert to it once at the boundary, and let no module behind the boundary know a
second form exists.**

- **Adopt and extend, don't invent.** The canonical form is a real, documented format plus our
  declared additions: a parquet pose table with a declared schema and key-value metadata, a
  LeRobot sidecar in LeRobot's own layout, safetensors with a `config.json` per component.
- **Specify it, then test it.** The canonical form is written down in `docs/`, and conformance is
  a _check_ in the module that owns the boundary. Joint order (OpenPose 21), camera frame
  (OpenCV), handedness (slot 0 left, 1 right) and units (metres) are specified there once.
- **Conversion is one function at the boundary.** `data/calibration.py` is the only reader of
  calibration files; `data/lerobot.py:episode_intrinsics` the only reader of a LeRobot camera
  feature; `inference.py:load_estimator` the only reader of the weights repository. A `if source == ...`
  inside the model means the glue leaked; fix the glue, not the caller. Ugly is allowed in glue
  — long, defensive, full of source-specific comments — and nowhere else.
- **Same shape for the model boundary.** The Wan2.2 trunk, VAE encoder, TAEHV encoder and umT5
  encoder are reimplemented from scratch in `wan/` so we own every line; no diffusers /
  transformers / peft / smplx at runtime. Upstream code is never imported — port what is needed
  into `src/` with attribution (§8).

## 8. House rules

Short rules. Each is one decision; the example shows the agreed form.

### Design principles

The repo is deliberately minimal — **one path per subsystem, no selectors** (§7). Four principles
keep it that way:

- **Limit generalization; allow duplication.** Don't add a config knob, abstraction, or branch for
  a use case you don't have yet — hard-code until something genuinely must vary.
- **Composition over inheritance.** Build from small, self-contained modules; avoid implementation
  inheritance (deep base classes). Use `ABC` only for genuine shared implementation.
- **Defer value checks to the point of use.** Validate a value in the lowest module that actually
  uses it, not in high-level callers — fewer `raise`s, and the check sits next to the assumption it
  guards (pairs with _Shape & error validation_ below).
- **Protocols are behavioral contracts.** Use a `Protocol` only for a method-bearing contract with
  **two or more implementers or a real external boundary** (e.g. `LatentEncoder`, implemented by
  the Wan VAE and TAEHV). Never a single-implementer or attribute-only Protocol — return the
  concrete frozen dataclass instead.

### Package layout

The package has a fixed shape so a reader always knows where to look:

```
src/inspatio_world/
  types.py        jaxtyping aliases + axis glossary
  config.py       frozen dataclass config (WorldConfig)
  scene.py        the Scene: source views + depth + cameras, the only input form
  controls.py     CameraRig: per-frame actions -> target poses
  sampler.py      flow schedule + the few-step denoising of one block
  session.py      the streaming generation: render -> encode -> denoise -> decode, per block
  world.py        WorldModel: the only reader of the weights repository
  wan/            the Wan2.1 stack: dit.py (causal DiT), vae.py, taehv.py, text_encoder.py
  depth/          Depth-Anything-3 nested: backbone, heads, camera, estimator, preprocess
  render/         depth forward-splatting (splat.py)
  data/           video / image files, scene folders, trajectories
  scripts/        tyro entrypoints (prepare / generate) and the `inspatio-world` command
  utils/          small shared plumbing (hub.py) -- not a dumping ground
  __init__.py     re-exports only (every sub-package likewise)
```

### Naming

Descriptive names everywhere — `latent_frames`, `joint_heatmaps`, `camera_intrinsics`,
`window_latents` — _not_ terse math letters. Three carve-outs, and only these:

1. **Shape unpacking:** `b, c, g, h, w = latents.shape`.
2. **Axis-size ints** matching the glossary: `H`, `W`, `G`, `J`.
3. **Universally-standard math symbols** when the paper reference is right there: `t` for the
   diffusion timestep, `eps` for a norm epsilon, `K` for a camera intrinsics matrix.

Modules/files `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.

### Configuration

Settings live in `@dataclass(frozen=True, slots=True)` configs (stdlib only — immutable is
reproducible). Nest per subsystem; use `field(default_factory=...)` for nested config defaults.
CLI parsing layers on top via `tyro`: each script owns one config dataclass whose field
docstrings become `--help`, and `cli.py` unions the four into subcommands. Config fields set
_quantities_ or checkpoint choices, not code paths (§7).

### Constants & magic numbers

Tunable quantities → a config field. Fixed domain/math constants → module-level `UPPER_CASE`
(`NUM_JOINTS = 21`, `LATENT_CHANNELS = 48`). No bare numeric literals in logic.

### Shape & error validation

Trust the annotations and einops — they document intent and raise on mismatch. Do **not** add
defensive `assert`s at every boundary. Add a targeted `assert` only at a genuinely ambiguous edge,
always with a message.

```python
assert num_frames == episode.length, (
    f"episode {episode.index}: {num_frames} predicted frames for {episode.length} rows"
)  # only at an ambiguous edge
```

### Imports

`ruff` groups stdlib / third-party / first-party (isort). Use **relative** imports inside the
package (`from ..types import Frames`); `scripts/` and `space/` import the
package absolutely (`from inspatio_world.data import load_scene`), never a private
`_name`. Every module opens with `from __future__ import annotations` (see §9).

### Device & dtype

Device-agnostic, always. New tensors inherit `device`/`dtype` from an input tensor; modules take a
`device`/`dtype` argument and create their params in `__init__`. Never call `.cuda()`.

### `nn.Module` conventions

Implement `forward` with jaxtyping annotations; call a module as `module(x)`, never
`module.forward(x)`. Register every non-learned tensor (RoPE tables, latent statistics, the
latent statistics) with `register_buffer` so it follows `.to(device)`;
`persistent=False` when the checkpoint must not carry it.

### Autograd context

`@torch.inference_mode()` on every entry point that runs the model (`Session.step`,
`DepthEstimator.estimate`, the text encoder). `torch.no_grad()` only for in-place parameter surgery (copying weights,
filling adapters) where inference-mode tensors would be the wrong kind.

### Logging & output

No `print()` in library code under `src/inspatio_world/` outside `scripts/`. Use `logging` for
human messages. `print()` and `rich` progress bars are allowed only in `scripts/` entrypoints
for CLI UX.

### String formatting

f-strings for all string building. Inside `logging` calls use `%`-style lazy args
(`logger.warning("%s: %s is non-zero", root, column)`) so the format is skipped when the level
is disabled — never `logger.info(f"...")`.

### Source attribution

Keep a one-line source comment at the top of any file ported or adapted from external code or a
paper (`# Adapted from https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/vae2_2.py`, or
`Ref: <paper> arXiv:…` in the module docstring). Attribution names the upstream GitHub
repository — `https://github.com/ggxxii/ACE-Ego-Hand`, `https://github.com/Wan-Video/Wan2.2`,
`https://github.com/aigc-apps/VideoX-Fun`, `https://github.com/huggingface/diffusers`,
`https://github.com/madebyollin/taehv` — with the file path when one file was the source, so
provenance and licensing stay clear.

---

## 9. Tooling (delegated)

Formatting, import order, modern-syntax upgrades, and type checking are enforced by tooling, not by
review. Config lives in `pyproject.toml`. Only run the lint and format when you are done with your main
task, at the very end.

- **`ruff` lint** — rule set `E, F, I, UP, B, SIM, C4, C90, RUF, PIE, PERF, NPY, FURB, PTH, RET,
  ANN` (jaxtyping's `F722`/`F821` ignored; `ANN002`/`ANN003`/`ANN401` ignored so
  `*args`/`**kwargs`/`Any` aren't flagged; `E501` ignored because the formatter owns line
  length). It enforces **complete signature annotations** (`ANN`) and a **function-complexity
  cap** (`C90`, max 12), plus modern syntax, likely bugs, simplifications, `pathlib` and modern
  NumPy.
- **`py.typed`** — the package ships a PEP 561 `py.typed` marker so downstream type-checkers
  consume our inline annotations.
- **`ruff format`** — the autoformatter, `line-length = 100`. Do not hand-format.
- **`basedpyright`** — static type checking over `src/` and `space/`;
  framework-noise reports are disabled in `pyproject.toml` so it focuses on real project mistakes.
- **`from __future__ import annotations`** — mandatory at the top of every module (enforced via
  `ruff` isort `required-imports`). All annotations become lazy strings: jaxtyping shape strings
  are never evaluated and type-only imports never cause cycles.

---

## 10. PyTorch conventions

Project-wide PyTorch idioms for an inference-only library. (Autograd context, device/dtype,
`nn.Module` and logging rules live in §8 and are not repeated here.)

**Precision**

- The numerics follow the upstream inference path and are stated once in the module docstring
  of each component. Upstream casts the whole world model to bf16: DiT (residual stream, adaLN,
  layer-norm outputs), VAE and umT5 run in bf16 with float32 RMS statistics; rotary rotation in
  float32; the sampler's x0 conversion in float64. DA3 reproduces its bf16 autocast explicitly
  (bf16 ViT linears, float32 norms / heads / alignment). No `torch.autocast` islands.
- Geometry (unprojection, splatting, camera rig) is float32 (rig integration float64).

**Attention**

- Use `torch.nn.functional.scaled_dot_product_attention`; never hand-roll `softmax(QKᵀ/√d)·V`.
  Each attention module is exactly one such call (the DiT's goes through `wan.dit.attention`).

**Weights**

- Every weight-bearing component (DiT, VAE, TAEHV decoder, umT5 encoder, depth estimator) is a
  `HubModule` with a `config.json` + `model.safetensors` pair in its own folder of
  `blanchon/inspatio-world-v1.5`; `strict=True` on every load. `world.py` is the only reader.

**Streaming & hot loop**

- Generation is a stream of blocks (`Session.step`); every causal state (VAE caches, previous
  latents, noise generator) lives in the session, so a video equals the concatenation of its
  blocks. No CPU↔GPU syncs inside a block; results are moved to the CPU by the caller.

**Compilation**

- Local: `WorldConfig.compile` regionally compiles the DiT block and the VAE encoder stages with
  static shapes (`mode="max-autotune-no-cudagraphs"`), the same code path traced or eager.
- ZeroGPU: `space/aoti.py` AoT-compiles one DiT block (steady-state prefill and denoise) and
  shares the package across the 30 blocks with per-block weights; packages are cached on the Hub
  per torch version / GPU / precision / DiT source. An AoT graph of one VAE encoder chunk (cache
  tensors in and out) was measured to corrupt the returned cache tails and was dropped.

**Measured and rejected (do not reintroduce without a new measurement)**

- Batching the render and source encoder streams into one batch-2 call (no gain: the VAE convs
  already saturate the GPU); caching the source latents of a static image (the causal cache
  never converges to a fixed point, so it would change the output); compiling the Wan VAE
  decoder (no gain); FA3 (Hopper-only, and ZeroGPU now runs Blackwell GPUs).
