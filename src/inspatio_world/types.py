"""Jaxtyping aliases and the axis glossary shared by every module in the package."""

from __future__ import annotations

import numpy as np
from jaxtyping import Bool, Float, Int, UInt8
from torch import Tensor

# --- axis glossary ---
# B batch             H height (pixels or latent)   N tokens / sequence length
# C channels          W width  (pixels or latent)   D model / head / embed dim
# F frames (video)    G latent frames               L text tokens
# V source views (1 image, 4 images, or one per video frame)
# A attention heads   T diffusion timesteps in the schedule

# --- video ---
type Frames = Float[Tensor, "B 3 F H W"]  # RGB in [-1, 1], the VAE's pixel space
type ClipArray = UInt8[np.ndarray, "F H W 3"]  # RGB frames at the numpy / file boundary
type ImageArray = UInt8[np.ndarray, "H W 3"]  # one RGB image at the numpy / file boundary

# --- latents ---
type VideoLatents = Float[Tensor, "B C G H W"]  # normalized Wan2.1 VAE latents (C = 16)
type VAEFeatures = Float[Tensor, "B C G H W"]  # internal causal VAE feature maps
type LatentStatistics = Float[Tensor, "1 C 1 1 1"]  # per-channel latent mean or inverse std
type MaskLatents = Float[Tensor, "B 4 G H W"]  # render validity in {-1, 1}, 4 frames per latent
type FrameFeatures = Float[Tensor, "BF C H W"]  # per-frame 2D feature maps (TAEHV)

# --- diffusion transformer ---
type Tokens = Float[Tensor, "B N D"]  # DiT token stream in (frame, height, width) raster order
type AttentionHeads = Float[Tensor, "B N A D"]  # projected query / key / value heads
type KVCache = Float[Tensor, "L2 B N A D"]  # per block, the (key, value) heads of the context
type TextContext = Float[Tensor, "B L D"]  # text embeddings the DiT cross-attends to
type CrossAttentionCache = Float[Tensor, "L2 B L A D"]  # per block, text (key, value) heads
type TokenIds = Int[Tensor, "B L"]  # tokenizer ids, right-padded with the pad id
type TokenMask = Bool[Tensor, "B L"]  # True on real tokens, False on padding
type RelativePositionBuckets = Int[Tensor, "L L"]  # T5 relative-position bucket indices
type AttentionMaskBias = Float[Tensor, "B 1 1 L"]  # additive padding mask for attention
type Timesteps = Float[Tensor, "B"]  # diffusion timestep per sample, on the 0..1000 scale
type TimeEmbedding = Float[Tensor, "B D"]  # embedded timestep before the adaLN projection
type Modulation = Float[Tensor, "B 6 D"]  # adaLN shift / scale / gate vectors of one block
type RotaryTable = Float[Tensor, "N D"]  # per-token cos or sin of the rotary angles
type LatentGrid = tuple[int, int, int]  # (frames, height, width) of the DiT token raster

# --- cameras and geometry (OpenCV convention: x right, y down, z forward) ---
type Intrinsics = Float[Tensor, "... 3 3"]  # pinhole K in pixels of the 832x480 frame
type Poses = Float[Tensor, "... 4 4"]  # world-to-camera rigid transforms (Tcw)
type DepthMaps = Float[Tensor, "V H W"]  # metric-ish z-depth per pixel, 0 where unknown
type SourceImages = Float[Tensor, "V 3 H W"]  # source RGB in [-1, 1]
type RenderedFrames = Float[Tensor, "F 3 H W"]  # splatted RGB in [-1, 1], -1 where unknown
type RenderMasks = Bool[Tensor, "F H W"]  # True where the splat covered the pixel
type PointsHomogeneous = Float[Tensor, "V H W 4"]  # camera-frame points of each source pixel
type PixelScores = Int[Tensor, "F V"]  # per target frame, pixels each source view covers
type IntrinsicsArray = Float[np.ndarray, "... 3 3"]
type PosesArray = Float[np.ndarray, "... 4 4"]
type DepthArray = Float[np.ndarray, "V H W"]

# --- depth estimator (DA3) ---
# O output pixels along one axis    T resampling taps per output pixel
# P pyramid levels / tapped layers  Q pose-encoding components (3 translation, 4 xyzw quaternion, 2 FoV)
type SourceViews = UInt8[Tensor, "V H W 3"]  # RGB views as given to the depth estimator
type ProcessedViews = Float[Tensor, "V 3 H W"]  # ImageNet-normalized RGB at the DA3 processing size
type ViewTokens = Float[Tensor, "V N D"]  # ViT stream per view: one special token, then patches
type PatchFeatures = Float[Tensor, "V N D"]  # patch tokens of one tapped ViT layer
type CameraTokens = Float[Tensor, "V D"]  # special token of the last tapped layer, per view
type TokenRotary = Float[Tensor, "V N D"]  # per-token cos or sin of the 2D rotary angles
type FeatureMaps = Float[Tensor, "V C H W"]  # dense head features at some pyramid scale
type ConfidenceMaps = Float[Tensor, "V H W"]  # DA3 depth confidence, >= 1
type SkyMaps = Float[Tensor, "V H W"]  # metric head sky score, sky where >= 0.3
type PoseEncoding = Float[Tensor, "V Q"]  # camera-to-world translation, xyzw quaternion, FoV y / x
type Quaternions = Float[Tensor, "V 4"]  # unit rotations, xyzw (scalar last)
type ResampleIndex = Int[Tensor, "O T"]  # source pixel of each tap
type ResampleWeights = Float[Tensor, "O T"]  # weight of each tap
