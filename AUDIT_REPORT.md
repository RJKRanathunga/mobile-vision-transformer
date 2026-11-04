# MVT Functional Equivalence Audit

## Scope and reference

This audit compares the implementation in this repository with
[`goutamyg/MVT`](https://github.com/goutamyg/MVT) at the pinned, unmodified
commit:

`880e74b1d1f720cb51d78b68655a4ffef94ed954`

The reference checkout used for testing was kept outside this repository.
The comparison traced the effective 256/128 GOT-10k experiment through input
cropping, MobileViT-S feature extraction, template/search interaction, neck,
pixel-wise correlation, center prediction, loss targets, optimizer setup,
checkpoint loading, and stateful inference.

## 1. Overall equivalence assessment

**Partially equivalent overall.** After the corrections documented below, the
implemented model, crop preprocessing, output decoding, and stateful inference
math are functionally equivalent to the corresponding pinned-upstream 256/128
MVT path. With an actual upstream state dictionary mapped into the local model,
all four end-to-end CUDA outputs were bit-identical. Deterministic backward
testing of the prediction head was also bit-identical.

The repository is not a complete replacement for the upstream training and
benchmark system. It has processing, target/loss, optimizer/scheduler, model,
checkpoint, and tracker primitives, but it still lacks the upstream dataset
sampler, augmentation composition, epoch runner, distributed trainer, and
benchmark launch/evaluation tooling. No trained MVT checkpoint was present, so
trained tracking accuracy and full benchmark metrics could not be verified.

## 2. Component-by-component comparison

| Component | Original implementation | Local implementation | Status | Evidence |
|---|---|---|---|---|
| Effective experiment geometry | `experiments/mobilevit_track/mobilevit_256_128x1_got10k_ep100_cosine_annealing.yaml`: 256 search, 128 template, stride 16 | `configs/*.yaml`, `MVT`, and `TrackerConfig` use the same geometry | Corrected | Config regression tests and 16x16/8x8 feature assertions |
| Image tensor conversion | RGB HWC uint8 to CHW float, scaled by 1/255; experiment mean 0 and std 1 | Same | Equivalent | Direct code trace and crop/model tests |
| Target crop, padding, resize | `lib/train/data/processing_utils.py::sample_target` | `mvt.data.processing_utils.sample_target` | Corrected | Three exact comparisons, including top-left and bottom-right boundary cases |
| Boundary attention mask | Float mask resized with OpenCV linear interpolation | Same after correction | Corrected | Exact array equality; prior uint8 resize differed at 68/33 boundary pixels |
| Bounding-box crop transform | xywh crop transform and resize factor | Same, including list/tuple boxes | Corrected | Exact crop comparison and tracker regression |
| MobileViT-S backbone | `lib/models/mobilevit_track/mobilevit.py` | `MVTBackbone` | Equivalent | 2,519,760 parameters; 153 parameter tensors and 75 buffers; mapped-weight CUDA outputs exactly equal |
| Patch unfold/fold and transformer | MobileViT blocks with direct template/search token concatenation | Same tensor layout, attention, residuals, FFN, and fold order | Equivalent | Intermediate backbone outputs exactly equal |
| Backbone initialization | Custom convolution, linear, and normalization initialization | Same after correction | Corrected | Initialization path matched by code and regression coverage |
| Template cache boundary | Upstream caches output through `layer_2` | Local caches the same stem boundary | Corrected | Cached and uncached full-model outputs exactly equal |
| Neck | Separate template/search batch normalization | `MVTNeck` | Equivalent | Mapped-weight tensor outputs exactly equal |
| Pixel-wise correlation/fusor | `PW_Corr_adj`, 64 kernels, 256 output channels | `FeatureFusor` | Equivalent | Mapped-weight tensor outputs exactly equal |
| Center prediction head | Center, size, and offset branches; argmax decode | `CenterPredictor` | Corrected | Mapped-weight outputs exactly equal; 3-D/4-D target map paths tested |
| Full model | MobileViT-S + neck + fusor + center head | `MVT` | Equivalent | 5,480,405 parameters, 334 state entries; all end-to-end output errors 0.0 |
| Hann response window | Symmetric 2-D Hann window applied before decode | Same | Corrected | Shape/symmetry and tracker tests |
| Stateful tracking | Crop, cached template, response window, decode, map back, clip, update | `MVTTracker` | Corrected | State transition regression and real GOT-10k GPU smoke test |
| Center heatmap | Upstream knowingly uses a bug-version radius denominator | Correct CornerNet denominator by default; explicit upstream-compatibility option | Different by Design | Literal compatibility mode is exactly equal to upstream |
| Focal, GIoU, L1 losses | Upstream actor losses and weights 2.0/5.0 | Corresponding local losses and weights | Equivalent for implemented path | Unit tests, exact identical-target checks, finite backward tests |
| Optimizer/scheduler | AdamW; backbone LR multiplier 0.1; cosine experiment | `build_optimizer_scheduler` | Corrected | Parameter grouping and LR/scheduler tests |
| Checkpoint format | Upstream checkpoint stores model under `net` with upstream names | Strict native loader plus shape/count-validated pinned-upstream conversion | Corrected | 334-tensor conversion test and real upstream-state full-model comparison |
| Dataset sampling and augmentations | Causal sampler; GOT-10k sampling; grayscale, horizontal flip, and color jitter composition | Processing primitives only | Missing | No local sampler/augmentation pipeline or dataloader builder exists |
| Training runner | Actor, loaders, epoch loop, validation, distributed support, checkpoint lifecycle | No complete runner | Missing | Static repository inventory |
| Benchmark/evaluation launchers | Dataset adapters and `tracking/test.py`/analysis tooling | Stateful tracker only | Missing | Static repository inventory |

## 3. Discrepancies found

### Genuine mismatches corrected

1. **Duplicate MobileNetV2 module registration — high compatibility impact.**
   Local inverted-residual submodules were registered both through a sequential
   block and through separate attributes. The forward math happened to reuse
   the same objects, but the state dictionary had 442 entries instead of the
   upstream 334 and could not safely consume upstream checkpoints. The local
   attributes are now properties, leaving one registered path per tensor.

2. **No complete stateful tracker — high functional impact.**
   The model could produce maps but the repository lacked upstream-equivalent
   template caching, Hann-window selection, coordinate mapping, clipping, and
   consecutive-frame state updates. `mvt.evaluation.tracker` now implements
   that inference path without the upstream constructor's unrelated file-write
   side effect and without hard-coding CUDA.

3. **Boundary-mask interpolation type — medium preprocessing impact.**
   The crop image and resize factor matched, but converting the mask to uint8
   before resize changed boundary values. Keeping it floating-point through
   OpenCV resize restores exact upstream output.

4. **List boxes rejected by crop processing — high tracker impact.**
   Tracker state is naturally a Python list, while the earlier crop utility
   assumed tensor-only arithmetic. The utility now accepts tensor, NumPy,
   list, and tuple xywh boxes.

5. **Ground-truth score-map rank — medium training impact.**
   Upstream accepts `[B,H,W]` and inserts the channel dimension before decode.
   The local head formerly expected only `[B,1,H,W]`. Both forms are now
   validated and supported.

6. **Backbone initialization — medium training impact.**
   The local backbone relied on framework defaults instead of upstream's
   convolution/linear/norm initialization. The explicit upstream scheme is now
   applied.

7. **Missing checkpoint interoperability — high usability impact.**
   A strict loader now handles native state dictionaries, `module.` prefixes,
   upstream `net` containers, and full-model or backbone-only pinned-upstream
   layouts. Count and every tensor shape are checked before order-based name
   conversion; drift fails loudly.

8. **Missing optimizer/scheduler construction — medium training impact.**
   AdamW groups, the 0.1 backbone multiplier, cosine annealing, and step
   scheduler support were added.

9. **Empty effective configuration and dependency metadata — medium
   reproducibility impact.** The three YAML files now record the exact 256/128
   experiment and inference values, and package/runtime/test dependencies are
   declared.

### Intentional difference retained

The upstream `lib/utils/heapmap_utils.py` explicitly labels its third Gaussian
radius equation as a bug-version and divides by `2` instead of `2*a3`. The
local default keeps the mathematically correct CornerNet equation, as requested
by the audit instructions. `upstream_bug_compatible=True` is available when
literal reproduction of upstream target maps or training is required, and its
output is tested against the pinned source.

### Missing functionality not silently treated as equivalent

The full dataset sampler/loader, augmentation composition, training actor and
epoch loop, distributed/AMP integration, validation lifecycle, dataset
adapters, and benchmark reporting remain absent. The configuration documents
these values but is not itself a replacement for those systems.

## 4. Corrections implemented

- Removed duplicate backbone state paths while preserving the public attribute
  accessors.
- Matched upstream backbone initialization and added explicit cached-template
  forward methods.
- Factored the model prediction path so raw-template training and
  cached-template inference share identical downstream computation.
- Corrected crop mask interpolation and generalized xywh input types.
- Added score-map shape validation and upstream-compatible 3-D handling.
- Added intended and literal-upstream heatmap modes.
- Added a strict native/upstream checkpoint loader pinned to the audited SHA.
- Added a device-aware stateful tracker with Hann windowing, box mapping,
  clipping, and state updates.
- Added AdamW optimizer groups and cosine/step scheduler construction.
- Populated model, training, and inference configs and dependency metadata.
- Added regression tests for every corrected path and direct pinned-upstream
  comparisons.

## 5. Verification results

### Automated suite

- Complete local and comparative suite: **50 passed**.
- The suite includes all pre-existing tests plus checkpoint, configuration,
  optimizer, tracker, boundary processing, head-rank, heatmap-mode, and pinned
  upstream equivalence regressions.
- `git diff --check`: passed; only Git line-ending conversion notices were
  emitted on Windows.

### Numerical comparisons

All comparisons used fixed seeds and evaluation mode unless the test concerned
training behavior.

| Comparison | Result |
|---|---:|
| Backbone search output, mapped weights, CUDA | max error 0.0; mean error 0.0 |
| Backbone template output, mapped weights, CUDA | max error 0.0; mean error 0.0 |
| Neck, fusor, and all head tensors | max error 0.0 |
| Full model `pred_boxes` against upstream | max error 0.0 |
| Full model `score_map` against upstream | max error 0.0 |
| Full model `size_map` against upstream | max error 0.0 |
| Full model `offset_map` against upstream | max error 0.0 |
| Cached-template versus raw-template full local forward | max error 0.0 for all outputs |
| Deterministic CUDA head input gradients | max error 0.0 |
| Deterministic CUDA head parameter gradients | max error 0.0 |
| Crop image, resize factor, and boundary mask | exact on all three test cases |
| Literal upstream heatmap compatibility mode | exact |

A first training-mode CUDA gradient probe, before deterministic algorithms were
enabled, showed maximum differences of approximately `2.50e-8` for input
gradients and `3.71e-6` for parameter gradients. Repeating with deterministic
kernels and evaluation-mode batch normalization reduced both to exactly zero,
identifying kernel reduction order—not different math—as the cause.

### Representative real input

The stateful local tracker initialized and processed two subsequent RGB frames
from `GOT-10k_Train_000001` on an NVIDIA GeForce RTX 4050 Laptop GPU. Template
caching, two search forwards, decoding, clipping, and state updates completed
with finite boxes. This is an integration smoke test only: random weights were
used because no trained checkpoint exists in `checkpoints`, so the resulting
boxes are not evidence of tracking accuracy.

## 6. Remaining differences and limitations

- **Training is incomplete as a system.** The mathematical primitives are
  present, but no end-to-end local training job can be claimed equivalent until
  sampling, transforms, loaders, actor/runner, validation, and checkpoint-save
  behavior are implemented and tested.
- **No trained checkpoint was available.** Therefore GOT-10k AO/SR metrics,
  exact trained sequence trajectories, and equivalence of learned accuracy are
  unverified.
- **No benchmark launcher/dataset adapter layer is present.** `MVTTracker`
  provides the core per-sequence state machine, not the upstream evaluation
  framework.
- **Heatmap default intentionally fixes an upstream bug.** Use the compatibility
  flag only when exact reproduction of upstream training labels is more
  important than the intended equation.
- **API structure differs by design.** The local full-model positional order is
  `(search, template)` instead of upstream's internal conventions, and the
  local tracker is device-aware. These differences do not alter verified
  tensor computation.

## 7. Final conclusion

The local **model and core inference implementation are verified equivalent**
to the pinned upstream 256/128 MVT computation after correction: preprocessing,
backbone intermediates, fusion, prediction tensors, cached-template behavior,
decoding, and deterministic gradients all have direct experimental evidence,
with the principal tensor comparisons matching exactly.

The repository as a whole remains **partially equivalent**, not fully
equivalent, because upstream's complete training and benchmark ecosystems are
not implemented and trained accuracy could not be measured without a
checkpoint. Those omissions are now explicit rather than being hidden behind a
model-only equivalence claim.
