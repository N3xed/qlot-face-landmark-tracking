# QLOT ablation study plan

## Purpose and scope

This document defines a **minimal, pre-specified ablation matrix** for the final
QLOT configuration to support the paper.  The aim is to test the claimed
components with one controlled counterfactual per comparison, rather than
performing a factorial sweep.  All ablations start from scratch; fine-tuning the
final checkpoint is not an ablation because it would bias the result toward the
full model.

`QLOT-final` is the locked reference implementation: the HGNetV2/FPN feature
extractor, query-conditioned correlation kernels, local correlation lookup,
three refinement iterations, temporal coordinate and hidden-state carry, the
WMR mixer, the GRU-like gated state update, the covariance head, and the final
loss recipe.  Before launching the study, record the commit
hash, command line, data manifest hashes, parameter count, MACs, and exact loss
weights in every run directory.

The covariance gate is **off** for the reference and every primary ablation
(`gating_radius=0`).  It is disabled in the main training path and is not part
of C4; mixing a hand-designed test-time gate into this comparison would make it
unclear whether a result came from learned covariance or post-processing.

## Claims

| ID | Paper claim | Evidence required |
| --- | --- | --- |
| **C1** | QLOT's query-conditioned iterative tracking design attains competitive landmark accuracy with substantially lower model cost and improved temporal stability relative to direct coordinate-regression baselines. | Compare QLOT with published or re-evaluated direct coordinate-regression models under a matched evaluation protocol; report accuracy, temporal stability, parameters, and MACs/FLOPs. |
| **C2** | Carrying a recurrent hidden state across frames improves temporal stability beyond coordinate initialization alone. | Compare QLOT with an otherwise identical tracker that carries the preceding coordinates but resets its hidden state at every video frame. |
| **C3** | The GRU-like gated update with lightweight WMR communication performs at least as well as a standard GRUCell updater, a local-only updater, and a mean-only low-rank baseline. | Three update-operator controls: standard GRUCell, no cross-landmark communication, and no WMR dispersion descriptor. |
| **C4** | A full 2D covariance output combined with spatial and temporal GNLL losses improves jitter, and may improve accuracy, over a simple L2 coordinate loss. | Compare QLOT with a coordinate-only, L2-trained tracker. Addtionally, compare with to a checkpoint with disabled temporal losses. |
| **C5** | `PhaseModulatedPE` improves performance relative to standard geometric-progression Fourier coordinate encoding [2]. | Compare QLOT with a Fourier-only query encoder (`encode_query_points` from commit `1c7e77b06126a94b1692f11fff111f8803dc8687`) while keeping the downstream dimensional interfaces fixed. |

“Performs at least as well” means that the final model should not be worse than a
baseline outside the paired confidence interval on the primary metrics.  A tiny
unpaired numerical difference is not sufficient evidence for C3.

## Data protocol and fixed splits

Dataset construction uses random, code-defined splits rather than a versioned
manifest.  The split seed stream starts from `np.random.default_rng(42)` and
derives one seed per split.  The split sizes are:

| Dataset | Training data | Validation data used by the code | Test data exposed by `Datasets` |
| --- | --- | --- | --- |
| FaceSynthetics | `floor(0.95N)` images from `FaceSyntheticsSmall` | `floor((N - floor(0.95N)) / 2)` images | The remaining half of the original 5% remainder |
| WFLW | Exactly 7,000 images from the official `train` split, with an unstratified permutation | The remaining official training images (500 for the standard 7,500-image train split) | The untouched official `test` split |
| 300-W | `num_images - 300` images from the combined Helen train, AFW, and LFPW train directories | Exactly 300 randomly selected images from that combined training pool | Official common, challenging, indoor, and outdoor subsets |
| WFLW-V | Exactly 800 videos, with training clips of length 16 and a random step from 1 through 10 | 50 videos from the remaining videos, with native-frame step 1 and 120-frame clips | The remaining videos from that 200-video pool |

For the standard 1,000-video WFLW-V release, the last row is therefore 800
training, 50 validation, and 150 test videos.  The split is stratified over the
easy and hard video ID lists and is disjoint by video ID.  The first WFLW-V
split uses an independent validation RNG; the second split uses the next
derived split seed.  Paths, checksums, and source-video IDs are not persisted by
`Datasets`.

Training data remain the same mixture for every configuration: FaceSynthetics,
WFLW training images, WFLW-V training clips, and 300-W training images.  Keep
the existing synthetic clip generation, real-video clip sampling, augmentations,
batch mixture, image resolution, pretrained backbone, freeze schedule,
optimizer, learning-rate schedule, iteration curriculum, and masking/dropout
schedule identical.  Validation and test inputs have no stochastic augmentation,
no test-time augmentation, and no covariance gating.

## Primary metrics and checkpoint selection

### Accuracy: NME

For each frame, calculate NME as the mean Euclidean landmark error divided by
the geometric mean of the ground-truth landmark bounding-box width and height:

$$
d_t = \sqrt{(\max_l p_{t,l,x} - \min_l p_{t,l,x})
            (\max_l p_{t,l,y} - \min_l p_{t,l,y})}.
$$

This is the normalization factor used by `validate`; it is computed from all
ground-truth landmarks in the frame.  Report each dataset separately; do
**not** average WFLW (98 points) and 300-W (68 points) into one NME.

The validation accuracy value is the **unweighted mean** of the
three per-dataset NMEs from `face_synth_val`, `wflw_val`, and `ibug_val`.
`validate` also logs a sample-count-weighted mean and each individual NME, but
the unweighted mean is what updates `best_nme` and controls checkpoint saves.
The WFLW official test set and the WFLW-V test clips are not used by this
selection path.

### Temporal stability and drift: normalized Allan deviation

Evaluate temporal metrics on complete held-out WFLW-V validation videos.  Let

$$
e_{t,l} = \frac{\hat p_{t,l} - p_{t,l}}{d_t}
$$

be the two-dimensional landmark error normalized by the ground-truth face scale
$d_t=\sqrt{w_t h_t}$.  For an averaging time of $m$ frames, let
$\bar e^{(m)}_{t,l}$ be an overlapping mean over $m$ consecutive errors.  Define
normalized Allan deviation (NAD) as

$$
\operatorname{NAD}(m) =
\sqrt{\frac{1}{2}\operatorname{mean}_{t,l}
\left\|\bar e^{(m)}_{t+m,l}-\bar e^{(m)}_{t,l}\right\|_2^2}.
$$

The `validate_jitter` path evaluates $m=1,\ldots,60$ for the 120-frame
validation clips.  It calls `navar_pos`, so the logged `nad_vals` are pooled
normalized Allan **variances**, not square-rooted deviations: each value is
averaged over landmarks and clips.  It also logs NMF and the sum of these 60
values (`nad_aoc`).

In `validate_jitter`, coordinates and hidden state are carried from the
preceding frame unless the ablation says otherwise.  The first frame uses three refinement iterations, while every
subsequent frame uses one iteration.

### Deterministic selection rule

Checkpoint selection follows these steps:

1. On each small validation compute the unweighted
  mixed-dataset NME and update the running minimum
  $\operatorname{NME}_{\min}$.
2. Mark the step as a save candidate when
  $\operatorname{NME}\leq1.01\,\operatorname{NME}_{\min}$, using
  `TrainParams.nme_coeff = 1.01`.
3. For every save candidate, run `validate_jitter` on the 50-video WFLW-V
  validation split unless that validation already occurred at the big-
  validation interval.  Store `(NME, NMF)` for the step; the NAD values are
  logged but do not participate in ranking.
4. Keep at most 20 candidate records and numbered checkpoint files.  When the
  limit is exceeded, rank by ascending NME, then ascending NMF, then ascending
  training step, and delete the worst entry.  Thus earlier steps win exact
  NME/NMF ties.  A separate `latest.pth` is written on the latest-checkpoint
  cadence (default every 200 steps) and is not a selected checkpoint.

Selection is based primarily on the unweighted mixed-dataset NME, then selecting lowest
NMF on the top-3. The numbered checkpoints are the artifacts
produced by this selection process; `latest.pth` is maintained separately.

## Minimal training matrix

The matrix has one reference configuration and seven ablation configurations:
**eight independent training configurations** in total.  The direct
coordinate-regression comparison is a benchmark comparison, not an internal
ablation: reproducing a separate direct detector would be an expensive compound
counterfactual and would not isolate lookup from iterative refinement.  The
matrix deliberately has no C3-by-C4 or encoding-by-updater factorial
combinations; such interactions are out of scope for an eight-page paper.

| ID | Configuration | Change relative to QLOT-final | Claim tested |
| --- | --- | --- | --- |
| **A0** | QLOT-final | No change. | Reference for C1–C5. |
| **A1** | Coordinates-only temporal carry | At every new video frame set `hidden_state=0`, but prefill the preceding frame's predicted coordinates and covariance exactly as A0 does. Apply this reset during training, PyTorch evaluation, and `OnnxQLOT` evaluation, not merely at test time. | **C2**: contribution of recurrent memory beyond coordinate initialization. |
| **A2** | Per-landmark GRU | Replace the WMR plus GRU-like gated update with a parameter-matched GRU applied independently to each landmark. Its only updater input is `corr_feat`: increase its width to 400, project it through `Linear(400, 424) -> SiLU -> Linear(424, 128)`, then apply `GRUCell(128, 128)` over flattened `(B * Q)` landmarks. Retain the prediction head and 128-dimensional temporal state. No cross-landmark message passing. | **C3**: chosen update operator versus a standard recurrent updater. |
| **A3** | Standard GRUCell updater | Set the WMR mixer `out_dim` to `hidden_state_dim` and replace `gate_proj` and `hidden_proj` with `torch.nn.GRUCell(hidden_state_dim, hidden_state_dim)`. Apply the cell independently over flattened `(B * Q)` landmarks, using the mixer output as the cell input and the preceding hidden state as its hidden input. Do not parameter-match this version; report its exact parameter count and MACs. | **C3**: standard GRUCell versus the current GRU-like gated updater. |
| **A4** | Mean-only WMR | Retain WMR routing, basis slot attention, and the GRU-like gated update, but remove the second-moment/spread input and its spread-conditioning branch. Basis slots contain only the weighted value mean (and the existing query/hidden centroid branch). | **C3**: value of WMR's within-slot dispersion descriptor rather than low-rank pooling alone. |
| **A5** | L2 coordinate-only model | Predict only $(x,y)$ and the coordinate delta; remove the covariance head, `LowRankCov2D`, spatial/acceleration/delta GNLL terms, covariance consistency term, and covariance gate. Retain the GRU-like gated update and a one-headed version of the prediction GLU MLP. Train with the same deeply supervised Euclidean coordinate loss (`sqrt(dx^2 + dy^2)`) and coordinate-only flip consistency, data, and update architecture. | **C4**: the complete 2D covariance plus GNLL supervision recipe versus coordinate L2. |
| **A6** | Spatial-only GNLL | Remove temporal GNLL terms (delta, acceleration). Keep covariance head and spatial GNLL active. | **C4**: isolates the contribution of temporal GNLL terms to jitter reduction over spatial GNLL alone. |
| **A7** | Geometric Fourier features | Replace `PhaseModulatedPE` with the Fourier-only geometric frequency progression from [2] (as implemented in `encode_query_points` in commit `1c7e77b06126a94b1692f11fff111f8803dc8687`). Use 24 sin/cos frequency pairs per coordinate (144 features), with `tau=0.02` and input scaling by 3; concatenate the unscaled raw points, then apply a learned `Linear(147, 147)` projection. | **C5**: final encoding versus the spectral Fourier baseline. |

### Required implementation details for fair controls

- A1 must preserve previous coordinates and covariance. Resetting both the
  landmark prefill and hidden state would test tracker initialization rather
  than recurrent memory. The 0.5% training recovery sample resets only the
  hidden state in A1.
- A2–A4 must use the same feature extractor, encoder, hidden-state dimension,
  three-update evaluation budget, initial-coordinate policy, training sequence
  length, loss recipe, and checkpoint rule as A0.  Match trainable parameters
  within approximately 5% where feasible; otherwise report the exact count and
  MACs next to the table. A2 uses a 400-dimensional `corr_feat` output and its
  specified two-layer local projection to match the removed updater capacity
  without introducing a single very wide layer. A3 is not parameter-matched;
  its standard `GRUCell` input and hidden dimensions are fixed by A0's
  `hidden_state_dim`.
- A4 removes only the spread/second-moment descriptor.  Removing the entire
  slot processor or read routing would no longer be a mean-only low-rank
  control.
- A5 has fewer output parameters by design.  Do not add an arbitrary auxiliary
  head merely to equalize counts; report its smaller count and ensure the
  coordinate pathway itself is unchanged. Where the unchanged encoder requires
  a covariance descriptor, provide a fixed zero descriptor rather than a
  learned covariance prediction.
- A7 is trained from scratch.  Equalize the query-encoding width and
  preserve the same canonical query points.  Do not change query-point
  optimization, kernel sizes, or the correlation pyramid while testing an
   encoding.
   In `src/train.py`, spatial GNLL, acceleration GNLL, and delta GNLL are active.
   Keep that exact final recipe and describe it accurately as the active temporal
   terms across all configurations (except A5 and A6, which ablate them).
   Never compare models trained with different unreported loss recipes.

### Export compatibility

Every ablation exports the standard one-step `OnnxQLOT` interface: six inputs
(`image`, `query_points`, `gating_cutoff`, `gating_radius`,
`prefill_hidden_state`, and `prefill_starting_landmarks`) and two outputs
(`predictions` and `hidden_state`). The prediction and landmark-prefill tensors
remain `(B, Q, 7)` in the legacy layout `[x, y, log_sigma_x, log_sigma_y,
rho_raw, dx, dy]`, and hidden state remains `(B, Q, 128)`.

- A1 accepts the standard hidden-state input but supplies zeros at runtime,
  while continuing to carry the landmark-prefill tensor.
- A2, A3, A4, A6, and A7 use the standard interface without adapters.
- A5's model is covariance-free internally. Its export adapter accepts the
  standard seven-channel landmark prefill, ignores its covariance channels,
  and returns `[x, y, 0, 0, 0, dx, dy]`; the three zero channels are finite,
  neutral dummy covariance parameters solely for `OnnxQLOT` compatibility.

## Runs, randomness, and statistical reporting

The absolute minimum budget is eight fixed-seed trainings (A0–A7).  This is
sufficient to screen the planned matrix, but it does not quantify optimization
variance.  The defensible paper budget is **three seeds per configuration**
(24 trainings) with the same three seed IDs and data manifests for every row.
If compute is constrained, run the eight-configuration screen first, then repeat
A0 and every ablation retained for the main table with two additional seeds;
do not claim significance for a single-seed result.

For each selected checkpoint, report mean and standard deviation across seeds.
For temporal metrics, compute 95% confidence intervals with a paired bootstrap
that resamples **videos**, not individual correlated frames or landmarks.  For
static NME, resample images.  Include the paired per-video NAD difference
between A0 and the relevant ablation whenever practical.

## Tables and figures for the paper

A compact results presentation can fit the page limit:

1. **External comparison table (C1):** QLOT and direct coordinate-regression
  baselines, with input resolution, landmark count, training-data differences,
  parameters, MACs/FLOPs, WFLW/300-W NME, WFLW-V NME, and NAD.  Use a matched
  in-house re-evaluation whenever weights and code are available; otherwise
  label literature numbers clearly and never compare temporal metrics measured
  on different clips or protocols.  This establishes a practical
  accuracy--efficiency--stability comparison, not a causal isolation of local
  lookup or refinement.
2. **Refinement-step analysis (C1):** evaluate the single selected A0 checkpoint
  at one, two, and three refinement iterations with all weights fixed.  Report
  NME, NAD, MACs, and latency.  Describe this as an inference-time
  accuracy--stability--cost analysis, not as an architectural ablation.
3. **Internal ablation table:** rows A0–A7; columns parameters, MACs, WFLW test
  NME, 300-W common/challenging NME, WFLW-V test NME, $\operatorname{NAD}(1)$,
  $\operatorname{NAD}(4)$, and $\operatorname{NAD}(16)$.  Bold only results
  whose seed/paired-video uncertainty supports the comparison.
4. **NAD curve figure:** A0 plus A1, A2, A3, A4, A5, and A6, plotting median
   NAD against averaging time on log axes.  This makes a jitter-versus-drift
   trade-off visible without adding a wide table.
5. **C5 inset or small table:** A0 and A7, with NME and the NAD geometric
   score.  State their identical query dimensions and parameter counts.
6. **Optional calibration supplement:** For A0 only, report covariance NLL and
   empirical coverage of predicted covariance ellipses.  Calibration supports
   interpretation of C4, but it must not substitute for NME or NAD.
