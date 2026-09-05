#import "template.typ": indent, paper
#import "utils.typ": TT, bu, pretty-table

#let authors = (
    (name: "Dominik Gschwind", affl: ("OST",), email: "dominik.gschwind@ost.ch"),
    (name: "Martin Weisenhorn", affl: ("OST",), email: "martin.weisenhorn@ost.ch"),
    (name: "Hannes Badertscher", affl: ("OST",), email: "hannes.badertscher@ost.ch"),
)

#let affils = (
    "OST": (
        institution: [Eastern Switzerland University of Applied Sciences (OST)],
        location: "Rapperswil-Jona, Switzerland",
    ),
)

#show: paper.with(
    title: [_QLOT_: Queried Learned Optimization for Face Landmark Tracking],
    authors: (authors, affils),
    keywords: (),
    date: auto,
    abstract: [
        Facial landmark detection is a mature field, with tracking approaches capable of
        real-time inference on mobile devices. Yet most applications treat tracking as a
        post-processing problem, applying classical filters for jitter suppression. We
        propose QLOT (Queried Learned Optimization for Tracking), which treats face
        alignment as an iterative, query-conditioned learned optimization process rather
        than single-shot inference. Incorporating ideas from RAFT, a recurrent optical
        flow estimator, QLOT fuses per-landmark correlation features with temporal context
        carried by a recurrent hidden state, exchanged across landmarks through a low-rank
        Write-Mix-Read block. A composite objective of decoupled spatial and temporal
        Gaussian negative log-likelihood terms supervises directional uncertainty and the
        temporal derivatives of the error. QLOT achieves competitive accuracy to the
        state-of-the-art while improving temporal stability, staying efficient enough
        for real-time mobile deployment (2M parameters, 0.65G MACs/frame). To evaluate
        tracking, we generalize Normalized Mean Flicker to a multi-timescale stability profile
        based on the Allan variance.
    ],
    bibliography: bibliography("bibliography/citations.bib"),
    appendix: [
    ],
    mode: "preprint",
    track: "algorithms",
    paper-id: "742",
    pagenumbers: true,
)

= Introduction <sec-intro>

Facial landmark detection (FLD), the task of extracting precise coordinate positions for semantic
facial features, is a mature field of research.
Such systems already power many applications, including facial expression recognition, gaze estimation, and augmented reality.
Approaches like FaceMesh @grishchenko2020-mp-facemesh, which extract not only 2D locations but even a dense 3D mesh of vertices, have been widely deployed.
Thus, FLD models have become commonplace in mobile devices, while typically also requiring real-time performance.

Despite this commercial success, surprisingly few approaches
@yin2024-1dformer @lee2023-stabilized-3d-face-alignment @lee2023-stabilized-temporal-3d-face-alignment
@micaelli2023-deep-equilibrium-models @zhu2020-spatial-temporal-deformable incorporate temporal information as a part of their end-to-end architecture.
Instead, most methods strictly function as _pure detectors_ that process continuous video streams
as a series of independent frames.
This stateless nature makes temporal instability or _jitter_ virtually unavoidable.
Visually, jitter manifests as high-frequency noise, where the predicted coordinates move erratically across subsequent frames, even though the underlying facial features remain static.

#set par(spacing: 0.58em)

Large-capacity models, for example  @chandran2024-infinite-3d-landmarks, can achieve relatively stable predictions through brute-force feature extraction, but
their massive computational requirements make them infeasible for real-time mobile deployment.
Practical detection systems designed for video-use thus rely on post-hoc classical filters @nugroho2023-temporal-filtering-comp @wu2021-fld-opticalflow @prabhu2012-kalman-asm to dampen jitter.
However, making use of temporal correlations as part of the architecture tackles the jitter
problem directly.
Moreover, using past alignments as structural priors can even increase detection performance.

We note five fundamental sources of jitter in FLD:

1. *Input instability.* Noise and instability of the cropped face region---typically from a separate
    model or tracking algorithm---propagate through the system and affect the landmark predictions @chandran2024-infinite-3d-landmarks.

2. *Label noise and inconsistencies.* Suboptimal annotation quality, semantic differences between datasets,
    or more generally, label noise, teach the model uncertain outputs that are input-sensitive @chandran2024-infinite-3d-landmarks @dong2018sbr @su2019-soft-labels.

3. *Geometric ambiguities.* Extracting the underlying 3D facial geometry from monocular
    2D images is an ill-posed problem
    due to the scale ambiguity.
    Thus, perspective changes or occlusions can cause large position variations.

4. *Visual variability, noise and occlusions.* Changes in lighting, pose, expression, and self or external occlusions
    can introduce oscillations between plausible landmark configurations @micaelli2023-deep-equilibrium-models @lee2023-triple-discriminators @jin2016facealignmentinthewildsurvey.

5. *Lack of temporal context.* A pure detector processes each frame independently, and so cannot utilize
    strong temporal correlations for antialiasing or noise suppression.

#indent
While post-hoc filtering stabilizes positions, it operates solely on output coordinates,
causing oversmoothing and lag.
Integrating a temporal pathway, learned end-to-end by training on video, can directly prevent this.
Still, training on static images remains essential
as annotated video data is scarce.
A practical solution must satisfy the mobile performance budget,
while bounding runaway error accumulation or _drift_---a critical concern for recurrent tracking.

#set par(spacing: 0.5em)

Based on RAFT @teed2020-raft-optical-flow, a recurrent optical flow estimator, and @chandran2023-3dqueries,
a query point-based FLD model, we
design a stable recurrent architecture exhibiting near-state-of-the-art
accuracy with mobile-friendly efficiency.
Furthermore, we generalize a temporal stability metric based on the Allan variance, which allows us to quantify both jitter and drift.
Our main contributions are as follows:

- We propose _QLOT_, Queried Learned Optimization for (Face Landmark) Tracking, a
    query-conditioned recurrent architecture that reformulates landmark detection as iterative learned
    optimization over time.
    It fuses semantic history and local image evidence through a low-rank
    _Write-Mix-Read_ (WMR) block, while being temporally stable and lightweight enough
    for real-time edge deployment.
- We extend the simple Gaussian Negative Log-Likelihood (GNLL) loss used in @wood2022dense @chandran2023-3dqueries
    to a mixture of decoupled spatial and temporal GNLL losses, using full 2D covariance representations to allow
    for directional uncertainty. A simple L2 loss is used for accuracy supervision.
- We demonstrate how to effectively train QLOT on a mixture of image and video data. Here, adopting
    the query mechanism of @chandran2023-3dqueries proves highly synergistic, as it eliminates the
    need for label translation or normalization across diverse datasets.
- To properly evaluate continuous tracking performance, we introduce a generalization of the
    Normalized Mean Flicker (NMF) @micaelli2023-deep-equilibrium-models metric based on the Allan variance.
    This provides a principled way to characterize the jitter and drift behavior of any tracking model.

= Related Work <sec-related>

FLD architectures are broadly categorized by their output representation into three
paradigms @jin2016facealignmentinthewildsurvey @meher2023surveyclassificationfacealignment.
_Direct coordinate regression_ methods predict points directly; they are fast but often yield local
structural imprecisions unless refined via cascaded networks @decafa2019-deep-convolutional-cascade.
_Heatmap-based_ architectures preserve spatial context by predicting dense probability distributions.
They are accurate but sensitive to outliers and computationally heavy @xiang2025-parallel-optimal-position-search @lan2021-hih-heatmap-in-heatmap @wang2019-adaptive-wing-loss.
Pushing them to sub-pixel precision required evolving beyond standard losses (MSE, $L_1$),
which
suffer from dilated responses and oscillatory convergence, to formulations like the _Adaptive Wing Loss_ @wang2019-adaptive-wing-loss.
Finally, _face model-based_ methods regress the parameters of statistical shape models.
3D Morphable Models (3DMMs) are preferred since they model the inherent 3D facial geometry,
gracefully handling occlusions, and are less affected by texture and lighting variations
@ximin2024-3d-facial-landmark-detection-survey.
Instead, they trade off fine-grained local accuracy to maintain rigid global geometry.

*Learned Optimization for Sparse Prediction.*
Rather than inferring coordinates in a single forward pass,
QLOT treats face alignment as a _learned optimization_ process.
Introduced to FLD by the Mnemonic Descent Method (MDM) @trigeorgis2016-mnemonic-descent-method,
this approach has recently dominated dense matching tasks like optical flow via RAFT @teed2020-raft-optical-flow,
which refines estimates by indexing a precomputed correlation volume using a Gated Recurrent Unit (GRU) @cho2014-gated-recurrent-unit.
While RAFT operates on dense pixel grids,
QLOT extracts vision evidence from per-landmark localized correlation volumes.
These volumes are also dense maps, but---being interpreted by a learned update
operator---they can remain at low backbone resolutions,
avoiding high-resolution decoding that makes heatmaps expensive.
Unlike cascaded estimation, where each stage is an independent sub-network
@decafa2019-deep-convolutional-cascade, the learned optimizer shares
parameters across iterations and landmarks, emitting position _deltas_ that
are accumulated as in @trigeorgis2016-mnemonic-descent-method @teed2020-raft-optical-flow.

#set par(spacing: 0.6em, leading: 0.52em)

*Query-based and Continuous Architectures.*
Datasets annotate landmarks under incompatible schemes where topology and semantic meaning differ.
Traditionally, this made multi-source training dependent on label translation,
adding label noise while discarding some semantics @chandran2023-3dqueries.
Recent continuous and query-based architectures break this limitation.
By prompting a network with 3D queries from a canonical face mesh @chandran2023-3dqueries @chandran2024-infinite-3d-landmarks
or a structural prompt @xia2025knowledge-discrepancies, these models learn to dynamically
output the corresponding landmarks.
Similarly, transformer-based aligners refine via _learned_
landmark queries, conditioning patch-based coordinate updates
@xia2022-sparse-local-patch-transformer @li2022-repformer,
a paradigm extended to tracking by 1DFormer @yin2024-1dformer
using temporal attention.
These queries encode a learned landmark identity, which for QLOT
is derived from an _encoding_ of the canonical 3D points.

*Probabilistic and Uncertainty-aware FLD.*
To tackle label noise, @wood2022dense uses fully synthetic data.
They then estimate a 2D circular Gaussian per landmark, allowing the model to differentiate
between hard or occluded, and easy in-view locations.
QLOT builds on this idea by extending it to a full _directional_ 2D covariance
representation,
and makes use of it within spatial @wood2022dense @chandran2023-3dqueries @chandran2024-infinite-3d-landmarks and temporal GNLL losses.

*Jitter-free Tracking.*
Jitter, also called flicker or temporal instability, can be quantified by the
Normalized Mean Flicker (NMF) @micaelli2023-deep-equilibrium-models, and evaluated
on dedicated video benchmarks such as 300-VW @tai2018highlyaccuratestableface
and WFLW-V @micaelli2023-deep-equilibrium-models.
Standard practices treat it as a post-processing problem, applying heuristic filters
@nugroho2023-temporal-filtering-comp, Kalman filtering @prabhu2012-kalman-asm,
learned smoothing @tai2018highlyaccuratestableface, or even optical flow-based
tracking @wu2021-fld-opticalflow.
However, these methods induce lag and discard the rich semantic history available
naturally in video.
Efforts to encode temporal priors natively include within-batch synthetic
temporal augmentation @guo2020-3ddfa-v2, multi-view tuning @zeng2023-multi-view,
or directly incorporating recurrent/transformer modules
@zhu2020-spatial-temporal-deformable @yin2020-attentive-1d-heatmap.
_Recurrence without Recurrence_ (RwR)
@micaelli2023-deep-equilibrium-models uses fixed-point optimization for
position estimation.
Although trained only on static images, RwR can apply temporal smoothing at test time, while reusing previous frame
predictions as initialization.
QLOT also forwards previous predictions, but learns actual temporal dynamics
by propagating a recurrent hidden state and explicitly training via backpropagation through time.

#set par(spacing: 0.5em, leading: 0.52em)

*Low-rank cross-element communication.*
Cross-element communication can avoid explicit _all_-pairs interactions by routing
information through $S << N$ bottleneck elements,
$
    bu(Z) = bu(B)_R cal(M)(bu(B)_W bu(V)), quad
    bu(B)_W, bu(B)_R^TT in RR^(S times N),
$
where $bu(B)_W$ writes values into the bottleneck, $cal(M)$ mixes them, and $bu(B)_R$ reads the result back.
Linformer @wang2020-linformer uses a static learned write projection, identity
mixing, and a query-to-projected-key attention matrix for the read.
Set Transformer's ISAB encoder @lee2019-set-transformer first attends from
learned inducing points to the elements and then from the elements to the
induced features. Slot Attention @locatello2020-slot-attention implements only
the first direction. Competing slots aggregate inputs and are iteratively
updated without elementwise readback. Perceiver IO @jaegle2022-perceiver-io
writes inputs to a latent array by cross-attention, mixes it with a deep latent
transformer, and reads arbitrary output queries by cross-attention.
Our WMR block predicts write and read
routes _asymmetrically_ from distinct signals,
bottleneck slots are weighted _moment statistics_ rather than attention outputs,
and temporal information enters through routing, keeping the slots stateless unlike
@locatello2020-slot-attention.


#v(3mm)
= Method <sec-method>

In this section, we introduce our proposed QLOT
framework, shown in @fig:qlot, including architectural details and training procedure.
Then, we discuss a specific temporal stability metric and propose a physically motivated generalization
which we use in @sec-experiments to evaluate jitter and drift.

#set par(spacing: 0.5em, leading: 0.5em)

#figure(
    image("figures/model.pdf"),
    scope: "parent",
    placement: top,
    caption: [
        The QLOT architecture.
        Green components do not contain learned parameters. Most activation functions are omitted for clarity.
    ],
) <fig:qlot>

== Queried Learned Optimization for Tracking <sec-method-architecture>

QLOT reformulates facial landmark detection as an iterative, query-conditioned refinement process.
It is composed of five core components: a lightweight _backbone_, _canonical query point_ encoding,
landmark-specific _correlation volume_ generation, a _feature encoding_ stage, and a _recurrent optimization_ module.

=== Feature Extraction
As a state-of-the-art backbone in both quality and efficiency,
we adopt _HGNetV2_ @pp-hgnetv2-2024. An ImageNet-pretrained B1-variant receives square RGB images of size 224#sym.times\224
pixels, outputting four levels of feature maps with strides 4, 8, 16, and 32 relative to the input image.
We trim the last two levels to 384 and 192 channels, correcting channel-inflation for the original classification task,
and removing 60% of the backbone parameters with a 10% reduction in FLOPs.
Towards semantically rich features, we construct a _Feature Pyramid Network_ (FPN) @lin2017featurepyramidnetworksobject
of the first three levels using lightweight grouped and depthwise convolutions, simultaneously
projecting to a common dimension of 96 channels.
The final stride-32 map is average pooled and projected to a 96-dimensional _global image context_ vector.

=== Canonical Query Points
// We adopt the simpler mechanism of
// #[---]rather than the extension
// with learned dataset-specific semantic landmark differences @chandran2024-infinite-3d-landmarks.
3D canonical points @chandran2023-3dqueries, as a proxy of landmark identity, are
annotated once on a shared reference mesh for all landmarks.
The predictor is then constructed such that
conditioning on any desired combination of these 3D points
causes it to output only the corresponding landmarks.
In our case, @li2020-ict-face-kit is chosen as the
canonical face shape (expression parameters $e_24 = 0.5$ and $e_26 = 0.4$,
others zero).
Like this, a single model can be trained jointly on multiple datasets with even
heterogeneous annotations, while queries close the semantic gap
between differing label schemes. At test time, any desired subset of keypoints can be
extracted without retraining. Anchored to real 3D facial geometry, the canonical
points also act as a structural prior over the spatial layout of facial features.

To obtain landmark identity features, each canonical query point
is---after spectral encoding---fed
through two 2-layer MLPs.
The first, higher capacity MLP outputs a 192-dimensional vector that is fed into the _correlation volume_ block.
A second MLP produces an encoding of size 96 used as query-conditioning for the _recurrent optimization_ module.

*Spectral Encoding.*
Each canonical query point $bu(p) in RR^3$ is encoded by a phase-modulated positional
encoding (PMPE),
#v(-0.4mm)
$ gamma(bu(p)) = [bu(p), sin(omega_1 bu(p) + phi_1), dots, sin(omega_L bu(p) + phi_L)], $
#v(-0.4mm)
where frequencies $omega_l$ and phases $phi_l$ are _learnable_. The initialization
provides a well-spread, aperiodic starting point.
It combines a _Fourier branch_ of 16 sin/cos pairs with geometric frequencies
$omega_k = pi tau^(-k\/16)$, $tau = 0.03$ @xia2025knowledge-discrepancies, and a
_phase-modulation branch_ of 8 pairs covering the complementary low-frequency band
@roblox2025-cube.
In contrast to the original PMPE, we assign each phase-modulation channel a distinct carrier
in $[pi\/4, pi]$ from a golden-ratio Weyl sequence, and concatenate the two branches
instead of summing,
to leave the downstream MLP free to mix a total of 147 channels.

=== Correlation Volume
#v(-0.25mm)

RAFT @teed2020-raft-optical-flow uses pairs of frames to compute a dense 4D correlation volume,
the same concept is applied to landmark-specific template matching.
We produce a multi-level multi-head correlation volume for each landmark.
The templates, or correlation kernels, stem from a learned per-level projection
of the 192-dimensional query point-encoding.
Each map level $i$---with feature map resolution 56#sym.times\56 for $i=0$, 28#sym.times\28 for $i=1$, and 14#sym.times\14 for $i=2$---is convolved with template kernels of size #box[$R_i$#sym.times$R_i$]
in groups of $K_i$ _correlation heads_ giving $K_i$ single-channel correlation maps per level.

Each feature map level, broadcast across $N$ landmarks, has the shape
$(N, K_i, 96slash K_i, H_i, W_i)$
with square #box[$R_i$#sym.times$R_i$] kernels.
This gives a computational complexity of $O(N R_i^2 C H_i W_i)$
per level $i$
where $C=96$ is the total channel dimension.
We experiment with different kernel sizes and group counts, but find that $R_i=2, 1, 1$
and $K_i=2, 3, 4$ give good results while keeping the cost of the expensive correlation operation low.
Note that $R=1$ simply computes a channel-wise dot product.
In total, the correlation volume consists of shape $(K_i, H'_i, W'_i)$ maps for each level $i$ and landmark $n$.
Correlation map resolutions $H'_i$#sym.times$W'_i$ are the result of valid-padded convolutions.
The volume is computed once per input image since neither the query points nor the
backbone features change across refinement iterations.

=== Feature Encoding

Every iteration, the correlation volume is indexed around the previous landmark estimates.
A 5#sym.times\5 grid centered on these coordinates bilinearly samples one patch
per head $k$, level $i$, and landmark $n$.
Noisy but weak correlation responses are dampened by a learned-temperature soft-shrink
activation inspired by @zhao2020-deep-residual-shrinkage-networks,
$
    sigma(x) = x dot "sigmoid"(abs(x)slash tau_(k,i)).
$
Instead of flattening, grouped 3#sym.times\3 level-specific convolutions---exploiting the persisting spatial structure---extract intermediate features
(126, 135, and 144 channels for levels $i=0,1,2$, respectively).
After projection and Root-Mean-Square (RMS) normalization, all levels are concatenated and compressed
with a 2-layer MLP followed by another RMS norm to form the initial _correlation feature vector_ $bu(f)'_n$.

For stable visual, geometric and temporal grounding we construct
_context features_ $bu(c)_n$ from three distinct signals:

1. *Image Context:* The _global image context_, conditioned via Feature-wise Linear Modulation
    (FiLM) @perez2018-film
    on the highest-level ($i=2$) correlations.

2. *Recurrent State:* The model's _last predictions_ (positions, covariances, and update deltas), projected to 32 dimensions each and processed by a 2-layer MLP.

3. *Landmark Identity:* The spectral _query encodings_, processed by a separate 2-layer MLP
    (denoted as $bu(q)_n$).

These three signals are concatenated and compressed via a 2-layer MLP before RMS normalization, giving the
resulting context vector $bu(c)_n$.
We residual-add a projected version of $bu(c)_n$ to the correlation features
to obtain a unified feature vector $bu(f)_n$ for each landmark $n$.
Both $bu(f)_n$ and $bu(c)_n$ are 256-dimensional.

=== Recurrent Optimization

The recurrent optimization module closes the iterative refinement loop.
It maintains a per-landmark hidden state $bu(h)_n in RR^128$ that persists both
across refinement iterations within a frame and across consecutive frames,
carrying temporal semantics forward in time.
Each update step first exchanges information across landmarks in a Write-Mix-Read
block (@fig:wmr-block) and then integrates the result into the hidden state through a
GRU update.

#[
    #set figure(gap: 11pt)
    #show figure.caption: it => [#it #v(0.4mm)]
    #figure(
        {
            image("figures/update_prediction.pdf")
        },
        placement: auto,
        caption: [
            Write-Mix-Read block for spatio-temporal fusion.
        ],
    ) <fig:wmr-block>
]

*Write-Mix-Read.*
WMR factorizes cross-landmark communication through $S=8$ _basis slots_
per head (4 heads of width $D=64$), reducing the interaction to a
low-rank mapping with $O(N)$ runtime versus $O(N^2)$ for full attention.
Routing is _asymmetric_. The stable context features $W=bu(c)_n$ decide what each landmark
_writes_, while the hidden-state/landmark-identity pairs $R=[bu(h)'_n, bu(q)_n]$, with
$bu(h)'_n = "RMSNorm"(bu(h)_n)$, decide what each landmark _reads_; the correlation
features $V=bu(f)_n$ are the _values_ that are written, mixed, and read back.

#set par(spacing: 0.52em, leading: 0.51em)

Concretely, the context features produce two write bases
$bu(B)^((v))_W, bu(B)^((r))_W in RR^(S times N)$ with elements $b_(s n)$, learned-temperature-scaled
and softmax-normalized _over landmarks_.
Given projected correlation and read features $bu(v)_n, bu(r)_n in RR^D$,
each slot accumulates weighted first and second moments,
#set math.vec(delim: "[")
$
    #h(-3mm)
    bu(mu)_s = sum_n b_(s n)^((v)) bu(v)_n, space
    bu(m)^((2))_s = sum_n b_(s n)^((v)) bu(v)^2_n, space
    tilde(bu(r))_s = sum_n b_(s n)^((r)) bu(r)_n, \
    #h(-3mm)
    bu(sigma)_s = sqrt(bu(m)^((2))_s - bu(mu)_s^2), quad
    "RMS"_s = sqrt(1/D bu(1) dot bu(m)^((2))_(s)).
$

#indent Conceptually, a basis slot can be viewed as a high-dimensional soft cluster of landmark
features.
The write basis softly assigns landmarks to $S$ overlapping groups with
centroid $bu(mu)_s$ and dispersion $bu(sigma)_s$. The dispersion lets a slot encode the
agreement of its members.
Normalizing both quantities by the RMS decomposes the slot energy into coherent
and dispersed components, $hat(bu(mu))_s = bu(mu)_s slash "RMS"_s$ and
$hat(bu(sigma))_s = bu(sigma)_s slash "RMS"_s$, so that
$sum_d (hat(mu)_(s d)^2 + hat(sigma)_(s d)^2) = D$.

Before readback, each slot is refined by four residual branches gated by a
learned per-head scalar initialized near zero @bachlechner2020-rezero @touvron2021-going-deeper-image-transformers.
(i) A Gated Linear Unit (GLU) @dauphin2017-glu projection of the
dispersion descriptor is added to $bu(mu)_s$,
allowing the slot to be reweighted by the agreement of its members.
(ii) The GLU-projected read centroid $tilde(bu(r))_s$ is added to the slot.
This is the pathway through which temporal semantics enter. Each slot is
conditioned on the recurrent states of the landmarks routed to it, so
the temporal history of one cluster can modulate the content read back by another.
(iii) Self-attention across $S$ slots, followed by (iv) a 2-layer MLP lets the learned
topology factors exchange information.
We denote the slot results by $tilde(bu(v))_s in RR^(1 times D)$
stacked as $tilde(bu(V)) in RR^(S times D)$.

The read side mirrors the write side. State/identity-pairs yield
a read basis $bu(B)_R in RR^(N times S)$, softmax-normalized _over slots_. Each
landmark reads a convex combination #box[$bu(Z)^((r)) = bu(B)_R tilde(bu(V)) in RR^(N times D)$] of the processed slots.
Composing write and read (before slot refinement) gives a directed interaction matrix
$bu(A) = bu(B)_R bu(B)^((v))_W$ of rank at most $S$,
a content-based, low-rank approximation of full landmark attention.
Since all per-landmark quantities are bound to corresponding query points
and all maps are shared,
WMR is _equivariant_ to landmark permutations, and the slots
(within a given query set) _invariant_.

The output integrates a local bypass via FiLM of the correlation features,
$bu(z)_n^((l)) = bu(z)_n^((r)) dot.o lambda(bu(f)_n) + beta(bu(f)_n)$
with $lambda$ initialized to one and $beta$ to near zero, so the bypass starts off inactive.
Finally, all parallel heads $h$ are concatenated and out-projected,
$bu(z)_n = bu(P)_o \[bu(z)_(n h)^((l)), dots]$.
During training, slot-writing is randomly disabled for 15% of
landmarks (entirely for pupil landmarks, whose motion is largely independent
of the rest of the face), with gradients back into the basis slots also stopped.
Affected landmarks still read normally.

#set par(spacing: 0.5em, leading: 0.50em)

*Gated State Update.*
The WMR output is split into candidate features and gate features.
The latter produce three sigmoid gates---reset $bu(r)^((g))$, update $bu(u)^((g))$, and gain
$bu(g)^((g))$---which form the candidate and the next hidden state,
$
    tilde(bu(h))_n & = tanh(
                         bu(r)_n^((g)) dot.o
                         bu(P)_h bu(h)'_(n,t-1) + bu(z)_n
                     ) dot.o bu(g)^((g))_n, \
       bu(h)_(n t) & = (bu(1) - bu(u)_n^((g))) dot.o bu(h)_(n,t-1) + bu(u)_n^((g)) dot.o
                     tilde(bu(h))_n.
$
$t$ indexes successive refinement steps, including those that cross
frame boundaries.
This setup mirrors a GRU, except that the input map is absorbed into the WMR output
projection. The added gain gate scales the bounded candidate multiplicatively,
independent of the $tanh$ argument, so unreliable candidates can be suppressed
without saturating the nonlinearity.
Gates are initialized toward state preservation. The update gate starts at
$op("logit")(0.4)$, the gain gate at $op("logit")(0.9)$, while the reset gate begins centered
at $op("logit")(0.5)$.
The new hidden state is mapped by a two-headed GLU projection into separate coordinate
and covariance streams; linear heads then emit the position delta
$[Delta x, Delta y]$ and three covariance parameters.
$[x, y]_t = [x_(t-1) + Delta x, y_(t-1) + Delta y]$.
We use normalized image coordinates with
$x,y in [-1, 1]$.

*Covariance Parameterization.*
Extending the scalar variance of @wood2022dense @chandran2023-3dqueries to a full 2D
covariance $bu(Sigma)$, the head predicts $bu(theta) = [log sigma_x, log sigma_y, "artanh"(rho)]$,
such that
$
    #h(-4.5mm)
    bu(Sigma) = mat(sigma_x^2, rho sigma_x sigma_y; rho sigma_x sigma_y, sigma_y^2), space
    bu(L) = mat(sigma_x, 0; rho sigma_y, sigma_y sqrt(1 - rho^2)).
$
Compared to
$[log l_11, log l_22, l_21]$,
a Cholesky parameterization
with $bu(Sigma) = bu(L)bu(L)^TT$---where
$sigma_y^2 = l_21^2 + l_22^2$ couples the marginal variance with the correlation and
log/linear domains are mixed---in the $sigma rho$-parameterization,
all parameters are in the $log$-domain and decoupled.
$2 op("artanh")(rho) = "logit"((1+rho)slash 2)$ is the
log-odds of the correlation rescaled to $(0,1)$.
Positive definiteness follows from
$sigma_x, sigma_y > 0$ and $abs(rho) < 1$.

== Training Procedure <sec-method-training>

QLOT is trained by unrolling the model over video clips of length $L$.
Frame $l in {1, dots, L}$ is refined over $i in {1, dots, I}$ iterations, yielding a total of
$T = L dot I$ refinement steps indexed by
$t(l, i) = (l - 1) dot I + i$.
Between frames, the coordinate estimate $bu(mu)$, covariance parameters $bu(theta)$, and hidden state
$bu(h)$ from the final iteration ($i = I$) of frame $l$ initialize the first iteration of frame $l + 1$.
Labels are denoted $bu(y)$.

To train for both cold-start detection and single-iteration steady-state tracking,
the per-frame iteration budget $I$ is sampled dynamically:
With a 0.1% chance, set $I=12$.
Otherwise, steps $<$ 2,000: #box[$I tilde.op cal(U)(2, 6)$]\;
steps 2,000 to 4,999: #box[$I tilde.op cal(U)(2, 5)$]\;
steps 5,000 to 9,999: #box[$I tilde.op cal(U)(2, 4)$]\;
steps $>=$ 10,000: #box[$I tilde.op cal(U)(2, 3)$] with $5%$ of clips trained at $I=1$.
We reset $bu(mu)$, $bu(theta)$ and $bu(h)$
with probability 0.5% to train failure recovery.
After all iterations of each frame we clamp $-10 <= log(sigma_x),$ $log(sigma_y) <= 3$
and $-2 <= x, y <= 2$  for numerical stability.
Furthermore, we stop gradients
to the previous frame's coordinates $bu(mu)_(t-1)$ 
as suggested by @teed2020-raft-optical-flow, but keep $bu(theta)_(t-1)$ live.
Initial coordinates are set to randomly perturbed $[x,y]$ query point positions,
$bu(theta)$ and $bu(h)$ are initialized to zero.

We limit training to 110 epochs (1,000 steps per epoch) using AdamW
(weight decay 0.005, gradient norm clipping at 1.0) with a cosine annealing learning
rate schedule from $eta_0 = 2 dot 10^(-4)$ to $eta_"min" = 10^(-5)$.
The HGNetV2 backbone is frozen for the first 400 steps.

=== Datasets and Video Synthesis

Each step we sample from all four datasets:
- *FaceSynthetics* @wood2021fake: 95,000 training, 2,500 validation, and 2,500 test synthetic images with 70 landmarks.
- *WFLW* @wayne2018-look-at-boundary: 7,000 training and 500 validation images from the training split, evaluated on the standard test sets with 98 landmarks.
- *WFLW-V* @micaelli2023-deep-equilibrium-models: 1,000 video sequences (800 training, 50 validation, 150 test) with 98 landmarks.
    Training videos are sampled with a random stride $tilde.op cal(U)(1, 10)$ as $L=16$ clips.
- *300-W* @sagonas2013-300-faces-in-the-wild-challenge: 2,848 training and 300
    validation images with 68 landmarks, evaluated on the standard common and challenging test sets.

#indent *Face Crop Generation.* Input crops are centered on the landmark labels' bounding box and
padded by a dataset-specific ratio (75% for FaceSynthetics, 35% for WFLW, 40% for WFLW-V, 55% for 300-W),
then expanded to a square aspect ratio and augmented by
a random affine per clip.
Validation and test sets use a fixed padding ratio of 10%.
Images are resized to 224#sym.times\224 pixels after augmentations.

*Clip Generation.* Static images are converted into
synthetic clips of length $L=4$. Correlated affine transformations
(translation, scale, rotation, shear)
synthesize video motion.
Photometric augmentations (color jitter, brightness/contrast, gamma, CLAHE, plasma shadows, sun flares)
are sampled once per clip and applied to all frames,
while noise (Gaussian, ISO, shot) and blur (Gaussian, motion) are applied per frame.

=== Loss Function

We use a composite loss of five terms,
$cal(L) = cal(L)_"pos" + cal(L)_"cov" + 0.5 cal(L)_"delta" + 0.25 cal(L)_"acc" + lambda(s) cal(L)_"hcons"$.
Where defined, each loss is averaged over landmarks $n$, iterations $i$, and
frames $l$.

*Spatial accuracy* is supervised by an $L_2$ loss
with exponential decay weights
$w_i = 0.85^(I - i)$
and $W = sum_i w_i$,
$
    cal(L)_"pos" = w_(i)/W norm(bu(mu)_(l i n) - bu(y)_(l n))_2.
$

#indent *Spatial uncertainty* is trained via Gaussian Negative Log-Likelihood (GNLL),
$
    op("GNLL")(bu(mu), bu(Sigma), bu(y)) = & 1/2 log det(bu(Sigma)) + & 1/2 bu(x)^TT bu(Sigma)^(-1) bu(x),
$
where $bu(x) = bu(mu) - bu(y)$.
The coordinate residual is detached ($op("sg")[dot]$),
$
    cal(L)_"cov" = op("GNLL")(op("sg")[bu(mu)_(l i n)], bu(Sigma)_(l i n), bu(y)_(l n)).
$
We split coordinate and uncertainty supervision into disjoint terms. Jointly
trained heteroscedastic NLL couples the mean and variance gradients through
$bu(Sigma)^(-1)$, which permits poor but stable equilibria where the
covariance absorbs the residual instead of the estimate being corrected
@seitzer2022-pitfalls-nll. Stopping the residual's gradients makes $cal(L)_"cov"$
a pure calibration objective for $bu(Sigma)$, while $cal(L)_"pos"$ trains
accuracy mostly unaffected by $bu(Sigma)^(-1)$.

#indent *Temporal GNLL* terms supervise the trajectory against the final
iteration of previous frames.
Covariances are detached and used only as inverse-variance weights,
allowing us to assume uncorrelated errors between frames.
The _delta_ term matches first differences
$Delta bu(mu)_(l i) = bu(mu)_(l i) - bu(mu)_(l-1, I)$ and
$Delta bu(y)_l = bu(y)_l - bu(y)_(l-1)$
to $bu(Sigma)^((v))_(l i) = bu(Sigma)_(l i) + bu(Sigma)_(l-1, I)$,
$
    cal(L)_"delta"
    = op("GNLL")(Delta bu(mu)_(l i n), op("sg")[bu(Sigma)^((v))_(l i n)],
        Delta bu(y)_(l n)).
$
The _acceleration_ term matches
$Delta^2 bu(mu)_(l i) = Delta bu(mu)_(l i) - Delta bu(mu)_(l-1, I)$
(and likewise $bu(y)$) to
$bu(Sigma)^((a))_(l i) = bu(Sigma)_(l i) + 4 bu(Sigma)_(l-1, I) + bu(Sigma)_(l-2, I)$,
$
    cal(L)_"acc"
    = op("GNLL")(Delta^2 bu(mu)_(l i n), op("sg")[bu(Sigma)^((a))_(l i n)],
        Delta^2 bu(y)_(l n)).
$

#indent *Horizontal consistency* is enforced for
WFLW and 300-W, which exhibit left-right annotation asymmetries. Batches contain interleaved original and horizontally flipped image pairs. Predictions on flipped images are mapped back to the original coordinate frame. The loss computes

$
    cal(L)_"hcons" = norm(Delta bu(mu)^"flip"_(l i n))_2^2 + 0.1 dot norm(Delta bu(theta)_(l i n))_1
$

where $Delta bu(mu)^"flip" = bu(mu) - bu(mu)'_"flip"$ (similarly for $Delta bu(theta)$).
The weight $lambda(s)$ ramps linearly from $0$ to $4.0$ between training steps
#box[$s=$ 2,000] and 12,000.


== Temporal Stability Metrics <sec-method-metrics>

While the Normalized Mean Error (NME) quantifies spatial accuracy, it is blind to the temporal behavior of
the error. A jitter metric, the Normalized Mean Flicker (NMF)
@micaelli2023-deep-equilibrium-models, is the frame-to-frame
increment $norm(bu(e)_(l n) - bu(e)_(l-1, n))_2$ of the landmark error
$bu(e)_(l n) = bu(mu)_(l n) - bu(y)_(l n)$, normalized by the face size
#box[$d_l = sqrt(h_l w_l)$] and aggregated in the RMS over landmarks and frames
(mean over clips). NMF is a _precision_ measure where
all temporal change is penalized equally and constant offsets are ignored.
As a first-difference statistic it cannot separate worst-case
jitter, a sign-alternating error, from smooth drift.

We therefore generalize NMF to a multi-timescale stability profile based on the Allan
variance @sheimy2008-allan-variance-inertial-sensors. With the normalized error
$tilde(bu(e))_(l n) = bu(e)_(l n) slash d_l$ and its block averages
#box(inset: (top: 0.6mm))[$overline(bu(e))_(l n) (m) = 1/m sum_(i=l)^(l+m-1) tilde(bu(e))_(i n)$] over $m$
consecutive frames, the#linebreak(justify: true)
#pagebreak()
normalized Allan variance compares
successive blocks,
$
    #h(-6mm)
    sigma^2 (m) =
    1/(2 (L - 2 m+1)) #h(-2.5mm) sum_(l=1)^(L - 2 m+1) #h(-2.5mm)
    norm(overline(bu(e))_(l+m) (m) - overline(bu(e))_(l) (m))_2^2
    .
$
Then, the _Normalized Allan Deviation_ (NAD)
is the RMS of $sigma(m)$ over landmarks (mean over clips).
$"NAD"(1) approx "NMF"slash sqrt(2)$, up to normalization convention. NMF is thus the shortest-timescale point of the NAD curve, and
its conflation of jitter and drift becomes precise: For the simple model
#box[$tilde(bu(e))_l = bu(j)_l + bu(d)_l$] where $"NAD"^2(m) = sigma_j^2 slash m + m^2 norm(bu(v))^2
slash 2$ with zero-mean white jitter $bu(j)_l$ of
per-frame variance $sigma_j^2$ and linear drift $bu(d)_l = l dot bu(v)$, block averaging attenuates
the jitter contribution as $sigma_j^2 slash m$, whereas linear drift
survives.
So, the NMF alone cannot distinguish between jitter and drift.

However, plotted against $m$ on log-log axes, the NAD curve diagnoses the random processes composing
the error (non-unique relation to power spectral density) @sheimy2008-allan-variance-inertial-sensors.
Asymptotically, _white jitter_ decays with slope $-1 slash 2$,
_long-range correlations_ ($1 slash f$ noise) flatten the curve into the zero-slope _bias-instability_ floor,
an _error random walk_ rises with slope $+1 slash 2$,
and _linear drift_ with $+1$.

The type of detector needs to be distinguished.
An absolute-coordinate predictor re-anchors to the image at every frame and
has bounded drift by
construction, whereas a delta predictor integrates
delta errors $tilde(bu(u))_l = tilde(bu(e))_l - tilde(bu(e))_(l-1)$ (exactly as measured by NMF) across frames.
Uncorrelated delta noise then accumulates into an _error random walk_.
A persistent delta bias drifts faster (slope $+1$), while anticorrelated
delta noise---visual feedback correcting accumulated errors---flattens and ultimately bounds the drift.
This produces a hump in the graph with the maximum at the error correlation time $m_"c"$.
Then, a falling slope separates a zero-reverting ($-1 slash 2$) error from the
bias-instability floor ($0$ slope).
See the supplement for further details.

#figure(
    {
        image("figures/fig_hidden_state_carry.pdf")
        v(-1mm)
    },
    caption: [
        Left, per-frame WFLW-V NME with (QLOT-final) and without (A1) recurrent
        state carry; right, the corresponding NAD curves; A2 and A6 shown for comparison.
        QLOT-final has much lower jitter and drift.
        Whereas A1 integrates freely (slope $+0.42 =$ free random walk for $m in [1, 10]$), QLOT-final's slower
        rise ($+0.22$) and later peak ($m_"c" = 22 > 10$) reveal memory-bounded,
        self-correcting dynamics.
    ],
    scope: "parent",
    placement: top,
) <fig:hidden-state-carry>

= Experiments <sec-experiments>

We evaluate QLOT-full (@sec-method) on static accuracy (WFLW, 300-W, FaceSynthetics)
and video stability (WFLW-V) against pre-specified claims.
Seven ablation variants that change one component of the model test each claim:
On the *recurrent memory* (_A1_, hidden-state reset), the *update operator*
(_A2_, parameter-matched local-only GRU; _A3_, standard GRU; _A4_, mean-only WMR),
the *supervision recipe* (_A5_, $L_2$ only; _A6_, no temporal GNLL), and the *query
encoding* (_A7_, geometric Fourier). All configurations are trained from scratch
under an identical protocol, mean over 3 seeds.
Checkpoints are selected on the validation sets by lowest static NME,
then choosing lowest NMF among the best three.
Detailed protocols, full
ablation tables, and per-claim statistical analysis are deferred to the
supplementary material. We report only the prominent results here.

== Results

#figure(
    {
        set text(size: 0.72em)
        pretty-table(
            columns: 17,
            align: (left,) + (center,) * 16,
            header: (
                table.cell(rowspan: 2, align: horizon)[*Method, Year*],
                table.cell(rowspan: 2, align: horizon)[*Params*],
                table.cell(rowspan: 2, align: horizon)[*MACs*],
                table.vline(stroke: 0.5pt),
                table.cell(colspan: 2)[*300-W* (NME)],
                table.vline(stroke: 0.5pt),
                table.cell(colspan: 7)[*WFLW* (NME)],
                table.vline(stroke: 0.5pt),
                [*Synth*],
                table.vline(stroke: 0.5pt),
                table.cell(colspan: 4)[*WFLW-V*],
                table.hline(stroke: 0.5pt),
                [Com],
                [Chal],
                [Full],
                [Pose],
                [Expr],
                [Ill],
                [Mkp],
                [Occl],
                [Blur],
                [NME],
                [NME#sub[E]],
                [NME#sub[H]],
                [NMF#sub[E]],
                [NMF#sub[H]],
            ),
            [RwR @micaelli2023-deep-equilibrium-models, 2023],
            [21.8 M],
            [--],
            [--],
            [--],
            [*3.92*],
            [6.86],
            [*3.94*],
            [4.17],
            [*3.75*],
            [4.77],
            [4.59],
            [--],
            [1.24],
            [2.30],
            [82.74],
            [172.95],
            [#text(0.9em)[Inf3DLmks] @chandran2024-infinite-3d-landmarks, 2024],
            [>100 M#super[\*]],
            [--],
            [2.89],
            [5.71],
            [--],
            [--],
            [--],
            [--],
            [--],
            [--],
            [--],
            [--],
            [--],
            [--],
            [--],
            [--],
            [TUFA @xia2025knowledge-discrepancies, 2025],
            [36.0 M],
            [17.6 G],
            [*2.59*],
            [*4.45*],
            [3.93],
            [*6.48*],
            [4.11],
            [*3.82*],
            [3.81],
            [*4.68*],
            [*4.53*],
            [--],
            [--],
            [--],
            [--],
            [--],
            table.hline(stroke: 0.5pt),
            [QLOT-final (ours)],
            [*1.98 M*],
            [*0.65 G*],
            [3.03],
            [5.37],
            [3.93],
            [6.65],
            [4.07],
            [3.88],
            [3.90],
            [4.75],
            [4.60],
            [1.62],
            [*0.83*],
            [*1.30*],
            [*77.7*],
            [*138.5*],
        )
    },
    caption: [
        Comparison with published face-landmark methods. Static NME (%, inter-ocular) on 300-W, WFLW, and
        FaceSynthetics (%, face-size). WFLW-V video NME (%, face-size) and
        NMF on our test split (easy, hard), each video truncated to 120 frames.
        MACs (Multiply-Accumulate operations) are the per-frame tracking cost.
        [--] = not reported. #super[\*]Estimated from paper.
    ],
    placement: top,
    scope: "parent",
) <tab:sota>


#figure(
    image("figures/fig_annotated_samples.pdf"),
    caption: [
        Qualitative results with predicted 95% confidence ellipses.
    ],
    placement: auto,
) <fig:qualitative-results>

*Recurrent memory (A1) is the dominant factor.*
Resetting the hidden state at
every frame, while still carrying the previous coordinates and
covariance, nearly doubles flicker (NMF 211.4 vs. 109.5, +93%), raises video
NME by 68%, and elevates NAD at all timescales, at essentially unchanged static
accuracy.
@fig:hidden-state-carry shows why: The variants are indistinguishable
at frame 0 (4 iterations each). The gap opens at frame 1---the first single-iteration frame---and persists for the rest of the
clip.
Thus, the hidden state
is not a smoother applied to otherwise converged estimates, but what lets the tracker stay
locked in one iteration per frame, and recover quickly when lost.
Coordinate forwarding tells the model _where_ to focus in the next frame,
the hidden state carries _past information of the track_---without it, every frame starts cold.

*Update operator (A2--A4).* Removing cross-landmark communication (A2) is the
most damaging architectural change after A1 (NMF +10.5%, video NME +31%, WFLW
NME +8.7%, significant in every seed). Landmarks do not move independently, and
the mixer exploits this structure.
Our GRU formulation is at least as good as a standard GRU variant (A3).
Removing the WMR dispersion descriptor (A4) degrades nothing and slightly
_improves_ stability (NMF #sym.minus\1.3%, video NME #sym.minus\2.1%, both significant).
The routed temporal states may carry similar information, and variance
estimation is itself noise-prone. We adopt the simpler mean-only WMR (1.7%
fewer parameters) and call it _QLOT-final_.

*Supervision (A5, A6) and query encoding (A7).* Dropping the temporal GNLL
terms (A6) raises NMF by 2.4% and NAD mostly at short timescales (@fig:hidden-state-carry),
with static NME unchanged. A pure $L_2$ loss (A5) is nearly
identical (+2.6%) to this. So, the temporal terms account for essentially all of the
supervision gain, an effect that is real but an order of magnitude smaller
than state carry. PMPE (A7) is a null result on all metrics and is retained for
flexibility. Checkpoint selection is itself a methodological variable. Purely
NME-based selection yields worse trackers (NMF +0.8%, video NME
+1.9%) at identical static accuracy, so we recommend reporting the selection criterion
explicitly in video-landmark work.

*Comparison to prior work.* @tab:sota compares QLOT-final with three published
methods: At 1.98M parameters and 0.65G MACs per tracked frame
(1 iteration, 98 landmarks), QLOT uses an
order of magnitude fewer parameters than prior work at competitive static
accuracy.
It runs at up to 680 FPS on a single NVIDIA RTX 3080 GPU, where it shows
almost linear inference cost in the number of landmarks tracked.
@fig:qualitative-results displays some annotated images.

Against RwR @micaelli2023-deep-equilibrium-models, the strongest
published video baseline, itself already below every EMA-smoothed detector it
was compared against, QLOT lowers NMF by 15% overall and by 20% on the hard
split.
The test protocol is near-identical.
While RwR calculates jitter on the whole WFLW-V dataset,
as they can only train on images, we evaluate on our test subset (75 hard, 75 easy videos).
See the supplement for a full protocol discussion.

= Conclusion <sec-conclusion>


We presented QLOT, a recurrent architecture that formulates
face landmark tracking as query-conditioned learned optimization.
A per-landmark hidden
state carried across frames and exchanged, in addition to RAFT-like correlation features, through a Write-Mix-Read
block keeps the tracker locked with a single iteration per frame. Ablations
identify recurrent state carry as the dominant source of temporal stability, with
temporal GNLL supervision adding a smaller gain. Canonical 3D
query points enable joint training on heterogeneous image and video data. At 1.98M
parameters and 0.65G MACs per frame, QLOT-final achieves competitive-to-SOTA
static accuracy while reducing flicker by 15% over the strongest video
baseline, running at up to 680 FPS on a consumer GPU. The proposed Normalized
Allan Deviation generalizes NMF to a multi-timescale profile that separates jitter
from drift, providing a principled stability diagnostic for future tracking
research.
