#import "template.typ": indent, paper
#import "@preview/retrofit:0.2.0": backrefs
#import "utils.typ": TT, bu, pretty-table

#let authors = (
    (name: "Dominik Gschwind", affl: ("OST",), email: "dominik.gschwind@ost.ch"),
    (name: "Martin Weisenhorn", affl: ("OST",), email: "martin.weisenhorn@ost.ch"),
    (name: "Hannes Badertscher", affl: ("OST",), email: "hannes.badertscher@ost.ch"),
    // (name: "Second Author", affl: ("Institution2",), email: "secondauthor@i2.org"),
)

#let affils = (
    "OST": (
        institution: [Eastern Switzerland University of Applied Sciences (OST)],
        location: "Rapperswil-Jona, Switzerland",
    ),
)

#show: paper.with(
    title: [_QLOT:_ Queried Learned Optimization for Face Landmark Tracking\
    _Supplementary Material_],
    authors: (authors, affils),
    keywords: (),
    date: auto,
    abstract: none,
    bibliography: bibliography("bibliography/citations.bib"),
    appendix: [
    ],
    mode: "preprint",
    track: "algorithms",
    pagenumbers: true,
    paper-id: "742",
)

// Append hyperref pagebackref-style page links to each reference.
#show: backrefs.with(
    format: links => [~#links.join(", ")],
    read: path => read(path),
)

= Ablations

We evaluate QLOT against a set of pre-specified claims, each paired with a controlled
counterfactual that changes exactly one component relative to the full model (_QLOT-full_).
All ablations are trained from scratch with the same data mixture, augmentations, optimizer,
learning-rate schedule, iteration curriculum, and checkpoint-selection rule. Except for A2's
approximately parameter-matched local updater, where a component is removed, the freed capacity
is left un-replaced rather than padded with additional parameters. Parameter counts, rounded to
the nearest 0.01M, are reported alongside. The claims and their tests are:

/ C1: *Efficiency-accuracy-stability.* QLOT attains competitive landmark accuracy with
    substantially lower model cost and improved temporal stability relative to direct
    coordinate-regression baselines. *Test:* external comparison under a matched protocol
    (accuracy, temporal stability, parameters, MACs).

/ C2: *Recurrent memory.* Carrying a recurrent hidden state across frames improves temporal
    stability beyond coordinate initialization alone. *Test:* (_A1, no hidden state carry_) reset the hidden state to
    zero at every new frame while still prefilling the previous frame's predicted coordinates
    and covariance, during training and evaluation.

/ C3: *Update operator.* The GRU-like gated update with lightweight Write-Mix-Read (WMR)
    communication performs at least as well as PyTorch's _`GRUCell` updater_, a _local-only_
    updater, and a _mean-only WMR_ variant. *Tests:*
    (_A2_) approximately parameter-matched per-landmark GRU (our variant) while _removing the WMR block_;
    (_A3_) a _standard GRU_ in place of our GRU variant;
    (_A4_) WMR with the _dispersion descriptor and slot-energy decomposition removed_.

/ C4: *Uncertainty supervision.* A full 2D covariance output with spatial and temporal GNLL
    losses reduces jitter, without harming accuracy, over a simple $L_2$ coordinate loss.
    *Tests:* (_A5_) covariance head and all GNLL terms removed, trained with _$L_2$ only_; (_A6_)
    _temporal GNLL terms removed_, spatial GNLL retained.

/ C5: *Query encoding.* The modified Phase-Modulated Positional Encoding (PMPE) improves performance over a
    standard geometric-progression Fourier encoding. *Test:* (_A7_) swap the PMPE-variant for
    the Fourier-only encoding of @xia2025knowledge-discrepancies at matched width.

#indent A1 isolates recurrent *memory* (not tracker initialization, since coordinates are still carried);
A2--A4 isolate the *update operator* under identical feature extractors, hidden-state width, and
training budget; A5--A6 isolate the *supervision recipe*; A7 isolates the *query encoding*. This
yields one reference plus seven single-change ablations (eight configurations), each targeting
one claim.

*Conventions.* All ablations report means over 3 seeds on identical data manifests.
Significance is assessed with a paired bootstrap (10,000 resamples) over pooled seed-video pairs for temporal
metrics and pooled seed-image pairs for static NME; n.s. (not significant) indicates $p>0.05$.
Static NME is inter-ocular (distance between outer eye corners) normalized on WFLW/300-W
and face-size (square-root of label bounding box area) normalized on FaceSynthetics.
WFLW-V metrics are face-size normalized on the 150-clip test split
(75 easy / 75 hard),
4 refinement iterations at frame 0, and 1 thereafter.

*Checkpoint selection rule.* Among the top-3 validation candidates ordered by NME, select
the one with the lowest validation NMF.

#let table-header(..args) = table.header(
    table.hline(stroke: 1pt),
    ..args,
    table.hline(stroke: 1pt),
)

#figure(
    text(size: 0.9em,
    pretty-table(
        columns: 11,
        align: (x, y) =>  if x > 0 and y > 0 {center} else {left},
        header: (
            [Config],
            [Params],
            [WFLW],
            [300-W],
            [Synth],
            [V-NME],
            [NMF#sub[E]],
            [NMF#sub[H]],
            [NAD(1)],
            [NAD(4)],
            [NAD(16)],
        ),
        [A0 (QLOT-full)],
        [2.01M],
        [3.934],
        [3.487],
        [1.617],
        [1.089],
        [78.97],
        [140.02],
        [3.02],
        [4.14],
        [5.15],
        [A1 (no state carry)],
        [2.01M],
        [3.938],
        [3.498],
        [1.646#super[†]],
        [1.825#super[†]],
        [170.72#super[†]],
        [252.11#super[†]],
        [5.84#super[†]],
        [10.37#super[†]],
        [11.83#super[†]],
        [A2 (local-only)],
        [1.95M],
        [4.277#super[†]],
        [3.626#super[†]],
        [1.767#super[†]],
        [1.425#super[†]],
        [85.14#super[†]],
        [156.83#super[†]],
        [3.34#super[†]],
        [4.77#super[†]],
        [6.35#super[†]],
        [A3 (standard GRU)],
        [2.01M],
        [3.947],
        [*3.431*#super[†]],
        [1.646#super[†]],
        [1.107#super[†]],
        [78.67],
        [139.62],
        [3.01],
        [4.13],
        [5.19],
        [A4 (mean-only WMR)],
        [1.98M],
        [3.934],
        [3.493],
        [1.619],
        [*1.066*#super[†]],
        [*77.69*#super[†]],
        [*138.52*#super[†]],
        [*2.99*#super[†]],
        [*4.07*#super[†]],
        [*5.08*#super[†]],
        [A5 ($L_2$ only)],
        [1.99M],
        [3.918],
        [3.493],
        [1.611#super[†]],
        [1.088],
        [80.47#super[†]],
        [144.16#super[†]],
        [3.10#super[†]],
        [4.19#super[†]],
        [5.18],
        [A6 (no temporal GNLL)],
        [2.01M],
        [*3.914*],
        [3.505#super[†]],
        [*1.597*#super[†]],
        [1.095],
        [80.34#super[†]],
        [143.95#super[†]],
        [3.10#super[†]],
        [4.20#super[†]],
        [5.17],
        [A7 (Fourier PE)],
        [2.03M],
        [3.930],
        [3.501],
        [1.617],
        [1.092],
        [78.00#super[†]],
        [140.52],
        [3.02],
        [4.13],
        [5.13],
    )),
    caption: [
        Ablation results, mean over 3 seeds.
        WFLW/300-W: static NME (%, inter-ocular); Synth: FaceSynthetics
        static NME (%, face-size); V-NME: WFLW-V video NME (%, face-size);
        NMF#sub[E] / NMF#sub[H]: NMF on
        the 75 easy / 75 hard WFLW-V test clips truncated to 120 frames; NAD($m$) values are scaled by $10^3$.
        Bold marks the best point estimate in each metric column. #super[†] marks a significant
        difference vs. A0 from paired bootstrap.
    ],
    placement: auto,
    scope: "parent",
) <tab:ablations>


== Results

Quantitative results are in @tab:ablations and @tab:checkpoint-selection.

=== Temporal memory enables one-iteration-per-frame tracking (C2)

A central design claim is that QLOT tracks with a single refinement iteration
per frame because it carries a recurrent hidden state,
not merely the previous coordinates.
Ablation A1 tests this directly.
It is architecturally identical to the full model but resets the hidden state to
zero at every new video frame while still receiving the previous frame's predicted
coordinates and covariance as the starting estimate.

The effect is large and unambiguous.
On WFLW-V, A1's normalized mean flicker nearly
doubles (211.4 vs. 109.5, +93%, $p<0.05$ paired),
video NME rises by 68%
(1.83% vs. 1.09%),
and NAD is elevated across the entire measured range
(NAD(1) +93%, NAD(4) +150%, NAD(16) +130%).
Static NME is unchanged on WFLW and 300-W
(3.94% vs. 3.93% and 3.50% vs. 3.49%, respectively; both n.s.),
while FaceSynthetics shows a small but significant +1.8% degradation.
At frame 0, where both models run four refinement iterations,
the two are indistinguishable
(1.41% vs. 1.42% NME, n.s.).
The gap opens at frame 1, the first one-iteration frame, and
persists until the end of the 120-frame clips (results are averaged over clips).
The hidden state is therefore not a smoother applied on top of an
otherwise-converged tracker.
It is what allows the tracker to stay locked in one frame, or when lost, recover more quickly.
Coordinate prefill tells the model _where_ to focus in the next frame,
the hidden state carries _what the track has learned_ about appearance, motion,
and local evidence.

=== Cross-landmark communication and the update operator (C3)

We ablate the update operator along three axes.
Removing cross-landmark communication while expanding the local GRU to approximately
match A0's parameter count (A2) is the most damaging architectural change after A1:
NMF +10.5%, video NME +31%,
and WFLW NME +8.7% (4.28% vs. 3.93%),
all significant.

Replacing our GRU-variant with a standard GRU
(A3, parameter-matched) leaves jitter unchanged (NMF #sym.minus\0.3%, n.s.)
and video NME marginally worse (+1.7%, small but significant in the pooled comparison),
while slightly _improving_ static 300-W accuracy (#sym.minus\1.6%, significant).
Thus, our GRU variant matches the standard recurrent cell on NMF and NAD, while A3 incurs
a small video-NME cost and improves 300-W accuracy; a broad advantage for either update
formulation is not demonstrated.

=== The dispersion descriptor is unnecessary (C3)

The WMR mixer's slot representation includes a dispersion descriptor in addition to the weighted mean.
Removing it as well as the slot-energy decomposition (A4) while keeping routing, slot
attention, and the gated update the same does not degrade any metric. To the contrary, temporal
stability is _improved_:
NMF −1.3% and video NME −2.1% (both significant across seeds),
NAD(1)/(4)/(16) all lower, with static accuracy unchanged
(WFLW 3.934% vs. 3.934%, n.s.).

We theorize that the spread information in the dispersion is compensated
by the routed temporal states.
Additionally, noise susceptibility is increased since
variance computation is a high-pass filter.
Other applications adopting the WMR block, however, may still
benefit from the slot disagreement and normalization mechanisms.

We adopt the simpler mean-only WMR, and call this configuration *QLOT-final*.
It is 1.7% smaller, marginally cheaper in MACs, and measurably more stable.

=== Supervision: Temporal GNLL drives stability (C4)

Training with a full 2D covariance and spatial+temporal GNLL losses reduces jitter at
no accuracy cost.
Removing the temporal GNLL terms (A6) raises NMF by 2.4% and NAD(1)/(4) significantly,
while static effects are small and mixed: WFLW is unchanged, FaceSynthetics improves by
1.2% significantly, and a +0.5% shift on 300-W barely reaches significance with
inconsistent signs across seeds;
removing the covariance head and all GNLL terms in favor of a pure L2 coordinate loss
(A5) produces a nearly identical +2.6% NMF, indicating the temporal terms account for essentially
all of the gain.
The effect is confined to short timescales (NAD(16) is unaffected for both A5 and A6).

For context, we place these NMF values next to the strongest published baseline. RwR
@micaelli2023-deep-equilibrium-models reports an NMF of 127.9 on WFLW-V (82.7 easy /
173.0 hard), already below every EMA-smoothed detector it was compared against,
whereas A5 and A6 attain 112.3 and 112.1 (80.5/80.3 easy, 144.2/144.0 hard) and
QLOT-full 109.5 (79.0/140.0)---12--14% lower overall, with the largest margin on the
hard split (19% vs. 4--5% on easy).

The test protocols are close enough to support
this reading. The metric and the official easy/hard video partition are
identical, our 150 held-out test videos are a stratified draw from the same
1,000-video pool that RwR evaluates in full, and our fixed 120-frame windows cover
all but at most the final 31 frames of each 120--151-frame video.
RwR's whole-dataset evaluation averages over more data from the same
distribution.
The remaining difference lies in training, not testing, and is architectural.
RwR is a purely per-frame model, so it
can only be trained on static images.
Our models, by contrast, train on WFLW-V clips from videos disjoint from the test
set, so part of the gap may reflect in-domain video adaptation that RwR structurally
cannot exploit, rather than the tracking method alone.
Subject to that caveat, we read the comparison as QLOT being measurably more
stable than the strongest published baseline under near-identical test conditions,
rather than as a fully controlled head-to-head.

Within our own protocol-matched
ablations, the temporal-GNLL effect is real but an order of magnitude smaller than the
state-carry effect (C2), +2.4% NMF for A6 vs. +93% for A1.

=== Query Encoding (C5)

Replacing the PMPE-variant with a geometric-progression Fourier encoding
(A7) yields no measurable difference in aggregate metrics. Easy-split NMF improves slightly
(78.00 vs. 78.97, significant), without a corresponding hard-split or overall effect.
We therefore find no aggregate benefit for either encoding and retain PMPE for its flexibility.

=== Checkpoint Selection

We found checkpoint selection to be a first-class methodological
variable for tracking.
To quantify it, we re-selected the A1 checkpoints purely by
validation NME
(dropping the minimal-NMF-of-top-3 rule)
and evaluated them on the test set (@tab:checkpoint-selection).

NME-only checkpoints are worse on every temporal metric
(NMF 213.1 vs. 211.4, +0.8% overall, +0.8% on easy and +0.8% on hard;
video NME +1.9%; NAD higher at all timescales; all significant)
while WFLW is unchanged and 300-W improves slightly (3.485 vs. 3.498, significant).

#figure(
    {
        set par(justify: false)
        pretty-table(
            columns: 8,
            align: left,
            header: ([A1 checkpoints], [300-W], [WFLW], [V-NME], [NMF#sub[e]], [NMF#sub[h]], [NAD(4)], [NAD(16)]),
            [stability-aware (A1)],
            [3.498],
            [3.938],
            [1.825],
            [170.72],
            [252.11],
            [10.37],
            [11.83],
            [NME-only (A1-nme)],
            [3.485#super[†]],
            [3.938],
            [1.861#super[†]],
            [172.16#super[†]],
            [254.10#super[†]],
            [10.52#super[†]],
            [12.08#super[†]],
        )
    },
    caption: [
        Effect of the checkpoint-selection rule on the A1 ablation, mean over 3
        seeds (columns as in @tab:ablations, NAD scaled by $10^3$). Both rows are
        the *same three training runs*; only the selected checkpoint differs.
        #super[†] marks a significant difference between the two selection rules
        from paired bootstrap. Static accuracy is identical on WFLW, while all
        temporal metrics degrade under NME-only selection, on both
        the easy and hard subsets.
    ],
    placement: top,
    scope: "parent",
) <tab:checkpoint-selection>

== Model Complexity

QLOT-final has 1.98M parameters. Its computational cost depends on the number of
refinement iterations $I$ run per frame: 872.79M MACs per frame for $I = 4$
iterations (used at frame 0 of each clip) and 648.87M MACs per frame for $I = 1$
(used for all subsequent frames), estimated from the architecture.
The backbone, FPN, correlation volume generation,
and query encoding run only once per frame.
The marginal cost
of each additional refinement iteration is about 75M MACs. The
four-iteration initialization is about 1.35 times the steady-state
per-frame cost.
All MAC counts are at 224#sym.times\224 input resolution and 98 landmarks.
@fig:runtime shows the inference time plotted against the number of tracked landmarks.

#figure(
    image("figures/fig_runtime.pdf"),
    caption: [
        Inference time of QLOT via ONNX runtime on a single NVIDIA RTX 3080 GPU (with and
        without CUDA graph capture) on the left, and on CPU (AMD Ryzen 7 3700X) on the right.
        CUDA graph capture mostly eliminates the kernel launch overhead and thus isolates the
        raw computation time of the model.
        All graphs show the near-linear scaling in the number of tracked landmarks.
    ],
    placement: auto,
    scope: "parent",
) <fig:runtime>



= Reproducibility

The code uses AR(1) dynamics correction for the delta GNLL loss with
$rho = 0.74$. This is a remnant of an earlier implementation
in which the coordinates were detached instead of the covariance.
Since the covariance is now detached (and the coordinates live), the
correction is not necessary. $rho$ can be absorbed into the loss weight
(so the effective weight is larger with AR(1) correction).

We use the HGNetV2 implementation from the `timm` (PyTorch Image Models) library @wightman-torch-image-models
as the backbone.

= Analysis of the Normalized Allan Deviation <sec-supp-nad>

This section derives the NAD signatures used in the _Temporal Stability Metrics_ section, gives
finite-range slope benchmarks, the estimation protocol, and the fitted results.

== Setup and Exact Identities

#let NAD = math.op("NAD")

Recall $tilde(bu(e))_(l n)$ is the face-size-normalized
landmark error for landmark $n$ in frame $l$,
$
  tilde(bu(e))_(l n) = (bu(p)_(l n) - bu(y)_(l n)) slash d_l,
$
with position estimate $bu(p)$, label $bu(y)$, and $d = sqrt(h w)$
of the#linebreak(justify: true)
labels' $h$$times$$w$ bounding box.
Assuming we take the RMS over landmarks for $NAD$ (or equivalently, mean over landmarks for $NAD^2$),
we drop the landmark index $n$.
Then,
$
    #h(-4.5mm)
    NAD^2(m) #h(0.3mm) = #h(0.3mm) 1/(2(L-2m+1))#h(-3mm)sum_(l=1)^(L-2m+1)#h(-3mm)norm(overline(bu(e))_(l+m)(m) - overline(bu(e))_l (m))_2^2
$ <eq:navar>
(cf. @sheimy2008-allan-variance-inertial-sensors, Eq. 6)
with block averages $overline(bu(e))(m)$ over $m$ frames,
$
    overline(bu(e))_l (m) = 1/m sum_(i=l)^(l+m-1) tilde(bu(e))_i.
$

#indent A delta predictor (like QLOT) outputs the position increment $Delta bu(p)_l = bu(p)_l - bu(p)_(l-1)$
instead of the absolute position $bu(p)$. Thus, we define the normalized error increment
$
tilde(bu(u))_l = tilde(bu(e))_l - tilde(bu(e))_(l-1)
= 1/(d_l) (Delta bu(p)_l - Delta bu(y)_l)
$
when $d_l = d_(l-1)$ (a good approximation for successive frames).
Note that NMF actually measures $tilde(bu(u))_l$,
$
  "NMF"_(l) = norm(bu(e)_l - bu(e)_(l-1))_2/d_l = norm(tilde(bu(e))_l - tilde(bu(e))_(l-1))_2 = norm(tilde(bu(u))_l)_2
$
and is aggregated in the RMS over frames.
We can also show that $NAD(1) = "NMF"slash sqrt(2)$ since $overline(bu(e))_l (1) = tilde(bu(e))$
and 

$
    #h(-4mm)
    NAD^2(1) = 1/(2(L-1)) sum_(l=1)^(L-1) norm(tilde(bu(e))_(l+1) -
    tilde(bu(e))_l)_2^2
    = "NMF"^2/2.
$

#figure(
    pretty-table(
        columns: 4,
        header: ([*Process*], [*Model / PSD*], [$"NAD"^2(m)$], [*Asymptotic log-log Slope*]),
        [white jitter],
        [$bu(p) = bu(y) + bu(epsilon)_j$],
        [$sigma_j^2 slash m$],
        [$-1 slash 2$],

        [bias instability],
        [$S(f) prop 1 slash f$],
        [plateau $approx (0.664 B)^2$],
        [$0$],

        [free random walk],
        [$Delta bu(p) = Delta bu(y) + bu(epsilon)_u$],
        [$sigma_u^2 (2m^2+1) slash (6m)$],
        [$+1 slash 2$],
        [linear drift],
        [$tilde(bu(e))_l = l dot bu(v)$],
        [$m^2 norm(bu(v))^2 slash 2$],
        [$+1$],

        [bounded walk, AR(1)],
        [$tilde(bu(e))_(l+1) = rho tilde(bu(e))_l + bu(epsilon)_(l,d)$],
        [@eq-ar1],
        [$+1slash 2$ for $m << tau_c$, $-1 slash 2$ for $m >> tau_c$],
    ),
    caption: [NAD signatures of canonical error processes. $bu(epsilon)$ is zero-mean
    white noise of variance $sigma^2$. $bu(v)$ is a non-zero constant.],
    scope: "parent",
    placement: auto,
)<tab-nad-signature>

== Ideal NAD Signatures

*White jitter.*
Assume the position estimate decomposes into the label's trajectory plus
white zero-mean estimation noise, $bu(p) = bu(y) + bu(epsilon)_j$,
whose pixel variance scales with face size,
$EE norm(bu(epsilon))^2 = sigma_j^2 d^2$.
Then, with independent noise-free labels and scales,
the normalized error $tilde(bu(e))$ is white and has the same variance
$EE norm(tilde(bu(e)))^2 = sigma_j^2$.
As a result, successive block averages share no samples,
and we obtain jitter of total variance
(cf. @sheimy2008-allan-variance-inertial-sensors, Eq. 17)
$
"NAD"^2(m) = "Var"(overline(bu(e))(m)) = sigma_j^2 slash m.
$
Independent label noise would add its own $sigma_"GT"^2 slash m$ term.
A log-log NAD curve of pure white jitter therefore has a slope of $-1 slash 2$.

*Bias instability.*
Assume instead that the normalized error contains a slowly fluctuating bias
with a power spectral density (PSD) of $S(f) prop 1 slash f$ (so-called _flicker noise_)
for $f <= f_0$ and zero above (cf. @sheimy2008-allan-variance-inertial-sensors, Eq. 18),
with $B^2$ the flicker power density.
In the time domain, such noise has no characteristic timescale.
Its autocorrelation decays only logarithmically, and it is equivalently
viewed as a superposition of AR(1)-type modes with time constants
spread uniformly in log-time.
Block-averaging $m$ frames therefore reduces the variance only
by $log m$, and, as common-mode wander, is cancelled by NAD's block differences.
The result is independent of $m$
and yields the plateau of
@sheimy2008-allan-variance-inertial-sensors, Eq. 19,
$
    "NAD"^2(m) = (2 ln 2 slash pi) B^2 approx (0.664 B)^2.
$
Thus, a flat region (slope $0$) in the log-log NAD plot indicates
long-range correlated error that temporal averaging cannot reduce:
the _bias instability floor_.

*Free random walk.*
When the position increment decomposes into the label increment plus
white zero-mean estimation noise,
$Delta bu(p)_l = Delta bu(y)_l + bu(epsilon)_l$,
whose pixel variance scales with face size,
$EE norm(bu(epsilon)_l)^2 = sigma_u^2 d_l^2$.
Then, with noise-free label increments and independent scales,
$tilde(bu(u))_l = tilde(bu(epsilon))_l slash d_l$ is white
with the same variance $EE norm(tilde(bu(u))_l)^2 = sigma_u^2$,
and it is integrated into a free error random walk,
$tilde(bu(e))_l = sum_(k <= l) tilde(bu(u))_k$.
The difference of successive block averages is
$
    overline(bu(e))_(l+m)(m) - overline(bu(e))_l (m)
    & = 1/m sum_(i=l)^(l+m-1) sum_(k=i+1)^(i+m) tilde(bu(u))_k \
    & = 1/m sum_(j=1)^(2m-1) w_j tilde(bu(u))_(l+j)
$ <eq:diff-random-walk>
with triangular weights $w_j = min(j, 2m - j)$.
Since the increments are white, inserting @eq:diff-random-walk into @eq:navar yields
$
    "NAD"^2(m) & = sigma_u^2/(2m^2) sum_j w_j^2 = sigma_u^2/(2m^2) dot m(2m^2+1)/3 \
               & = sigma_u^2 (2m^2+1)/(6m),
$
recovering the asymptote $sigma_u^2 m slash 3$
(cf. @sheimy2008-allan-variance-inertial-sensors, Eq. 21, with $K^2 = sigma_u^2$) and
$"NAD"^2(1) = sigma_u^2 slash 2$.
A log-log NAD curve of a free random walk therefore has an asymptotic slope of $+1 slash 2$.

*Linear drift.*
Assume a constant increment bias
$ tilde(bu(u))_l = bu(v), $
which integrates into a linear drift of the normalized error,
$ tilde(bu(e))_l = tilde(bu(e))_0 + l dot bu(v). $
For a delta predictor, this is the failure mode of an uncompensated
velocity bias. The process is deterministic
and $"NAD"(m)$ measures it exactly for any clip length.
Block averaging preserves the drift exactly,
$ overline(bu(e))_l (m) = tilde(bu(e))_0 + (l + (m - 1) / 2) bu(v), $
so the difference of successive block averages is independent of the
window position $l$,
$ overline(bu(e))_(l+m)(m) - overline(bu(e))_l (m) = m bu(v), $
giving (cf. @sheimy2008-allan-variance-inertial-sensors, Eq. 23)
$ "NAD"^2(m) = (m^2 norm(bu(v))^2) / 2. $
$"NAD"^2(1) = norm(bu(v))^2 slash 2$ and $"NMF"^2 = norm(bu(v))^2$.
Temporal averaging is powerless against drift. The block difference grows
linearly in $m$, the log-log slope $+1$ is the steepest signature in
@tab-nad-signature, and even a small bias $bu(v)$ eventually dominates
the considered error processes at large $m$.

*Bounded walk, AR(1).*
Assume the normalized error reverts to zero as a stationary AR(1) process,
$ tilde(bu(e))_(l+1) = rho tilde(bu(e))_l + bu(epsilon)_(l,d), $
driven by zero-mean white innovations $bu(epsilon)_(l,d)$ with
$EE norm(bu(epsilon)_(l,d))^2 = sigma_d^2$
(pixel variance $sigma_d^2 d^2$, labels noise-free and scales
independent, as before).
Squaring the recursion and taking expectations gives the stationary
variance
$ sigma^2 = EE norm(tilde(bu(e)))^2 = sigma_d^2 slash (1 - rho^2), $
and the autocovariance
$ 
EE chevron.l tilde(bu(e))_l, tilde(bu(e))_(l+h) chevron.r = sigma^2 rho^(|h|)
$
decays with correlation time $tau_c = -1 slash ln rho$ frames
($rho = e^(-1 slash tau_c)$).
Unlike the free random walk, the geometric pull toward zero dissipates
integrated jitter as fast as the innovations inject it, so the error stays
bounded at $sigma^2$ regardless of the clip runtime $L$.
In the delta domain, this restoring force appears as anticorrelated
increments.
Defining $gamma_u (h)$ as the delta-autocovariance
$
  gamma_u (h) = EE chevron.l tilde(bu(u))_l, tilde(bu(u))_(l+h) chevron.r,
$
at lag 1 this evaluates to
$
  #h(-5mm)
  gamma_u (1) = EE chevron.l tilde(bu(e))_(l) - tilde(bu(e))_(l-1), tilde(bu(e))_(l+1) -
  tilde(bu(e))_l chevron.r
  = - sigma^2_d (1-rho)/(1 + rho),
$
a fraction $-(1 - rho) slash 2$ of the increment variance
$sigma_u^2 = 2 sigma_d^2 slash (1 + rho)$.
Each frame partially cancels the previous frame's delta error.

Furthermore, defining $"NAD"$ in expectation,
$
  NAD^2 (m) = 1/2 EE norm(overline(bu(e))_(l+m)(m) - overline(bu(e))_l (m))^2,
$
it can be shown to equal $[V(m) - C(m)]slash m^2$
with
$
#h(-5mm)
V(m) = EE norm(m space.hair overline(bu(e))_l)^2 = sigma^2 [m + 2 sum_(h=1)^(m-1) (m-h) rho^h],
$
the variance of a length-$m$ block sum, and
$
  C(m) = EE chevron.l m space.hair overline(bu(e))_l, m space.hair overline(bu(e))_(l+m) chevron.r =  sigma^2 rho (1 - rho^m)^2 / (1 - rho)^2,
$
the covariance of two adjacent blocks.
Putting it together,
$
    "NAD"^2(m) = sigma_d^2/(m^2 (1-rho^2)) [ & m + 2 sum_(h=1)^(m-1) (m-h) rho^h \
                               & - rho (1-rho^m)^2 / (1-rho)^2 ].
$ <eq-ar1>
At $m = 1$ this gives exactly $"NAD"^2(1) = sigma_d^2 slash (1 + rho)$.
For slow zero-reversion
($rho -> 1$), this expression measures the driving variance $sigma_d^2$ directly.
For $m << tau_c$ (so $rho^m approx 1 - m slash tau_c$) the restoring force
has not yet acted, the innovations integrate freely, and @eq-ar1 reduces
to the free-walk curve driven by $bu(epsilon)_(l,d)$ with
$ "NAD"^2(m) approx sigma_d^2 (2m^2 + 1) slash (6m) approx sigma_d^2 m slash 3, $
a $+1 slash 2$ rise.
For $m >> tau_c$ the block averages decorrelate ($rho^m -> 0$) and
$
    "NAD"^2(m) approx sigma^2 (1+rho) / ((1-rho) m) = sigma_d^2 / ((1-rho)^2 m),
$
a $-1 slash 2$ tail whose level grows like $tau_c^2$
(since $1 - rho approx 1 slash tau_c$). Slower zero-reversion gives a
later, higher hump but the same decay slope.
The maximum sits at $m_c approx 1.9 tau_c$ (numerical).
The rising slope fitted over $[1, m_c]$ is damped relative to the free-walk
benchmark, since the fit's lower endpoint is fixed at $m = 1$ while the branch
rolls over at $m_c prop tau_c$.

== Finite-Range Slope Benchmarks

The slopes of @tab-nad-signature are asymptotic.
Least-squares fits over a finite range of $m$ yield smaller
effective slopes.
For example, the exact random-walk curve
$ "NAD"(m) = "NAD"(1) sqrt((2m^2+1) slash (3m)) $
fitted over the range $m in [1, 10]$ has slope $+0.43$, not $+1 slash 2$.
All fitted slopes in @tab-nad-fits are compared against finite-range benchmarks computed from the
exact curves over the reported branch ranges.

#figure(
    image("figures/fig_delta_error_acf.pdf", width: 55%),
    caption: [
        Autocorrelation of the face-size-normalized delta error
        $tilde(bu(u))_l = tilde(bu(e))_l - tilde(bu(e))_(l-1)$ on WFLW-V (150 clips,
        mean over 3 seeds, seed-to-seed spread below the marker size).
        QLOT-final's lag-1 correlation is negative ($-0.10$).
        Each frame's update partially cancels
        the previous frame's delta error, which damps the NAD rise.
        A1's lag-1
        correlation is slightly positive ($+0.08$) and turns negative only from
        lag $3$ onward, so its correction is weaker and acts at longer lags.
    ],
    placement: auto,
    scope: "parent",
) <fig-acf>

#figure(
    image("figures/fig_nad_vs_theory.pdf", width: 55%),
    caption: [
        Measured NAD curves of A1 and QLOT-final (solid, shaded bands are the relative error of @eq:rel-error, divided by $sqrt(150 dot 3)$) against the canonical error
        processes of @tab-nad-signature (dashed/dotted).
        Theory curves are anchored at the measured $"NAD"(1)$,
        except A1's free walk, which is scaled by $1.07$ for a better rise fit.
        Correlation times and mixture weights are fit by grid search, minimizing
        log-space RMSE over $m in [1, 60]$.
        A1's best-fit AR(1) ($tau_c = 8.5$) tracks the turnover but damps the
        rise to $+0.27$ (measured $+0.42$) and peaks too late
        ($m_c = 16$ vs. $10$). A continuous restoring force is not supported
        for A1.
        QLOT-final's AR(1) ($tau_c = 11$) matches the damped rise
        ($+0.25$ benchmark over $[1, 22]$ vs. $+0.22$ measured) and the peak
        time ($m_c approx 1.9 tau_c approx 21$ vs. $22$) but peaks too high and decays too steeply past the peak ($-0.22$ vs. $-0.13$).
        The two-AR(1) mixture
        $gamma(h) = sigma^2 [0.4 e^(-h slash 4.5) + 0.6 e^(-h slash 20)]$
        reproduces the rise ($+0.23$), peak ($m_c = 22$), and flat tail
        ($-0.12$), consistent with the cross-clip spread of correlation times
        (per-clip $m_c$ IQR $[11, 49]$).
    ],
    placement: auto,
    scope: "parent",
) <fig-nad-vs-theory>

== Estimation and Fitting Protocol

The per-clip relative
error of $"NAD"(m)$ is
$
    sigma(delta) = 1 slash sqrt(2(L slash m - 1))
$ <eq:rel-error>
(cf. @sheimy2008-allan-variance-inertial-sensors, Eq. 26).
With $L = 120$ frames per clip: 21% at $m = 10$, 34% at $m = 22$, and 71% at
$m = 60$.
Averaging over 150 clips reduces this to 1.7%, 2.7%, 5.8%,
and over 3 seeds (assuming approximately independent, homogeneous clips and seeds) to 1%, 1.6%, 3.3%, respectively.
We therefore fit the clip and seed averaged curves.

Slopes are least-squares fits of $log "NAD"(m)$ vs. $log m$ over locally stable intervals.
Two caveats: (i) ground-truth annotation noise adds an
independent $sigma_"GT"^2 slash m$ term, damping apparent rises.
The near-exact match of A1's rise to the free-walk benchmark shows this contribution is
negligible.
(ii) If $m_c$ varies across clips, the averaged falling flank is a mixture of
humps and its slope is biased toward $0$.
Per-clip curves confirm a large spread of peak locations
(QLOT-final: median $m_c = 22$, IQR $[11, 49]$; A1: median $12$, IQR $[8, 26]$),
so the averaged tails in @tab-nad-fits are conservative (biased toward $0$).
The per-clip rises are themselves concave, i.e., each clip is closer to a
single-timescale bounded walk.

== Fitted Results and Interpretation

#figure(
    pretty-table(
        columns: 5,
        column-gutter: 1em,
        header: ([*Model*], [*Rise (fit)*], [*Benchmark (branch)*], [*Fall (fit)*], [$m_c$]),
        [A1],
        [$+0.42$ ($m in [1,4]$)],
        [$+0.43$, free walk ($m in [1,10]$)],
        [$-0.36$ ($m in [25,60]$)],
        [10],

        [QLOT-final],
        [$+0.22$ ($m in [1,9]$)],
        [$approx +0.25$, AR(1) ($m in [1,22]$); $tau_c approx 11$],
        [$-0.13$ ($m in [30,55]$)],
        [22],
    ),
    caption: [Fitted NAD slopes and maxima. Empirical slopes are estimated over
        locally stable fit intervals, while benchmarks are evaluated over the full
        observed rising branch from $m=1$ to $m_c$. Averaging clips with different
        peak locations smears the mean curve near $m_c$, so the fit intervals stop
        before the rising branch rolls over.],
    placement: auto,
    scope: "parent",
) <tab-nad-fits>

*A1 integrates freely, then corrects late.* Its rising slope
matches the finite-range $m in [1, 10]$ free random walk benchmark.
Delta errors are essentially uncorrelated, and visual feedback engages only after $approx
10$ frames of free integration, consistent
with a memoryless architecture whose corrections require the accumulated error to be
visible in the current frame.
A continuous (AR(1)-type) restoring force would damp the rise to $approx +0.25$ and is
not supported by the data. The best-fit bounded walk
($tau_c = 8.5$, @fig-nad-vs-theory) damps the rise to $+0.27$ and peaks too late
($m_c = 16$ vs. measured $10$), even though it tracks the turnover better than
the free walk.

*QLOT-final corrects immediately.*
Its rise, half the free-walk benchmark, is quantitatively matched
by an AR(1) error with $tau_c approx 11$ frames (@fig-nad-vs-theory).
Temporal memory enables anticorrelated, self-correcting delta errors from the first frame onward.
The later maximum ($m_c = 22$) marks a longer error correlation time:
smaller and slower errors.

*Delta-error autocorrelation supports this interpretation.*
The standardized, channel-averaged correlogram in @fig-acf shows a negative
lag-1 correlation for QLOT-final ($-0.10$), whereas A1 is positive at lag 1
($+0.08$) and first becomes negative at lag 3.
Thus, QLOT-final's normalized error increments tend to oppose the preceding
increment immediately, while A1 exhibits a delayed correction signature.
Because the figure normalizes each clip and channel before averaging, it is a
qualitative check of correction timing rather than a covariance-weighted
decomposition of the NAD curve.

*Both tails are flatter than mean reversion.*
The stable falling slopes ($-0.36$, $-0.13$) lie between the $-1 slash 2$ of a
single-timescale AR(1) and a flat curve, consistent with heterogeneous correlation
times but not uniquely identifying bias instability. A two-timescale
correlation $gamma(h) = sigma^2 [0.4 e^(-h slash 4.5) + 0.6 e^(-h slash 20)]$,
one fast mode near the per-clip IQR lower edge and one slow mode near the
median, reproduces QLOT-final's rise, peak, and flat tail simultaneously
(@fig-nad-vs-theory).
From $m_c = 22$ to $m = 60$,
QLOT-final's NAD decreases only from $5.14 times 10^(-3)$ to
$4.78 times 10^(-3)$, or 7%. Thus, temporal averaging provides limited additional
attenuation over the measured range. The data do not establish an asymptotic floor
or a hard bound for smoothing-based post-processing.

*Negatives.* Neither mean curve shows a $+1$ segment.
No drift-dominated regime is detected within the measured range.
