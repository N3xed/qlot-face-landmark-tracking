# Anisotropic 2D Radial Laplace Mixture Loss for Facial Landmark Detection

This document describes the theoretical motivation, mathematical formulation, and PyTorch implementation of an uncertainty-aware regression loss for 2D facial landmark localization and tracking. The formulation adapts the **2-component Mixture-of-Laplace (MoL)** loss from **SEA-RAFT** (Wang et al., ECCV 2024) and **WAFT** (Wang & Deng, 2026) to **2D Anisotropic Radial Laplace** distributions.

---

## 1. Motivation: Why Replace $L_2$ and 2D Gaussian NLL?

In 2D facial landmark detection, accuracy is typically evaluated via **Normalized Mean Error (NME)**, which is proportional to the average Euclidean (un-squared $L_2$, hereafter simply $L_2$) distance between predicted and ground-truth coordinates:

$$\text{EPE}(\mathbf{y}_{\text{gt}}, \hat{\mathbf{y}}) = \|\mathbf{y}_{\text{gt}} - \hat{\mathbf{y}}\|_2 = \sqrt{(x_{\text{gt}} - \hat{x})^2 + (y_{\text{gt}} - \hat{y})^2}$$

Standard training losses struggle to simultaneously optimize fine sub-pixel precision on clear landmarks (e.g., eye corners, pupils) and handle ambiguous or occluded landmarks (e.g., self-occluded profiles, hair/hand occlusions, or sliding annotations along the jawline):

1. **$L_2$ Loss ($\|\mathbf{e}\|_2$):**
   * **Strength:** Directly optimizes the NME numerator and has a unit-norm gradient $\nabla_{\hat{\mathbf{y}}} \|\mathbf{e}\|_2 = -\frac{\mathbf{e}}{\|\mathbf{e}\|_2}$ regardless of error magnitude.
   * **Weakness:** Assumes homoscedastic, isotropic noise across all landmarks. Ambiguous or occluded landmarks produce full-strength gradients in random directions, pulling shared backbone features away from high-precision landmarks.

2. **2D Anisotropic Gaussian NLL (GNLL):**
   Models the residual $\mathbf{e} = \mathbf{y}_{\text{gt}} - \hat{\mathbf{y}} \in \mathbb{R}^2$ as $\mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$, yielding the loss:
   $$\mathcal{L}_{\text{GNLL}} = \frac{1}{2} \mathbf{e}^\top \boldsymbol{\Sigma}^{-1} \mathbf{e} + \frac{1}{2}\log\det\boldsymbol{\Sigma}$$
   * **Strength:** Captures directional (anisotropic) uncertainty—such as sliding ambiguity tangent to the facial contour—via the $2\times 2$ covariance matrix $\boldsymbol{\Sigma}$.
   * **Weakness 1 (Vanishing Sub-Pixel Gradients):** Because the Mahalanobis term is **squared**, the gradient with respect to $\hat{\mathbf{y}}$ is $-\boldsymbol{\Sigma}^{-1}\mathbf{e}$, which shrinks linearly to $0$ as $\|\mathbf{e}\|_2 \to 0$. This weakens supervision in the $< 1\text{ px}$ regime.
   * **Weakness 2 (Exploding Outlier Gradients):** On large errors or sudden occlusions, the gradient grows linearly with $\|\mathbf{e}\|_2$, destabilizing training.
   * **Weakness 3 (Variance Collapse):** On easy, well-localized landmarks where $\|\mathbf{e}\|_2 \approx 0$, the network can drive $\det\boldsymbol{\Sigma} \to 0$ to achieve arbitrarily negative NLL, causing $\boldsymbol{\Sigma}^{-1}$ and its gradients to explode.

### Summary Comparison of Loss Families

| Property | $L_2$ | 2D Anisotropic GNLL | 2D Factored $L_1$ Laplace | **2D Anisotropic Radial Laplace Mixture (Ours)** |
| :--- | :--- | :--- | :--- | :--- |
| **Distance metric** | $\lVert\mathbf{e}\rVert_2$ | $\frac{1}{2}\mathbf{e}^\top \boldsymbol{\Sigma}^{-1}\mathbf{e}$ | $\frac{\lvert e_x \rvert}{b_x} + \frac{\lvert e_y \rvert}{b_y}$ | $\sqrt{\mathbf{e}^\top \boldsymbol{\Sigma}_1^{-1}\mathbf{e}}$ & $\lVert\mathbf{e}\rVert_2$ |
| **Gradient norm as $\lVert\mathbf{e}\rVert \to 0$** | Constant ($1$) | Vanishes ($\propto \lVert\mathbf{e}\rVert$) | Constant per axis | Constant ($1$ in whitened space) |
| **Gradient norm as $\lVert\mathbf{e}\rVert \to \infty$** | Bounded ($1$) | Unbounded ($\propto \lVert\mathbf{e}\rVert$) | Bounded ($1/b_d$) | Bounded ($1/b_k$) |
| **Captures rotated contour ambiguity** | No | Yes ($2\times 2$ $\boldsymbol{\Sigma}$) | Axis-aligned only | **Yes ($2\times 2$ $\boldsymbol{\Sigma}_1$)** |
| **Protected against $b \to 0$ collapse** | N/A | No (requires clamping) | Via 2-comp mixture | **Yes (fixed $\boldsymbol{\Sigma}_2 = \mathbf{I}_2$ inlier component)** |

---

## 2. Background: The SEA-RAFT / WAFT Mixture-of-Laplace Formulation

In optical flow, **SEA-RAFT** and **WAFT** replace the standard $L_1$ loss with the Negative Log-Likelihood (NLL) of a **2-component Mixture of Laplace (MoL)** distribution. Each coordinate $d \in \{u, v\}$ of the residual $\mathbf{e} = \mathbf{f}_{\text{gt}} - \hat{\mathbf{f}}$ is modeled independently as:

$$p(e_d) = \sum_{k=1}^{2} \alpha_k \frac{1}{2b_k} \exp\left(-\frac{|e_d|}{b_k}\right), \qquad \alpha_k = \frac{e^{w_k}}{e^{w_1} + e^{w_2}}$$

### Why a 2-Component Mixture Prevents Scale Collapse

If a single Laplace distribution $p(e_d) = \frac{1}{2b}\exp(-|e_d|/b)$ is trained with an unconstrained scale $b = e^s$, any sample with $|e_d| \to 0$ encourages $s \to -\infty$ ($b \to 0$), which causes the gradient scale $\frac{1}{b} = e^{-s}$ to explode.

SEA-RAFT and WAFT solve this by splitting the density into two regimes via clamping (`var_min = 0`, `var_max = 10`):
* **Component 2 (Inlier / Standard Loss Branch):** Clamped to $\log b_2 \in [0, 0] \implies b_2 = 1$. Its log-density is $-\log 2 - |e_d|$, which is **identical to standard $L_1$ loss** with unit gradient magnitude.
* **Component 1 (Outlier / Uncertainty Branch):** Clamped to $\log b_1 \in [0, s_{\max}] \implies b_1 \in [1, e^{s_{\max}}]$. It can only **increase** uncertainty ($b_1 \ge 1$) to attenuate noisy/occluded samples by $e^{-\log b_1}$, never shrink below $1$ to inflate gradients.

The network learns mixture logits $(w_1, w_2)$ to smoothly interpolate between exact $L_1$ optimization on confident samples ($\alpha_2 \to 1$) and uncertainty-attenuated optimization on ambiguous samples ($\alpha_1 \to 1$).

---

## 3. From 1D Laplace to 2D Radial (Elliptical) Laplace

While optical flow models use a factored 1D $L_1$ Laplace distribution over horizontal and vertical axes, 2D facial landmarks benefit from an **$L_2$ (Euclidean/Mahalanobis)** distance that is equivariant to 2D in-plane rotations and aligns with the NME evaluation metric.

### 3.1 Isotropic 2D Radial Laplace Density on $\mathbb{R}^2$

A 2D isotropic density on $\mathbf{e} = (e_x, e_y)^\top \in \mathbb{R}^2$ whose log-density is proportional to the $L_2$ norm $\|\mathbf{e}\|_2 = \sqrt{e_x^2 + e_y^2}$ has the form:

$$p(\mathbf{e} \mid b) = \frac{1}{Z(b)} \exp\left(-\frac{\|\mathbf{e}\|_2}{b}\right)$$

Integrating over $\mathbb{R}^2$ in polar coordinates ($e_x = r\cos\theta, \; e_y = r\sin\theta$, with area element $de_x\,de_y = r\,dr\,d\theta$):

$$Z(b) = \int_0^{2\pi} d\theta \int_0^\infty r \exp\left(-\frac{r}{b}\right) dr = 2\pi \cdot b^2 \Gamma(2) = 2\pi b^2$$

Thus, the normalized 2D Radial Laplace density (a 2D Multivariate Generalized Gaussian with shape $\beta = 1$, also used in LUVLi) is:

$$p(\mathbf{e} \mid b) = \frac{1}{2\pi b^2} \exp\left(-\frac{\|\mathbf{e}\|_2}{b}\right), \qquad \mathcal{L}_{\text{NLL}}(\mathbf{e}; b) = \frac{\|\mathbf{e}\|_2}{b} + 2\log b + \log(2\pi)$$

> **Note on the $2\log b$ factor:** Because $b$ scales length in two spatial dimensions, the 2D normalization volume scales as $b^2$, producing $+2\log b$ in the NLL.

### 3.2 Pitfall to Avoid: 2D Density on $\mathbb{R}^2$ vs. 1D Gamma($k=2$) on $r = \|\mathbf{e}\|_2$

Under the 2D density above, the marginal distribution of the scalar radius $r = \|\mathbf{e}\|_2 \in [0, \infty)$ incorporates the polar Jacobian factor $r$, yielding a **Gamma (Erlang-2)** distribution $\text{Gamma}(k=2, \theta=b)$:

$$p_R(r \mid b) = \frac{r}{b^2}\exp\left(-\frac{r}{b}\right), \qquad -\log p_R(r \mid b) = \frac{r}{b} + 2\log b - \log r$$

**Never include the $-\log r$ Jacobian term in the training loss:**
* The mode of $\text{Gamma}(k=2, b)$ is at $r = b$ (where the ring circumference $2\pi r$ balances the exponential decay), **not** at $r = 0$.
* Minimizing $-\log r$ with respect to $\hat{\mathbf{y}}$ introduces a repulsive gradient $-\frac{\mathbf{e}}{\|\mathbf{e}\|_2^2}$ that pushes predictions **away** from the ground truth whenever $\|\mathbf{e}\|_2 < b$.
* Always minimize the NLL of the **2D density on $\mathbb{R}^2$** ($\frac{\|\mathbf{e}\|_2}{b} + 2\log b$), whose mode is uniquely at $\mathbf{e} = \mathbf{0}$.

### 3.3 Anisotropic Extension via a $2\times 2$ Scale Matrix $\boldsymbol{\Sigma}$

To allow elliptical uncertainty contours that align with anatomical boundaries (e.g., jawline, eyebrows, lip contours), we replace the scalar $b^2 \mathbf{I}_2$ with a $2\times 2$ symmetric positive-definite scale matrix $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$. With the change of variables $\tilde{\mathbf{e}} = \mathbf{L}^{-1}\mathbf{e}$ ($d\mathbf{e} = |\det \mathbf{L}|\,d\tilde{\mathbf{e}} = \sqrt{\det\boldsymbol{\Sigma}}\,d\tilde{\mathbf{e}}$):

$$p(\mathbf{e} \mid \boldsymbol{\Sigma}) = \frac{1}{2\pi \sqrt{\det\boldsymbol{\Sigma}}} \exp\left(-\sqrt{\mathbf{e}^\top \boldsymbol{\Sigma}^{-1} \mathbf{e}}\right)$$

$$\mathcal{L}_{\text{NLL}}(\mathbf{e}; \boldsymbol{\Sigma}) = \sqrt{\mathbf{e}^\top \boldsymbol{\Sigma}^{-1} \mathbf{e}} + \frac{1}{2}\log\det\boldsymbol{\Sigma} + \log(2\pi)$$

Compared to 2D anisotropic GNLL, the quadratic form $\frac{1}{2}\mathbf{e}^\top \boldsymbol{\Sigma}^{-1}\mathbf{e}$ is simply replaced by its square root $\sqrt{\mathbf{e}^\top \boldsymbol{\Sigma}^{-1}\mathbf{e}}$.

---

## 4. The 2-Component Anisotropic Radial Laplace Mixture

Combining the 2D anisotropic Radial Laplace density with the SEA-RAFT / WAFT 2-component mixture trick yields a per-landmark density:

$$p(\mathbf{e}) = \alpha_1 \underbrace{\frac{1}{2\pi \sqrt{\det\boldsymbol{\Sigma}_1}} \exp\left(-\sqrt{\mathbf{e}^\top \boldsymbol{\Sigma}_1^{-1} \mathbf{e}}\right)}_{\text{Component 1: Anisotropic Outlier / Contour Branch}} \;+\; \alpha_2 \underbrace{\frac{1}{2\pi} \exp\left(-\|\mathbf{e}\|_2\right)}_{\text{Component 2: Isotropic Inlier Branch } (\boldsymbol{\Sigma}_2 = \mathbf{I}_2)}$$

where:
* $\alpha_k = \operatorname{softmax}(w_1, w_2)_k$ are the learned mixture weights per landmark.
* **Component 1 ($\boldsymbol{\Sigma}_1 \succeq \mathbf{I}_2$):** Anisotropic $2\times 2$ scale matrix whose eigenvalues are clamped to $\ge 1$.
* **Component 2 ($\boldsymbol{\Sigma}_2 = \mathbf{I}_2$):** Fixed isotropic unit scale matrix ($\frac{1}{2}\log\det\boldsymbol{\Sigma}_2 = 0$).

### Numerically Stable NLL via `logsumexp`

To ensure smooth, finite gradients at $\mathbf{e} = \mathbf{0}$, we include a small constant $\epsilon$ (e.g., $\epsilon = 10^{-4}$) inside the square roots:

$$d_1(\mathbf{e}) = \sqrt{\mathbf{e}^\top \boldsymbol{\Sigma}_1^{-1} \mathbf{e} + \epsilon^2}, \qquad d_2(\mathbf{e}) = \sqrt{e_x^2 + e_y^2 + \epsilon^2}, \qquad s_{\det, 1} = \frac{1}{2}\log\det\boldsymbol{\Sigma}_1, \qquad s_{\det, 2} = 0$$

The Negative Log-Likelihood per landmark is:

$$\mathcal{L}_{\text{MoL}}(\mathbf{e}) = \operatorname{logsumexp}(w_1, w_2) - \operatorname{logsumexp}\Big( w_1 - \log(2\pi) - s_{\det, 1} - d_1(\mathbf{e}), \;\; w_2 - \log(2\pi) - d_2(\mathbf{e}) \Big)$$

*(Note: $\log(2\pi)$ shifts both terms inside the second `logsumexp` equally and only adds an additive constant $+\log(2\pi)$ to the total loss; it can be kept for exact NLL evaluation or omitted during optimization.)*

### 4.1 Parameterizing Mixture Weights: 1 Logit vs. 2 Logits vs. $N$ Logits

In **SEA-RAFT** and **WAFT**, the mixture loss is implemented for a general $N$-component Laplace mixture where $N$ is a configurable hyperparameter (`dim = N` with `softmax` / `logsumexp` along the component dimension), and $N = 2$ is simply the default choice. When $N > 2$, predicting $N$ unconstrained logits $(w_1, \dots, w_N)$ and normalizing via $\alpha_k = \operatorname{softmax}(\mathbf{w})_k$ is the standard multi-class parameterization (despite its 1D shift invariance $\mathbf{w} \mapsto \mathbf{w} + c\mathbf{1}$).

However, when the mixture is **structurally fixed at $N = 2$** (1 anisotropic outlier branch + 1 fixed isotropic inlier branch), predicting two logits $(w_1, w_2)$ has no representational or numerical advantage over predicting a **single scalar logit $w = w_1 - w_2$** with $\alpha_1 = \sigma(w)$ and $\alpha_2 = 1 - \alpha_1 = \sigma(-w)$:

1. **Exact Functional Equivalence and Redundant Nullspace:**
   Because $\operatorname{softmax}(w_1, w_2)_1 = \frac{1}{1 + e^{-(w_1 - w_2)}} = \sigma(w_1 - w_2)$, the loss depends solely on the difference $w = w_1 - w_2$. If $\mathbf{a}_1, \mathbf{a}_2$ are the two rows of the linear projection head producing $(w_1, w_2)$, their gradients are exact negatives:
   $$\frac{\partial \mathcal{L}}{\partial w_1} = -\frac{\partial \mathcal{L}}{\partial w_2} \implies \nabla_{\mathbf{a}_1}\mathcal{L} = -\nabla_{\mathbf{a}_2}\mathcal{L}$$
   The common-mode sum $\mathbf{a}_1 + \mathbf{a}_2$ lies in the nullspace of the loss, receiving zero supervision gradient (only decaying toward $\mathbf{0}$ under weight decay).
2. **Optimizer Step Scaling in Adam / AdamW:**
   Because $|\nabla_{\mathbf{a}_1}\mathcal{L}| = |\nabla_{\mathbf{a}_2}\mathcal{L}|$, Adam's second-moment accumulators $v_1$ and $v_2$ are identical. Adam updates $\mathbf{a}_1$ by $+\Delta$ and $\mathbf{a}_2$ by $-\Delta$, causing the effective logit $w = w_1 - w_2$ to take a step of $2\Delta$ per iteration instead of $\Delta$.
3. **Fewer Output Parameters and Direct Bias Control:**
   Using a single logit $w$ saves 1 output channel per landmark (reducing the uncertainty head from 6 to 5 outputs: $(\tilde{s}_1, \tilde{s}_2, \tilde{u}_x, \tilde{u}_y, w)$, or 7 total outputs including $(\hat{x}, \hat{y})$) and exposes a single interpretable bias parameter. For example, initializing `bias[w] < 0` (e.g., $-1$ or $-2$) directly biases early training toward the clean inlier $L_2$ branch ($\alpha_2 = \sigma(-w) \to 1$).
4. **Identical Numerical Stability via `logsigmoid` and `logaddexp`:**
   Subtracting $w_2$ from both `logsumexp` terms in $\mathcal{L}_{\text{MoL}}(\mathbf{e})$ yields an equally stable 1-logit formulation without stacking 2-component weight tensors:
   $$\log\alpha_1 = \log\sigma(w) = -\operatorname{softplus}(-w), \qquad \log\alpha_2 = \log\sigma(-w) = -\operatorname{softplus}(w)$$
   $$\mathcal{L}_{\text{MoL}}(\mathbf{e}) = -\operatorname{logaddexp}\Big(\log\alpha_1 - s_{\det, 1} - d_1(\mathbf{e}), \;\; \log\alpha_2 - d_2(\mathbf{e})\Big) + \log(2\pi)$$

---

## 5. Parameterizing $\boldsymbol{\Sigma}_1$: Principal Axes + Continuous 2D Rotation

To construct $\boldsymbol{\Sigma}_1$ without matrix inversions, singularities, or eigenvalue leakage below $1$, we parameterize $\boldsymbol{\Sigma}_1$ via its spectral (eigendecomposition) form:

$$\boldsymbol{\Sigma}_1 = \mathbf{R}(\theta) \begin{pmatrix} b_1^2 & 0 \\ 0 & b_2^2 \end{pmatrix} \mathbf{R}(\theta)^\top, \qquad \mathbf{R}(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

where $b_1 = e^{s_1}$ and $b_2 = e^{s_2}$ are the principal semi-axes of the uncertainty ellipse.

### 5.1 Clamping the Principal Log-Scales $(s_1, s_2)$

By clamping both raw log-scales to $[0, s_{\max}]$ (e.g., $s_{\max} = 10$):

$$s_1 = \operatorname{clamp}(\tilde{s}_1, 0, s_{\max}), \qquad s_2 = \operatorname{clamp}(\tilde{s}_2, 0, s_{\max})$$

we obtain three major benefits:
1. **Strict Eigenvalue Guarantee ($\boldsymbol{\Sigma}_1 \succeq \mathbf{I}_2$):** Both principal semi-axes satisfy $b_1 \ge 1$ and $b_2 \ge 1$ for **any** rotation angle $\theta$. The model can never shrink uncertainty below $1$ in any direction.
2. **Exact Contour Decoupling:** For a landmark on the jawline, the network can set the normal log-scale $s_\perp = 0$ ($b_\perp = 1$, applying full $L_2$ supervision perpendicular to the jaw boundary) while setting the tangent log-scale $s_\parallel > 0$ ($b_\parallel \gg 1$, absorbing annotation sliding along the boundary).
3. **Trivial Log-Determinant and Whitening:** Since $\mathbf{R}(\theta)$ is orthogonal ($\det\mathbf{R}(\theta) = 1$ and $\mathbf{R}(\theta)^{-1} = \mathbf{R}(\theta)^\top$), the log-determinant penalty is simply:
   $$s_{\det, 1} = \frac{1}{2}\log\det\boldsymbol{\Sigma}_1 = s_1 + s_2$$
   and the Mahalanobis norm is computed by rotating $\mathbf{e} = (e_x, e_y)^\top$ into the ellipse's principal frame:
   $$\begin{pmatrix} e_\parallel \\ e_\perp \end{pmatrix} = \mathbf{R}(\theta)^\top \begin{pmatrix} e_x \\ e_y \end{pmatrix} = \begin{pmatrix} \cos\theta \, e_x + \sin\theta \, e_y \\ -\sin\theta \, e_x + \cos\theta \, e_y \end{pmatrix}$$
   $$d_1(\mathbf{e}) = \sqrt{\left(e_\parallel \, e^{-s_1}\right)^2 + \left(e_\perp \, e^{-s_2}\right)^2 + \epsilon^2}$$

### 5.2 Predicting $(\cos\theta, \sin\theta)$ Directly via $L_2$-Normalized $(u_x, u_y)$

Rather than predicting a scalar angle $\theta \in \mathbb{R}$ and passing it through periodic `cos` and `sin` activations—which creates wrap-around discontinuities and periodic local minima (Zhou et al., CVPR 2019)—the network predicts two unconstrained values $(\tilde{u}_x, \tilde{u}_y)$ and projects them onto the unit circle:

$$u_x = 1 + \tilde{u}_x, \qquad u_y = \tilde{u}_y$$

$$\cos\theta = \frac{u_x}{\sqrt{u_x^2 + u_y^2 + \epsilon_{\text{norm}}}}, \qquad \sin\theta = \frac{u_y}{\sqrt{u_x^2 + u_y^2 + \epsilon_{\text{norm}}}}$$

**Why this parameterization works well:**
* **Mandatory Unit-Norm Constraint ($\cos^2\theta + \sin^2\theta = 1$):** Normalizing $(u_x, u_y)$ is essential; without it, $\det\mathbf{R} = u_x^2 + u_y^2 \neq 1$, and the network would trivially drive $(u_x, u_y) \to (0, 0)$ to shrink $d_1(\mathbf{e}) \to 0$ without increasing $s_{\det, 1} = s_1 + s_2$.
* **Offsetting $u_x = 1 + \tilde{u}_x$ Avoids the Origin Singularity at Initialization:** When network head weights are initialized near zero ($\tilde{u}_x \approx 0, \tilde{u}_y \approx 0$), the unnormalized vector starts at $(u_x, u_y) \approx (1, 0)$—far from the $(0, 0)$ normalization singularity—and defaults to the canonical axis-aligned frame $(\cos\theta, \sin\theta) = (1, 0)$.
* **Zero Orientation Gradient Under Isotropic Uncertainty ($s_1 = s_2$):** When $s_1 = s_2 = s$,
  $$(e_\parallel e^{-s})^2 + (e_\perp e^{-s})^2 = e^{-2s}(\cos^2\theta + \sin^2\theta)(e_x^2 + e_y^2) = e^{-2s}(e_x^2 + e_y^2)$$
  Because $\cos^2\theta + \sin^2\theta = 1$ holds identically, the gradient with respect to $(\tilde{u}_x, \tilde{u}_y)$ automatically vanishes whenever the predicted uncertainty is isotropic.

### 5.3 Alternative: Correlation Parameterization $(\tilde{s}_x, \tilde{s}_y, \tilde{\rho})$

If reusing a 3-parameter head that outputs axis log-scales $(\tilde{s}_x, \tilde{s}_y)$ and correlation $\tilde{\rho}$:

$$\boldsymbol{\Sigma}_1 = \begin{pmatrix} b_x^2 & \rho b_x b_y \\ \rho b_x b_y & b_y^2 \end{pmatrix}, \qquad s_x, s_y \in [0, s_{\max}], \qquad \rho = \rho_{\max}\tanh(\tilde{\rho})$$

$$d_1(\mathbf{e}) = \sqrt{\frac{(e_x e^{-s_x})^2 - 2\rho(e_x e^{-s_x})(e_y e^{-s_y}) + (e_y e^{-s_y})^2}{1 - \rho^2} + \epsilon^2}, \qquad s_{\det, 1} = s_x + s_y + \frac{1}{2}\log(1 - \rho^2)$$

> **Caveat with Correlation Parameterization:** Even when $s_x = s_y = 0$ ($b_x = b_y = 1$), the eigenvalues of $\boldsymbol{\Sigma}_1$ are $1 + |\rho|$ and $1 - |\rho|$. If $|\rho|$ approaches $1$, the minor eigenvalue $1 - |\rho|$ drops below $1$, allowing the model to shrink scale below $1$ along the minor axis. The Principal Axes + Rotation parameterization in Section 5.1–5.2 avoids this issue completely.

### 5.4 Orientation Biasing and Symmetry-Breaking Gradient Analysis

A natural question in Section 5.2 is whether biasing the unnormalized rotation vector toward $\mathbf{u}_0 = (1, 0)$ ($\theta_0 = 0^\circ$) suppresses one of the axes at initialization, and whether biasing toward $\mathbf{u}_0 = \frac{1}{\sqrt{2}}(1, 1)$ ($\theta_0 = 45^\circ$) would better allow both axes to compete from the start. Analyzing the gradients of $(s_1, s_2)$ and $(\tilde{u}_x, \tilde{u}_y)$ shows that **$\mathbf{u}_0 = (1, 0)$ is strictly preferable to $\frac{1}{\sqrt{2}}(1, 1)$**, and reveals how scale and orientation symmetries interact early in training:

1. **Both Principal Axes Are Active at $\mathbf{u}_0 = (1, 0)$:**
   Because $(u_x, u_y) = (\cos\theta, \sin\theta)$ parameterizes an orthogonal rotation $\mathbf{R}(\theta)$ rather than coordinate weights, $\mathbf{u}_0 = (1, 0)$ corresponds to the identity rotation $\mathbf{R}(0) = \mathbf{I}_2$. At $\theta_0 = 0^\circ$, $e_\parallel = e_x$ and $e_\perp = e_y$, so $s_1$ scales horizontal residuals and $s_2$ scales vertical residuals; both principal axes receive full supervision from step 1. By contrast, $\mathbf{u}_0 = \frac{1}{\sqrt{2}}(1, 1)$ initializes the principal axes along the $+45^\circ$ and $+135^\circ$ diagonals ($e_\parallel = \frac{e_x + e_y}{\sqrt{2}}$, $e_\perp = \frac{-e_x + e_y}{\sqrt{2}}$).

2. **Why $\mathbf{u}_0 = \frac{1}{\sqrt{2}}(1, 1)$ ($\theta_0 = 45^\circ$) Creates a Saddle Point for Upright Facial Contours:**
   Differentiating $d_1(\mathbf{e})$ with respect to the rotation angle $\theta$ gives:
   $$\frac{\partial d_1}{\partial \theta} = \frac{\left(e^{-2s_1} - e^{-2s_2}\right) e_\parallel e_\perp}{d_1(\mathbf{e})}$$
   Because of the factor $(e^{-2s_1} - e^{-2s_2})$, **orientation $\theta$ receives zero gradient whenever $s_1 = s_2$**. When $s_1$ and $s_2$ are initialized equally ($s_1 = s_2$), the ellipse cannot begin rotating until $(s_1, s_2)$ break symmetry. At $s_1 = s_2$, the difference between the scale gradients $\frac{\partial \mathcal{L}}{\partial s_1}$ and $\frac{\partial \mathcal{L}}{\partial s_2}$ is proportional to:
   $$e_\parallel^2 - e_\perp^2 = (e_x^2 - e_y^2)\cos(2\theta_0) + 2 e_x e_y \sin(2\theta_0)$$
   * **At $\mathbf{u}_0 = (1, 0)$ ($\theta_0 = 0^\circ$):** $e_\parallel^2 - e_\perp^2 = e_x^2 - e_y^2$. In canonical upright face crops, most anatomical contours are predominantly horizontal (eyebrows, eyelids, lips: $\mathbb{E}[e_x^2] > \mathbb{E}[e_y^2]$) or vertical (nose bridge, upper jawline: $\mathbb{E}[e_y^2] > \mathbb{E}[e_x^2]$). Moreover, $e_x^2 - e_y^2$ is invariant under horizontal flip augmentation ($e_x \mapsto -e_x$). Thus $\mathbb{E}[e_\parallel^2 - e_\perp^2] \neq 0$ immediately drives $s_1 \neq s_2$, which unlocks $\frac{\partial d_1}{\partial \theta} \neq 0$ so oblique landmarks can rotate to their true tangent angles.
   * **At $\mathbf{u}_0 = \frac{1}{\sqrt{2}}(1, 1)$ ($\theta_0 = 45^\circ$):** $e_\parallel^2 - e_\perp^2 = 2 e_x e_y$. Any purely horizontal ($e_y = 0$) or purely vertical ($e_x = 0$) error projects equally ($50\% / 50\%$) onto both $+45^\circ$ and $+135^\circ$ diagonals ($e_\parallel^2 = e_\perp^2$), and under random horizontal flips ($e_x \mapsto -e_x$), $\mathbb{E}[e_x e_y] = 0$ for symmetric landmarks. Consequently, $\mathbb{E}[\nabla_{s_1}\mathcal{L}] = \mathbb{E}[\nabla_{s_2}\mathcal{L}]$, keeping $s_1 \approx s_2$ and trapping $\frac{\partial d_1}{\partial \theta} \approx 0$ at a $45^\circ$ saddle point.

3. **Unit-Circle Tangent Gradients and $180^\circ$ Ellipse Periodicity:**
   The Jacobian of the unit-circle projection $\hat{\mathbf{u}} = \mathbf{u} / \|\mathbf{u}\|_2$ at any unit base vector $\mathbf{u}_0 = (\cos\theta_0, \sin\theta_0)^\top$ projects onto the tangent vector $\mathbf{t}_0 = (-\sin\theta_0, \cos\theta_0)^\top$:
   $$\nabla_{\tilde{\mathbf{u}}} \mathcal{L}\big|_{\tilde{\mathbf{u}}=\mathbf{0}} = \frac{\partial \mathcal{L}}{\partial \theta} \begin{pmatrix} -\sin\theta_0 \\ \cos\theta_0 \end{pmatrix}$$
   At $\mathbf{u}_0 = (1, 0)$, the tangent is vertical $(0, 1)^\top$, so at the exact initial point $\tilde{\mathbf{u}} = (0, 0)$, $\frac{\partial\mathcal{L}}{\partial\tilde{u}_x} = 0$ and the initial angular step flows entirely through $\tilde{u}_y$ (with $\tilde{u}_x$ receiving non-zero gradients as soon as $\tilde{u}_y \neq 0$). Furthermore, because a 2D covariance ellipse is $\pi$-periodic ($\mathbf{R}(\theta + \pi)$ produces the exact same $\boldsymbol{\Sigma}_1$ as $\mathbf{R}(\theta)$), the right half-plane $u_x > 0$ with $\tilde{u}_y \in (-\infty, \infty)$ already spans $\theta \in (-\frac{\pi}{2}, \frac{\pi}{2})$, covering every distinct ellipse orientation in $\mathbb{R}^2$ without requiring $u_x$ to cross zero.

4. **Optional: Breaking Scale Symmetry Directly via Distinct $(\tilde{s}_1, \tilde{s}_2)$ Biases:**
   Because $\frac{\partial d_1}{\partial \theta} \propto (e^{-2s_1} - e^{-2s_2})$, initializing both $\tilde{s}_1$ and $\tilde{s}_2$ with identical biases ($s_1 = s_2$) means $\theta$ receives zero gradient on the very first step and must wait for $s_1$ and $s_2$ to diverge first. Additionally, if $\tilde{s}_1, \tilde{s}_2$ are initialized at $0$, any negative pre-activations fall into the flat region of $\operatorname{clamp}(\tilde{s}_k, 0, s_{\max})$ where the gradient is zero. To ensure both scale and orientation gradients are active from step 1 and to break the $90^\circ$ axis-swap degeneracy $(s_1, s_2, \theta) \leftrightarrow (s_2, s_1, \theta + \frac{\pi}{2})$, one can initialize the output biases of $(\tilde{s}_1, \tilde{s}_2)$ to small, slightly different positive values (e.g., $b_{\tilde{s}_1} = 0.2, \; b_{\tilde{s}_2} = 0.05$, or parameterize $s_1 = s_2 + \Delta s$ with $\Delta s \ge 0$ so $s_1$ is always the major axis).

---

## 6. Inference-Time Uncertainty Extraction

For each landmark, the model predicts **8 values** (or **7 values** if using a single logit $w = w_1 - w_2$):
* `pred_xy` $\in \mathbb{R}^2$: Landmark coordinates $(\hat{x}, \hat{y})$, used directly as the point prediction.
* `weights` $\in \mathbb{R}^2$: Mixture logits $(w_1, w_2)$, with mixture probabilities $\alpha = \operatorname{softmax}(w_1, w_2)$.
* `raw_params` $\in \mathbb{R}^4$: $(\tilde{s}_1, \tilde{s}_2, \tilde{u}_x, \tilde{u}_y)$ for Component 1.

At inference time, you can extract:

1. **Scalar Uncertainty Score (Confidence / Occlusion Proxy):**
   Analogous to SEA-RAFT and WAFT's `get_heatmap`, the expected log-determinant scale per landmark is:
   $$u_{\text{scalar}} = \alpha_1 (s_1 + s_2) + \alpha_2 (0) = \alpha_1 (s_1 + s_2)$$
   Higher values indicate occluded or highly ambiguous landmarks.

2. **Effective $2\times 2$ Scale and Covariance Matrix:**
   First construct $\boldsymbol{\Sigma}_1 = \mathbf{R}(\theta)\operatorname{diag}(e^{2s_1}, e^{2s_2})\mathbf{R}(\theta)^\top$:
   $$\boldsymbol{\Sigma}_1 = \begin{pmatrix} b_1^2 \cos^2\theta + b_2^2 \sin^2\theta & (b_1^2 - b_2^2)\cos\theta\sin\theta \\ (b_1^2 - b_2^2)\cos\theta\sin\theta & b_1^2 \sin^2\theta + b_2^2 \cos^2\theta \end{pmatrix}$$
   The mixture-weighted effective scale matrix is:
   $$\boldsymbol{\Sigma}_{\text{eff}} = \alpha_1 \boldsymbol{\Sigma}_1 + \alpha_2 \mathbf{I}_2$$
   *(Note: For a 2D Radial Laplace distribution with scale matrix $\boldsymbol{\Sigma}$, the second radial moment is $\mathbb{E}[r^2] = \int_0^\infty \frac{r^3}{b^2}e^{-r/b}\,dr = 6b^2$, so the exact statistical covariance matrix is $\operatorname{Cov}(\mathbf{e}) = 3\boldsymbol{\Sigma}_{\text{eff}}$. This matrix can be fed directly into a Kalman filter for temporal tracking or used to render 2D confidence ellipses.)*

---

## 7. Coordinate Scaling and the Implicit Inlier Scale

### 7.1 The Implicit Inlier Threshold ($b = 1$)

In the unscaled formulation (Sections 3–5), the inlier component is fixed to $\boldsymbol{\Sigma}_2 = \mathbf{I}_2$ ($b = 1$) and the outlier semi-axes $b_k = e^{s_k}$ are clamped to $s_k \in [0, s_{\max}] \implies b_k \ge 1$.

To see what physical role $b = 1$ plays, consider the single-component outlier NLL along principal axis $k \in \{\parallel, \perp\}$ when the orthogonal residual vanishes:

$$\mathcal{L}_1(e_k; s_k) = |e_k| e^{-s_k} + s_k, \qquad \frac{\partial \mathcal{L}_1}{\partial s_k} = 1 - |e_k| e^{-s_k} = 1 - \frac{|e_k|}{b_k}$$

Setting $\frac{\partial \mathcal{L}_1}{\partial s_k} = 0$ shows that the unconstrained optimal scale is $b_k^* = |e_k|$. Because $s_k$ is clamped to $s_k \ge 0$ ($b_k \ge 1$):
* **Sub-unit residuals ($|e_k| \le 1$):** $\frac{\partial \mathcal{L}_1}{\partial s_k} = 1 - |e_k|e^{-s_k} \ge 0$ for all $s_k \ge 0$. The loss strictly increases with $s_k$, driving $s_k \to 0$ ($b_k \to 1$).
* **Super-unit residuals ($|e_k| > 1$):** $\frac{\partial \mathcal{L}_1}{\partial s_k}\big|_{s_k=0} = 1 - |e_k| < 0$, so $s_k$ grows toward $\log |e_k| > 0$ to attenuate the outlier gradient.

Thus, **$b = 1$ is the implicit inlier threshold** separating clean inliers from ambiguous/occluded outliers:
* In **pixel coordinates** (as in SEA-RAFT and WAFT), $b = 1$ corresponds to **$1\text{ px}$**, which naturally separates sub-pixel localization errors ($< 1\text{ px}$) from multi-pixel contour/occlusion ambiguities ($> 1\text{ px}$).
* In **normalized coordinates** $[-1, 1]^2$, however, a unit distance of $1$ equals **half the image dimension** ($W/2$ or $H/2$, e.g., $112\text{ px}$ on a $224\times 224$ crop). Typical landmark residuals in $[-1, 1]^2$ are on the order of $10^{-3}\text{–}10^{-1} \ll 1$. If evaluated directly in $[-1, 1]^2$ with $b = 1$, $|e_k| < 1$ holds for virtually every sample, $\frac{\partial \mathcal{L}_1}{\partial s_k} > 0$ everywhere, and the outlier branch collapses to $s_1 = s_2 = 0$.

### 7.2 2D Base Scale Parameterization $\mathbf{B} = \operatorname{diag}(b_x, b_y)$

To make the inlier threshold explicit and support arbitrary coordinate systems (including non-square aspect ratios), we introduce a 2D diagonal scale matrix:

$$\mathbf{B} = \begin{pmatrix} b_x & 0 \\ 0 & b_y \end{pmatrix}, \qquad b_x, b_y > 0$$

where $(b_x, b_y)$ specifies the inlier scale along the $x$- and $y$-axes in the active coordinate system—for example, $(b_x, b_y) = (1, 1)$ in pixel coordinates, or $(b_x, b_y) = \left(\frac{2}{W}, \frac{2}{H}\right)$ for a $1\text{ px}$ inlier threshold in normalized $[-1, 1]^2$ coordinates.

Both mixture components are defined relative to the base scale frame $\mathbf{B}$:

$$\boldsymbol{\Sigma}_2 = \mathbf{B}\mathbf{I}_2\mathbf{B}^\top = \begin{pmatrix} b_x^2 & 0 \\ 0 & b_y^2 \end{pmatrix}, \qquad \boldsymbol{\Sigma}_1 = \mathbf{B}\,\mathbf{R}(\theta)\begin{pmatrix} e^{2s_1} & 0 \\ 0 & e^{2s_2} \end{pmatrix}\mathbf{R}(\theta)^\top \mathbf{B}^\top$$

With this factorization:
1. **Whitened Residual Evaluation:**
   Defining the base-scale-normalized residual $\tilde{\mathbf{e}} = \mathbf{B}^{-1}\mathbf{e} = (e_x / b_x, \, e_y / b_y)^\top$, the rotated projections and Mahalanobis distances reduce to the standard unit-scale formulas applied to $\tilde{\mathbf{e}}$:
   $$\begin{pmatrix} \tilde{e}_\parallel \\ \tilde{e}_\perp \end{pmatrix} = \mathbf{R}(\theta)^\top \tilde{\mathbf{e}}, \qquad d_1(\mathbf{e}) = \sqrt{(\tilde{e}_\parallel e^{-s_1})^2 + (\tilde{e}_\perp e^{-s_2})^2 + \epsilon^2}, \qquad d_2(\mathbf{e}) = \sqrt{\tilde{e}_x^2 + \tilde{e}_y^2 + \epsilon^2}$$
   When $(b_x, b_y) = (2/W, 2/H)$ in normalized $[-1, 1]^2$ coordinates, $\tilde{\mathbf{e}}$ is identical to the pixel-space residual $\mathbf{e}_{\text{px}}$, and the NLL includes the Jacobian term $\log(b_x b_y)$:
   $$\mathcal{L}_{\text{MoL}}(\mathbf{e}) = -\operatorname{logaddexp}\Big(\log\alpha_1 - s_{\det, 1} - d_1(\mathbf{e}), \;\; \log\alpha_2 - d_2(\mathbf{e})\Big) + \log(2\pi b_x b_y)$$

2. **Closed-Form Coordinate Scaling Without Eigendecomposition:**
   Under any axis-aligned coordinate transformation $\mathbf{e}' = \mathbf{S}\mathbf{e}$ with $\mathbf{S} = \operatorname{diag}(c_x, c_y)$ (e.g., scaling from normalized $[-1, 1]^2$ coordinates to pixel coordinates via $(c_x, c_y) = (W/2, H/2)$ or vice versa), the covariance matrices transform as $\boldsymbol{\Sigma}_k' = \mathbf{S}\boldsymbol{\Sigma}_k\mathbf{S}^\top$. Because both $\mathbf{S}$ and $\mathbf{B}$ are diagonal, they commute and compose directly into a new base scale:
   $$\mathbf{B}' = \mathbf{S}\mathbf{B} = \operatorname{diag}(c_x b_x, \, c_y b_y)$$
   while the dimensionless distribution parameters $(s_1, s_2, \tilde{u}_x, \tilde{u}_y, w)$ remain completely unchanged.

3. **Scaled Inference Covariance:**
   At inference time, the effective scale matrix and statistical covariance matrix in the current coordinate frame are obtained by applying $\mathbf{B}$ on the outside of the unit-scale mixture matrix:
   $$\boldsymbol{\Sigma}_{\text{eff}} = \mathbf{B}\Big(\alpha_1 \mathbf{R}(\theta)\operatorname{diag}\!\left(e^{2s_1}, e^{2s_2}\right)\mathbf{R}(\theta)^\top + \alpha_2 \mathbf{I}_2\Big)\mathbf{B}^\top, \qquad \operatorname{Cov}(\mathbf{e}) = 3\boldsymbol{\Sigma}_{\text{eff}}$$

---

## 8. References

1. **SEA-RAFT:** Yihan Wang, Lahav Lipson, and Jia Deng. *"SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow."* ECCV 2024.
2. **WAFT:** Yihan Wang and Jia Deng. *"WAFT: Warping-Alone Field Transforms for Optical Flow."* arXiv:2506.21526, 2026.
3. **LUVLi:** Abhinav Kumar, Tim K. Marks, Wenxuan Mou, Ye Wang, Michael Jones, Anoop Cherian, Toshiaki Koike-Akino, Xiaoming Liu, and Chen Feng. *"LUVLi Face Alignment: Estimating Landmarks' Location, Uncertainty, and Visibility Likelihood."* CVPR 2020.
4. **RLE:** Jiefeng Li, Siyuan Bian, Ailing Zeng, Can Wang, Bo Pang, Wentao Liu, and Cewu Lu. *"Human Pose Regression with Residual Log-Likelihood Estimation."* ICCV 2021.
5. **Continuous Rotation Representations:** Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. *"On the Continuity of Rotation Representations in Neural Networks."* CVPR 2019.
