# Contrast-Weighted Frame Delta Design

This document details the design, rationale, and implementation of the human-centric, contrast-weighted frame delta calculation and scale-space aggregation used in Terry-2 to highlight changes between consecutive grid frames.

---

## 1. The Challenge of Categorical Frame Delta

In standard computer vision, the difference (delta) between two frames is calculated using pixel-wise subtraction or simple distance metrics on continuous color channels (e.g., RGB). However, grid states in ARC-AGI tasks are **categorical** palette indices (`0..15`).

Applying simple subtraction or absolute differences to categorical values yields two main failures:

1. **Semantic Nonsense**: A change from Off-White `1` to Magenta `6` yields a numerical delta of `5`, while a change from Neutral `3` to Red `8` also yields `5`. Numerically they are equal, but perceptually and semantically they represent completely different changes.
2. **Visual Insensitivity**: The human visual system (HVS) does not perceive changes uniformly across channels (we are most sensitive to green, then red, then blue). A simple index delta fails to capture this perceptual difference.
3. **Loss of sparse changes at low resolutions**: A single-pixel flashing on a large dark background is highly salient to a human observer. If we aggregate changes using standard average pooling when downscaling, these critical, localized changes are averaged to zero and lost.

---

## 2. Redefining Temporal Change Across Scales

Instead of calculating deltas directly on already-quantized frames, we compute change saliency at the base $64\times64$ resolution first, and then downsample/aggregate the saliency map. This matches how the human visual system processes temporal changes across spatial frequencies:

* **At $64\times64$ ($1\times1$ pixels)**: The delta represents the perceptual change of a single cell.
* **At $4\times4$ ($16\times16$ blocks)**: The delta represents the **maximum local change saliency** present in that spatial quadrant, preventing temporal details from washing out.

---

## 3. Human-Centric Delta Formulation

To calculate the frame delta between frame $F_{t-1}$ and frame $F_t$ at any scale, we use a three-step pipeline:

### Step 1: Base Perceptual Color Distance ($D_{\text{temporal}}$)

We compute the raw change distance at the base $64\times64$ resolution using a luma-weighted Euclidean distance in RGB space:

$$D_{\text{temporal}}(x, y) = \sqrt{0.299 \cdot \Delta R^2 + 0.587 \cdot \Delta G^2 + 0.114 \cdot \Delta B^2}$$

where $\Delta R$, $\Delta G$, and $\Delta B$ are the differences between the RGB components of the palette colors $P[F_{t-1}(x,y)]$ and $P[F_t(x,y)]$. The maximum possible distance is $255.0$ (representing the transition between absolute white and black).

### Step 2: Change Saliency ($S_{\text{change}}$)

To incorporate contrast, we weight the temporal change by how much the changing pixels stand out from the global background color ($bg$). If a pixel transitions to or from a color that is highly contrasting with the background, its visual salience is boosted:

$$S_{\text{change}}(x, y) = D_{\text{temporal}}(x, y) \times \left(1.0 + \gamma \cdot \frac{\max(\text{dist}(P[F_{t-1}(x,y)], P[bg]), \text{dist}(P[F_t(x,y)], P[bg]))}{\text{MAX\_PERCEPTUAL\_DIST}}\right)$$

* **Background reference (`bg`)**: The most common color across the entire $64\times64$ current frame.
* **Contrast boost ($\gamma = 4.0$)**: Determines the influence of background contrast in highlighting changes.
* **$\text{MAX\_PERCEPTUAL\_DIST} = 255.0$**: The normalization factor.

This formulation ensures that a transition involving a prominent foreground color (high distance to background) is marked as highly salient, while minor background fluctuations are suppressed.

### Step 3: Scale-Space Aggregation ($\text{BlockDelta}_K$)

For a quantization factor $K \in \{2, 4, 8, 16\}$, each quantized cell $(X, Y)$ covers a block of size $K \times K$ in the base frame.

To compute the delta $\text{BlockDelta}_K(X, Y)$, we use a **Generalized Mean ($p$-norm)** of the base saliency values within that block:

$$\text{BlockDelta}_K(X, Y) = \left( \frac{1}{K^2} \sum_{dx=0}^{K-1} \sum_{dy=0}^{K-1} S_{\text{change}}(X \cdot K + dx, Y \cdot K + dy)^p \right)^{1/p}$$

Using $p = 3.0$ acts as a visual compromise between:

* **$p = 1$ (Average)**: Measures the total area of change, but washes out small details.
* **$p \to \infty$ (Max-pooling)**: Preserves single-pixel changes but cannot tell them apart from whole-block changes.

The result is normalized and scaled to the range $[0.0, 1.0]$.

---

## 4. Visual Representation

The computed delta values are mapped to grayscale intensities:

* `0.0` maps to absolute black (`[0, 0, 0]`), representing no change.
* `1.0` maps to absolute white (`[255, 255, 255]`), representing maximum perceptual change.

For the initial frame ($t = 0$), where no previous frame exists, a zeroed (black) grid is generated at all scales.
