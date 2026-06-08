# Contrast-Weighted Salience Quantization

This document details the design, rationale, and implementation of the contrast-weighted salience quantization algorithm used in Terry-2 to downscale 2D grids of categorical cell states.

---

## 1. The Challenge of Categorical Grid Downscaling

In typical image processing, downscaling is performed using spatial averaging (e.g., bilinear or bicubic interpolation). However, grid states in ARC-AGI tasks are **categorical**. Each integer in `0..15` represents a unique semantic category/color (e.g., Red, Blue, Neutral, Off-black). 

Using numerical averaging on categorical labels results in two major failures:
1. **Semantic Corruption**: Averaging color categories yields arbitrary, unrelated colors (e.g., averaging Yellow `11` and Black `5` results in Magenta `8`, which is not in the source block).
2. **Feature Erasure**: High-contrast, sparse foreground details (such as a 1-pixel red key on a black background) are completely overwhelmed by the background color, erasing them from low-resolution views.

---

## 2. Redefining Categorical Meanings Across Scales

Instead of viewing a downscaled cell as a uniform representation of the block's color, we redefine its meaning based on scale:
*   **At $64\times64$ (1x1 pixels)**: The category represents the exact color of that single cell.
*   **At $4\times4$ ($16\times16$ blocks)**: The category represents the **most visually salient/significant foreground color** present in that spatial quadrant.

This allows both human observers and cellular automata transition rules operating on downscaled grids to retain critical information about sparse objects.

---

## 3. Contrast-Weighted Salience Formulation

To select the most representative category for a block, we evaluate each candidate color $c$ present in the block using a scoring function that balances **spatial coverage** and **perceptual contrast**:

$$\text{Score}(c) = \text{Count}(c)^\alpha \times \left(1.0 + w \cdot \frac{\text{dist}(P[c], P[\text{bg}])}{\text{MAX\_DIST}}\right)$$

### Key Components

1. **Global Background Color (`bg`)**
   The most common color across the entire $64\times64$ frame is dynamically detected as the background reference. For the initial frame in `ls20`, this is Off-Black (`4`).
   
2. **Perceptual Contrast Distance (`dist`)**
   We calculate the Euclidean distance in RGB space between the palette color $P[c]$ and the background color $P[\text{bg}]$:
   $$\text{dist}(P[c], P[\text{bg}]) = \sqrt{(R_c - R_{\text{bg}})^2 + (G_c - G_{\text{bg}})^2 + (B_c - B_{\text{bg}})^2}$$
   This is normalized by $\text{MAX\_DIST} \approx 441.67$ (the distance between absolute White and Black).

3. **Non-Linear Count Exponent ($\alpha = 0.3$)**
   Using $\alpha < 1.0$ compresses the influence of pure area/count. A linear count would require $200\times$ more pixels for a foreground object to overcome the background. With $\alpha = 0.3$, a small set of high-contrast pixels can override the background, acting as a visual dilation operator.

4. **Contrast Weight ($w = 8.0$)**
   Determines the sensitivity of the algorithm to high-contrast colors relative to the background.

---

## 4. Visual Results Comparison

Below is the text representation comparison of the $4\times4$ grid panels (each cell representing a $16\times16$ block in the original $64\times64$ grid):

### Original Grid Layout (Key Objects)
*   **Top Right**: A small Blue box (`B`)
*   **Bottom Left**: A small Blue box (`B`)
*   **Bottom Right**: A large Yellow area (`Y`) with small Red pixels (`R`)
*   **Middle**: Horizontal Neutral/Gray dividers (`-`)

### Quantization Output comparison:

| Average (Old) | Mode/Majority | Contrast-Weighted (New) |
| :---: | :---: | :---: |
| <pre>   <br>  -<br> - <br>####</pre> | <pre>   <br>  -<br> --<br>   </pre> | <pre># B <br>---<br>---<br>BYYY</pre> |

*   **Average**: Corrupts states, producing arbitrary Black `####` blocks, completely losing the Blue box, Yellow area, and Red pixels.
*   **Mode/Majority**: Erases all foreground objects because background pixels numerically dominate every block.
*   **Contrast-Weighted**: Perfectly preserves the Blue box (`B`), the Yellow region (`Y`), and the Neutral separators (`-`), retaining the structural meaning of the puzzle.
