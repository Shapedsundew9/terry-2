# Logic Table

## The Question

X, Y and M are all fixed length bit strings of the same length.
If a bit position in X is 1 the same position in Y must be 0 and vice vera. i.e. for any bit position:

| X | Y | Valid |
|:-:|:-:|:-----:|
| 1 | 1 | False |
| 1 | 0 | True  |
| 0 | 1 | True  |
| 0 | 0 | True  |

Assume X and Y strings are always valid. M is one hot. The 1 may be in any position.
The following logic is valid:

| X | Y | M | x | y |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 0 | 1 | 0 | 0 |
| 0 | 1 | 1 | 0 | 0 |
| 0 | 0 | 1 | b | ~b |
| 1 | 0 | 0 | 1 | 0 |
| 0 | 1 | 0 | 0 | 1 |
| 0 | 0 | 0 | 0 | 0 |

Where x is a new valid X bit string and y is a new valid Y bit string. b is 0 or 1 and ~ represents the NOT operator.
The choice of the state of b must be uniformly random. i.e. for a specific set of bit strings X, Y and M the ratio
of x, y with b=1 and x, y with b=0 is 1:1. For this random function we define the unary random inversion operator ? which
is used as ?M resulting in a bit string of either M or ~M with equal probability.

Is there a sequence of bitwise logic functions, including ?, that can generate x and y?  

## The Answer

Yes, there is a sequence of bitwise logic functions that can reliably generate **x** and **y** exactly as defined.

### The Logic Sequence

You can generate the new strings using standard bitwise AND (`&`), OR (`|`), and NOT (`~`) operators along with your unary random inversion operator (`?`):

```text
R = ?M
B = M & R
B_inv = M & ~R

x = (X & ~M) | (~X & ~Y & B)
y = (Y & ~M) | (~X & ~Y & B_inv)

```

### How It Works

Because **M** is a one-hot bit string, the result of `?M` (which we capture as **R**) allows us to isolate a random bit specifically at the target position without introducing noise into the rest of the string.

**For all bit positions where M = 0:**

* `~M` evaluates to `1`.
* Because `M = 0`, both `B` and `B_inv` evaluate strictly to `0`.
* The equations simplify to `x = (X & 1) | 0`, which means `x = X` (and equivalently, `y = Y`).
* This flawlessly satisfies the bottom three rows of your logic table.

**For the single bit position where M = 1:**

* `~M` evaluates to `0`. This effectively erases the original **X** and **Y** values for this position (`X & ~M` becomes `0`).
* **R** has a 50% chance to be **M** and a 50% chance to be **~M**. Therefore, `B = M & R` will equal `1` half the time and `0` half the time. We can call this perfectly random state **b**.
* Conversely, `B_inv = M & ~R` yields the exact opposite, cleanly generating **~b**.
* The second half of the sequence, `~X & ~Y`, acts as a gatekeeper:
* **Rows 1 & 2 (X=1 or Y=1):** The gatekeeper evaluates to `0`, blocking the random bit and resulting in `x = 0` and `y = 0`.
* **Row 3 (X=0 and Y=0):** The gatekeeper evaluates to `1`, allowing the random states to pass through, resulting in `x = b` and `y = ~b`.

### Optimized Logic Sequence

By identifying common sub-expressions and using De Morgan's laws, we can significantly reduce the total operation count.
We can bring the entire calculation down to just **11 operations** by using temporary bitstrings to store reused values.

```text
// 1. Pre-calculate shared masks (3 operations)
not_M  = ~M
nor_XY = ~(X | Y)

// 2. Clear the M position from the original strings (2 operations)
X_base = X & not_M
Y_base = Y & not_M

// 3. Isolate the exact condition where we need randomness (1 operation)
// C will have a 1 ONLY if M=1 AND X=0 AND Y=0. Otherwise, it is all 0s.
C = M & nor_XY 

// 4. Generate the random bits for that specific condition (3 operations)
R   = ?M
B_x = C & R
B_y = C ^ B_x       // (See note below on XOR)

// 5. Combine the base strings with the newly generated bits (2 operations)
x = X_base | B_x
y = Y_base | B_y

```

#### Why this is more efficient:

* **`not_M`**: Inverting **M** once allows us to cleanly erase the target bit from both **X** and **Y** using a single AND (`&`) operation each.
* **De Morgan's Law (`nor_XY`)**: Instead of calculating `~X & ~Y` (which takes 3 operations), we calculate `~(X | Y)` (which takes 2). This gives us a mask showing everywhere both strings are 0.
* **The XOR trick (`B_y = C ^ B_x`)**: Because **B_x** is perfectly confined within the bounds of the **C** mask (it can only ever be 1 exactly where **C** is 1), an exclusive OR (`^`) perfectly flips the random bit `b` to `~b` while guaranteeing all other positions remain cleanly at 0.
