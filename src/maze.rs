use rand::rngs::StdRng;
/// Maze generation and environment.
///
/// Implements iterative DFS (recursive-backtracker) on an odd-index cell
/// lattice, identical to the Python implementation.  The local-observation
/// cache pre-computes the 9-bit wall neighbourhood for every
/// (x, y, orientation) triple so each automaton tick is a single array
/// lookup.
use rand::{RngCore, SeedableRng};

/// Orientation constants (same order as Python's `Boolean2DGrid.Orientation`).
pub const UP: u8 = 0;
pub const RIGHT: u8 = 1;
pub const DOWN: u8 = 2;
pub const LEFT: u8 = 3;

/// (dx, dy) deltas for each orientation when moving forward.
/// Matches Python's `orientation_moves = [(0,-1),(1,0),(0,1),(-1,0)]`.
pub const ORIENTATION_MOVES: [(i32, i32); 4] = [
    (0, -1), // UP
    (1, 0),  // RIGHT
    (0, 1),  // DOWN
    (-1, 0), // LEFT
];

/// Scan order offsets for each of the 4 orientations (radius = 1, 3×3 window).
///
/// Returns `[(dy, dx); 9]` in the order that produces bits MSB→LSB for the
/// orientation-relative view (front-left, front, front-right, left, center,
/// right, back-left, back, back-right).
///
/// Matches Python's `Boolean2DGrid._generate_orientation_offsets(radius=1)`.
fn orientation_offsets(ori: u8) -> [(i32, i32); 9] {
    const R: i32 = 1;
    let mut out = [(0i32, 0i32); 9];
    let mut idx = 0usize;
    match ori {
        UP => {
            // row by row top-to-bottom (dy -r..r), left-to-right (dx -r..r)
            for dy in -R..=R {
                for dx in -R..=R {
                    out[idx] = (dy, dx);
                    idx += 1;
                }
            }
        }
        RIGHT => {
            // column by column right-to-left (dx r..-r), top-to-bottom (dy -r..r)
            for dx in (-R..=R).rev() {
                for dy in -R..=R {
                    out[idx] = (dy, dx);
                    idx += 1;
                }
            }
        }
        DOWN => {
            // row by row bottom-to-top (dy r..-r), right-to-left (dx r..-r)
            for dy in (-R..=R).rev() {
                for dx in (-R..=R).rev() {
                    out[idx] = (dy, dx);
                    idx += 1;
                }
            }
        }
        LEFT => {
            // column by column left-to-right (dx -r..r), bottom-to-top (dy r..-r)
            for dx in -R..=R {
                for dy in (-R..=R).rev() {
                    out[idx] = (dy, dx);
                    idx += 1;
                }
            }
        }
        _ => panic!("Invalid orientation: {ori}"),
    }
    out
}

/// Generate a perfect maze using iterative DFS (recursive-backtracker).
///
/// Returns `(wall, goal_pos, free)`:
/// - `wall`: row-major flat `bool` array (`width * height`); `true` = wall.
/// - `goal_pos`: `(gx, gy)` of the single goal cell.
/// - `free`: list of `(x, y)` non-wall cells.
pub fn generate_maze(
    side_length_bits: u8,
    seed: u64,
) -> (Vec<bool>, (usize, usize), Vec<(usize, usize)>) {
    assert!(side_length_bits >= 4, "side_length_bits must be >= 4");
    let side = 1usize << side_length_bits;
    let mut rng = StdRng::seed_from_u64(seed);

    // All walls initially.
    let mut wall = vec![true; side * side];

    let n_cells = side / 2 - 1; // number of cells per axis

    let mut visited = vec![false; n_cells * n_cells];

    let cell_to_grid = |cr: usize, cc: usize| -> (usize, usize) { (2 * cr + 1, 2 * cc + 1) };

    let dirs: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    // Random starting cell.
    let start_cr = (rng.next_u64() as usize) % n_cells;
    let start_cc = (rng.next_u64() as usize) % n_cells;

    let (mut gr, mut gc) = cell_to_grid(start_cr, start_cc);
    wall[gr * side + gc] = false;
    visited[start_cr * n_cells + start_cc] = true;

    let mut stack: Vec<(usize, usize)> = vec![(start_cr, start_cc)];

    while let Some(&(cr, cc)) = stack.last() {
        // Collect unvisited neighbours.
        let mut neighbours: Vec<(usize, usize, i32, i32)> = Vec::new();
        for &(dr, dc) in &dirs {
            let nr = cr as i32 + dr;
            let nc = cc as i32 + dc;
            if nr >= 0 && nr < n_cells as i32 && nc >= 0 && nc < n_cells as i32 {
                let (nr, nc) = (nr as usize, nc as usize);
                if !visited[nr * n_cells + nc] {
                    neighbours.push((nr, nc, dr, dc));
                }
            }
        }

        if !neighbours.is_empty() {
            let idx = (rng.next_u64() as usize) % neighbours.len();
            let (nr, nc, dr, dc) = neighbours[idx];
            let (ngr, ngc) = cell_to_grid(nr, nc);
            wall[ngr * side + ngc] = false;
            // Carve passage between current and neighbour.
            let wr = (gr as i32 + dr) as usize;
            let wc = (gc as i32 + dc) as usize;
            wall[wr * side + wc] = false;
            visited[nr * n_cells + nc] = true;
            stack.push((nr, nc));
            gr = ngr;
            gc = ngc;
        } else {
            stack.pop();
            if let Some(&(cr2, cc2)) = stack.last() {
                let (gr2, gc2) = cell_to_grid(cr2, cc2);
                gr = gr2;
                gc = gc2;
            }
        }
    }

    // Collect free cells.
    let free: Vec<(usize, usize)> = (0..side)
        .flat_map(|y| (0..side).map(move |x| (x, y)))
        .filter(|&(x, y)| !wall[y * side + x])
        .collect();

    // Place goal at a random free cell.
    let goal_idx = (rng.next_u64() as usize) % free.len();
    let goal_pos = free[goal_idx];

    (wall, goal_pos, free)
}

/// The maze environment with a precomputed local-observation cache.
#[allow(dead_code)]
pub struct Maze {
    pub name: String,
    pub side_length_bits: u8,
    pub seed: u64,
    pub width: usize,
    pub height: usize,
    /// Row-major flat wall array: `wall[y * width + x]` = true means wall.
    pub wall: Vec<bool>,
    /// Goal position (x, y).
    pub goal: (usize, usize),
    /// All non-wall cells.
    pub free: Vec<(usize, usize)>,
    /// Precomputed cache: `local_cache[ori * height * width + y * width + x]`
    /// holds the 9-bit wall observation as a `u16`.
    local_cache: Vec<u16>,
}

impl Maze {
    pub fn new(name: impl Into<String>, side_length_bits: u8, seed: u64) -> Self {
        let (wall, goal, free) = generate_maze(side_length_bits, seed);
        let side = 1usize << side_length_bits;
        let width = side;
        let height = side;

        let local_cache = build_local_cache(&wall, width, height);

        Maze {
            name: name.into(),
            side_length_bits,
            seed,
            width,
            height,
            wall,
            goal,
            free,
            local_cache,
        }
    }

    /// Return the 9-bit wall observation at `(x, y)` facing `orientation`.
    ///
    /// Bit 8 (MSB) = front-left, bit 0 (LSB) = back-right.
    #[inline]
    pub fn get_local(&self, x: usize, y: usize, orientation: u8) -> u16 {
        self.local_cache[orientation as usize * self.height * self.width + y * self.width + x]
    }

    #[inline]
    pub fn is_wall(&self, x: usize, y: usize) -> bool {
        self.wall[y * self.width + x]
    }

    #[inline]
    #[allow(dead_code)]
    pub fn is_goal(&self, x: usize, y: usize) -> bool {
        self.goal == (x, y)
    }

    /// Pick a random free cell using a raw RNG.
    pub fn random_free_cell(&self, rng: &mut dyn RngCore) -> (usize, usize) {
        self.free[(rng.next_u64() as usize) % self.free.len()]
    }
}

/// Precompute the local observation cache for all (x, y, orientation) triples.
fn build_local_cache(wall: &[bool], width: usize, height: usize) -> Vec<u16> {
    let mut cache = vec![0u16; 4 * height * width];
    for ori in 0u8..4 {
        let offsets = orientation_offsets(ori);
        for y in 0..height {
            for x in 0..width {
                let mut bits: u16 = 0;
                for (shift, &(dy, dx)) in offsets.iter().enumerate() {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let bit = if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        if wall[ny as usize * width + nx as usize] {
                            1u16
                        } else {
                            0u16
                        }
                    } else {
                        0u16 // border_value = false
                    };
                    // MSB first: bit 8 for shift=0, bit 0 for shift=8
                    bits |= bit << (8 - shift);
                }
                cache[ori as usize * height * width + y * width + x] = bits;
            }
        }
    }
    cache
}
