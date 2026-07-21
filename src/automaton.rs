use rand::rngs::StdRng;
/// MazeAutomaton — Mealy-machine automaton that navigates a maze.
///
/// Mirrors the Python `MazeAutomaton` / `AutomatonISBase` / `AutomatonBase`
/// hierarchy in a single flat struct for performance.
use rand::{RngCore, SeedableRng};

use crate::genetic_code::GeneticCode;
use crate::maze::{Maze, ORIENTATION_MOVES};

/// A single maze-navigating automaton.
///
/// Internal state layout (matching Python's `AutomatonISBase`):
/// - `input_code = (internal_state << env_bits) | env_local`
/// - `output_code = genetic_code[input_code]`
/// - `new_internal_state = output_code & state_mask`
/// - `action = output_code >> state_bits`
pub struct MazeAutomaton {
    // --- Position & orientation ---
    pub x: usize,
    pub y: usize,
    pub orientation: u8, // 0=UP 1=RIGHT 2=DOWN 3=LEFT

    // --- Mealy machine ---
    pub internal_state: u8,
    pub env_bits: u8,   // always 9 (3×3 wall observation)
    pub state_bits: u8, // configurable
    pub resp_bits: u8,  // always 2 (forward/left/right/invalid)
    state_mask: u8,
    resp_mask: u8,
    env_mask: u16,

    // --- Energy & exploration ---
    /// Current energy budget.  Decremented by 1 per tick; automaton
    /// becomes inactive when it reaches 0.
    pub energy: i32,
    /// Per-cell energy grid (flat, width×height).  Starts at 1; zeroed on
    /// first visit to that cell to reward exploration.
    pub energy_grid: Vec<u8>,
    pub grid_width: usize,

    // --- Fitness ---
    pub fitness: f64,

    // --- Last action (for checkpoint) ---
    pub last_action: i32,

    // --- Genetic code ---
    pub genetic_code: Box<dyn GeneticCode>,

    // --- Automaton RNG (for reset random placement) ---
    rng: StdRng,
    /// Seed stored for checkpoint round-trips.
    pub seed: Option<u64>,
}

impl MazeAutomaton {
    /// Create a new automaton with an empty `GeneticCodeDict`.
    pub fn new(maze: &Maze, state_bits: u8, seed: u64) -> Self {
        use crate::genetic_code::GeneticCodeDict;
        let mut rng = StdRng::seed_from_u64(seed);
        let code_seed = rng.next_u64();
        let output_bits = state_bits + 2; // state_bits + resp_bits
        let genetic_code = Box::new(GeneticCodeDict::new(output_bits, code_seed));
        Self::from_parts(genetic_code, maze, state_bits, &mut rng, Some(seed))
    }

    /// Create a new automaton with a supplied genetic code (for offspring).
    pub fn with_code(
        genetic_code: Box<dyn GeneticCode>,
        maze: &Maze,
        state_bits: u8,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::from_parts(genetic_code, maze, state_bits, &mut rng, Some(seed))
    }

    fn from_parts(
        genetic_code: Box<dyn GeneticCode>,
        maze: &Maze,
        state_bits: u8,
        rng: &mut StdRng,
        seed: Option<u64>,
    ) -> Self {
        let env_bits: u8 = 9;
        let resp_bits: u8 = 2;
        let state_mask: u8 = ((1u16 << state_bits) - 1) as u8;
        let resp_mask: u8 = ((1u16 << resp_bits) - 1) as u8;
        let env_mask: u16 = (1u16 << env_bits) - 1;

        let (x, y) = maze.random_free_cell(rng);
        let orientation = (rng.next_u32() & 3) as u8;
        let grid_size = maze.width * maze.height;

        let automaton_rng = StdRng::seed_from_u64(rng.next_u64());

        MazeAutomaton {
            x,
            y,
            orientation,
            internal_state: 0,
            env_bits,
            state_bits,
            resp_bits,
            state_mask,
            resp_mask,
            env_mask,
            energy: 15,
            energy_grid: vec![1u8; grid_size],
            grid_width: maze.width,
            fitness: 0.0,
            last_action: -1,
            genetic_code,
            rng: automaton_rng,
            seed,
        }
    }

    /// Whether this automaton still has energy and should continue ticking.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.energy > 0
    }

    /// Perform one tick: look up the environment, advance the Mealy machine,
    /// pay the energy cost, and execute the resulting action.
    pub fn tick(&mut self, maze: &Maze) {
        let env_local = maze.get_local(self.x, self.y, self.orientation);
        let input_code =
            ((self.internal_state as u32) << self.env_bits) | (env_local & self.env_mask) as u32;
        let output_code = self.genetic_code.get(input_code);
        self.internal_state = output_code & self.state_mask;
        let action = output_code >> self.state_bits;

        self.energy -= 1;
        self.attempt_action(action & self.resp_mask, maze);
    }

    /// Apply one action.  Mirrors Python's `MazeAutomaton.attempt_action`.
    fn attempt_action(&mut self, action: u8, maze: &Maze) {
        self.last_action = action as i32;
        match action {
            0 => {
                // Move forward.
                let (dx, dy) = ORIENTATION_MOVES[self.orientation as usize];
                let nx = self.x as i32 + dx;
                let ny = self.y as i32 + dy;

                // Bounds check (maze borders are always walls, so OOB = wall).
                if nx < 0 || ny < 0 || nx >= maze.width as i32 || ny >= maze.height as i32 {
                    self.fitness -= 0.05;
                    return;
                }
                let nx = nx as usize;
                let ny = ny as usize;

                if maze.is_wall(nx, ny) {
                    self.fitness -= 0.05;
                    return;
                }

                // Move succeeds.
                let idx = ny * self.grid_width + nx;
                let energy_gain = self.energy_grid[idx];
                self.fitness += energy_gain as f64 + 0.1;
                self.energy += (energy_gain as i32) * 2;
                self.energy_grid[idx] = 0;
                self.x = nx;
                self.y = ny;
            }
            1 => {
                // Turn left.
                self.orientation = self.orientation.wrapping_sub(1) & 3;
            }
            2 => {
                // Turn right.
                self.orientation = (self.orientation + 1) & 3;
            }
            _ => {
                // Invalid action.
                self.fitness -= 0.1;
            }
        }
    }

    /// Reset position, energy, energy grid, fitness, and internal state.
    /// The genetic code is preserved.  Mirrors Python's `MazeAutomaton.reset`.
    pub fn reset(&mut self, maze: &Maze) {
        let (x, y) = maze.random_free_cell(&mut self.rng);
        self.x = x;
        self.y = y;
        self.orientation = (self.rng.next_u32() & 3) as u8;
        self.energy = 15;
        let grid_size = maze.width * maze.height;
        self.energy_grid.iter_mut().for_each(|e| *e = 1);
        if self.energy_grid.len() != grid_size {
            self.energy_grid = vec![1u8; grid_size];
        }
        self.fitness = 0.0;
        self.internal_state = 0;
        self.last_action = -1;
    }
}
