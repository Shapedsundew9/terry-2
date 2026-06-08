// Welcome to Terry-2
//
// Terry is a cellular automata simulator written in Rust. It is designed to be super fast so we can
// simulate large worlds with many different types of cells.
// Anything that can be determined at build time is determined at build time, so we can optimize the
// code for the specific world we want to simulate. This means that we can have a very large number
// of different cell types without sacrificing performance.
//
// The ultimate goal is to create an automata that can solve https://docs.arcprize.org/
//
// However, step 1 is simply to choose the right action for the first move of the first level of the
// simplest game, "ls20".
// Since ARC3 is implemented in python, we request the initial frame from a REST API by
// posting GameAction.RESET to /action and consuming response.frame.
//
// The frame is 64x64

use image::{ImageBuffer, Rgba};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::path::Path;
use std::time::Duration;

const BASE_SIZE: usize = 64;
const OUTPUT_PATH: &str = "arc3_agi/initial_frame.png";
const QUANTIZATION_FACTORS: [usize; 4] = [2, 4, 8, 16];
const PANEL_GAP_PX: u32 = 4;
const DEFAULT_API_BASE_URL: &str = "http://127.0.0.1:8000";

const PALETTE: [[u8; 4]; 16] = [
    [0xFF, 0xFF, 0xFF, 0xFF], // 0: White
    [0xCC, 0xCC, 0xCC, 0xFF], // 1: Off-white
    [0x99, 0x99, 0x99, 0xFF], // 2: Neutral Light
    [0x66, 0x66, 0x66, 0xFF], // 3: Neutral
    [0x33, 0x33, 0x33, 0xFF], // 4: Off Black
    [0x00, 0x00, 0x00, 0xFF], // 5: Black
    [0xE5, 0x3A, 0xA3, 0xFF], // 6: Magenta
    [0xFF, 0x7B, 0xCC, 0xFF], // 7: Magenta Light
    [0xF9, 0x3C, 0x31, 0xFF], // 8: Red
    [0x1E, 0x93, 0xFF, 0xFF], // 9: Blue
    [0x88, 0xD8, 0xF1, 0xFF], // 10: Blue Light
    [0xFF, 0xDC, 0x00, 0xFF], // 11: Yellow
    [0xFF, 0x85, 0x1B, 0xFF], // 12: Orange
    [0x92, 0x12, 0x31, 0xFF], // 13: Maroon
    [0x4F, 0xCC, 0x30, 0xFF], // 14: Green
    [0xA3, 0x56, 0xD6, 0xFF], // 15: Purple
];

const MAX_DIST: f64 = 441.67295593;

#[derive(Serialize)]
struct ActionRequest<'a> {
    action: &'a str,
}

#[derive(Deserialize)]
struct ActionResponse {
    state: String,
    frame: serde_json::Value,
    step: u64,
    reset_performed: bool,
}

fn extract_first_frame(frame_value: serde_json::Value) -> Vec<Vec<u8>> {
    if let Ok(frame_2d) = serde_json::from_value::<Vec<Vec<u8>>>(frame_value.clone()) {
        return frame_2d;
    }

    let frame_3d = serde_json::from_value::<Vec<Vec<Vec<u8>>>>(frame_value)
        .unwrap_or_else(|error| panic!("Frame payload was neither 2D nor 3D u8 data: {error}"));

    frame_3d
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("Frame payload was empty; expected at least one layer"))
}

fn color_distance(c1: u8, c2: u8) -> f64 {
    let p1 = PALETTE[c1 as usize];
    let p2 = PALETTE[c2 as usize];
    let dr = p1[0] as f64 - p2[0] as f64;
    let dg = p1[1] as f64 - p2[1] as f64;
    let db = p1[2] as f64 - p2[2] as f64;
    (dr * dr + dg * dg + db * db).sqrt()
}

struct Frame64 {
    cells: Vec<Vec<u8>>,
}

impl Frame64 {
    // Keep source frame data in u8 form and enforce the 64x64 invariant once.
    fn new(cells: Vec<Vec<u8>>) -> Self {
        if cells.len() != BASE_SIZE || cells.iter().any(|row| row.len() != BASE_SIZE) {
            panic!(
                "Expected a 64x64 frame, got {}x{}",
                cells.len(),
                cells.first().map_or(0, |row| row.len())
            );
        }

        Self { cells }
    }

    fn base_grid(&self) -> Vec<Vec<u8>> {
        self.cells.clone()
    }

    // Compute quantized grids on demand using contrast-weighted salience quantization.
    // This preserves high-contrast details/foreground objects even at low resolutions.
    fn quantized_grid(&self, factor: usize) -> Vec<Vec<u8>> {
        if factor == 0 || BASE_SIZE % factor != 0 {
            panic!("Quantization factor {factor} must evenly divide {BASE_SIZE}");
        }

        // 1. Identify global background color (the most frequent color in the 64x64 frame).
        let mut global_counts = [0_u32; 16];
        for row in &self.cells {
            for &val in row {
                if val < 16 {
                    global_counts[val as usize] += 1;
                }
            }
        }
        let global_bg = global_counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, count)| count)
            .map(|(val, _)| val as u8)
            .unwrap_or(0);

        // 2. Perform contrast-weighted quantization.
        let quantized_size = BASE_SIZE / factor;
        let mut quantized = vec![vec![0_u8; quantized_size]; quantized_size];

        // Weight determines the priority of contrast over area.
        // Alpha determines the scale/influence of pixel count.
        let weight = 8.0;
        let alpha = 0.3;

        for y in 0..quantized_size {
            for x in 0..quantized_size {
                let mut counts = [0_u32; 16];
                for dy in 0..factor {
                    for dx in 0..factor {
                        let val = self.cells[y * factor + dy][x * factor + dx];
                        if val < 16 {
                            counts[val as usize] += 1;
                        }
                    }
                }

                let mut best_color = 0_u8;
                let mut best_score = -1.0;

                for color in 0..16 {
                    let count = counts[color];
                    if count > 0 {
                        let d = color_distance(color as u8, global_bg);
                        // Score = count^alpha * (1.0 + weight * (d / MAX_DIST))
                        let score = (count as f64).powf(alpha) * (1.0 + weight * (d / MAX_DIST));
                        if score > best_score {
                            best_score = score;
                            best_color = color as u8;
                        }
                    }
                }

                quantized[y][x] = best_color;
            }
        }

        quantized
    }

    fn grids_for_rendering(&self) -> Vec<Vec<Vec<u8>>> {
        let mut grids = Vec::with_capacity(1 + QUANTIZATION_FACTORS.len());
        grids.push(self.base_grid());

        for factor in QUANTIZATION_FACTORS {
            grids.push(self.quantized_grid(factor));
        }

        grids
    }
}

fn validate_frame_shape(frame: &[Vec<u8>], source: &str) {
    if frame.len() != BASE_SIZE || frame.iter().any(|row| row.len() != BASE_SIZE) {
        panic!(
            "Expected a 64x64 frame from {source}, got {}x{}",
            frame.len(),
            frame.first().map_or(0, |row| row.len())
        );
    }
}

fn load_initial_frame_from_api(base_url: &str) -> Vec<Vec<u8>> {
    let endpoint = format!("{}/action", base_url.trim_end_matches('/'));
    let client = Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap_or_else(|error| panic!("Failed to build HTTP client: {error}"));

    let response = client
        .post(&endpoint)
        .json(&ActionRequest { action: "RESET" })
        .send()
        .unwrap_or_else(|error| {
            panic!("Failed to request initial frame from {}: {error}", endpoint)
        });

    let response = response
        .error_for_status()
        .unwrap_or_else(|error| panic!("Initial frame request failed for {}: {error}", endpoint));

    let payload: ActionResponse = response
        .json()
        .unwrap_or_else(|error| panic!("Failed to parse JSON response from {}: {error}", endpoint));

    if payload.state.is_empty() {
        panic!("Initial frame response from {} had empty state", endpoint);
    }
    let _step = payload.step;
    let _reset_performed = payload.reset_performed;

    let initial_frame = extract_first_frame(payload.frame);
    validate_frame_shape(&initial_frame, "REST API response.frame[0]");
    initial_frame
}

fn render_frame_panels(grids: &[Vec<Vec<u8>>]) {
    // Render every resolution into one horizontal image, scaling each panel up to 64x64.

    if grids.is_empty() {
        panic!("No grids provided for rendering");
    }

    let panel_size = BASE_SIZE as u32;
    let panel_count = grids.len() as u32;
    let output_width = panel_count * panel_size + (panel_count - 1) * PANEL_GAP_PX;
    let output_height = panel_size;
    let mut image = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(output_width, output_height);

    for (panel_index, grid) in grids.iter().enumerate() {
        let grid_size = grid.len();
        if grid_size == 0 || grid.iter().any(|row| row.len() != grid_size) {
            panic!("Grid at index {panel_index} is not square");
        }
        if BASE_SIZE % grid_size != 0 {
            panic!("Grid at index {panel_index} has unsupported size {grid_size}");
        }

        let scale = (BASE_SIZE / grid_size) as u32;
        let panel_offset_x = panel_index as u32 * (panel_size + PANEL_GAP_PX);

        for (y, row) in grid.iter().enumerate() {
            for (x, value) in row.iter().enumerate() {
                let color = PALETTE.get(*value as usize).unwrap_or_else(|| {
                    panic!(
                        "Cell value {} at panel {}, ({}, {}) is out of palette range 0..=15",
                        value, panel_index, x, y
                    )
                });

                let start_x = panel_offset_x + x as u32 * scale;
                let start_y = y as u32 * scale;
                for dy in 0..scale {
                    for dx in 0..scale {
                        image.put_pixel(start_x + dx, start_y + dy, Rgba(*color));
                    }
                }
            }
        }
    }

    let output_path = Path::new(OUTPUT_PATH);
    image.save(output_path).unwrap_or_else(|error| {
        panic!("Failed to save {}: {error}", output_path.display());
    });
}

fn main() {
    // Load source data from REST API, derive quantized u8 grids on demand, and render all scales together.
    let base_url = env::var("ARC3_API_URL").unwrap_or_else(|_| DEFAULT_API_BASE_URL.to_string());
    let initial_frame = load_initial_frame_from_api(&base_url);
    let frame = Frame64::new(initial_frame);
    let grids = frame.grids_for_rendering();

    println!(
        "Loaded initial frame from {}: {}x{}",
        base_url,
        grids[0].len(),
        grids[0][0].len()
    );

    render_frame_panels(&grids);
    println!(
        "Rendered composite frame panels (64, 32, 16, 8, 4) to {}",
        OUTPUT_PATH
    );
}
