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
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::path::Path;
use std::time::Duration;

const BASE_SIZE: usize = 64;
const OUTPUT_PATH: &str = "target/frame_sequence.png";
const DELTA_OUTPUT_PATH: &str = "target/delta_sequence.png";
const QUANTIZATION_FACTORS: [usize; 4] = [2, 4, 8, 16];
const PANEL_GAP_PX: u32 = 4;
const FRAME_ROW_GAP_PX: u32 = 8;
const DEFAULT_API_BASE_URL: &str = "http://127.0.0.1:8000";
const DEFAULT_ACTION_SEED: u64 = 42;
const REQUEST_COUNT: usize = 10;
const RANDOM_ACTIONS: [&str; 4] = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"];
const MAX_PERCEPTUAL_DIST: f64 = 255.0;

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

fn perceptual_color_distance(c1: u8, c2: u8) -> f64 {
    let p1 = PALETTE[c1 as usize];
    let p2 = PALETTE[c2 as usize];
    let dr = p1[0] as f64 - p2[0] as f64;
    let dg = p1[1] as f64 - p2[1] as f64;
    let db = p1[2] as f64 - p2[2] as f64;
    (0.299 * dr * dr + 0.587 * dg * dg + 0.114 * db * db).sqrt()
}

/// Map a normalised saliency value in [0.0, 1.0] to a 4-bit palette index in [0, 15].
/// Uses uniform linear quantization: bin = round(value * 15), clamped to [0, 15].
fn quantize_delta_value(value: f64) -> u8 {
    (value * 15.0).round().clamp(0.0, 15.0) as u8
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

    fn multiscale_grids_raw(&self) -> Vec<Vec<Vec<u8>>> {
        let mut grids = Vec::with_capacity(1 + QUANTIZATION_FACTORS.len());
        grids.push(self.base_grid());

        for factor in QUANTIZATION_FACTORS {
            grids.push(self.quantized_grid(factor));
        }

        grids
    }

    fn change_delta_grid(&self, previous: &Frame64, factor: usize) -> Vec<Vec<u8>> {
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

        // 2. Compute base change saliency map (64x64)
        let mut base_saliency = vec![vec![0.0; BASE_SIZE]; BASE_SIZE];
        let gamma = 4.0;

        for y in 0..BASE_SIZE {
            for x in 0..BASE_SIZE {
                let c_prev = previous.cells[y][x];
                let c_curr = self.cells[y][x];
                let d_temp = perceptual_color_distance(c_prev, c_curr);

                if d_temp > 0.0 {
                    let d_bg_prev = perceptual_color_distance(c_prev, global_bg);
                    let d_bg_curr = perceptual_color_distance(c_curr, global_bg);
                    let max_bg_dist = d_bg_prev.max(d_bg_curr);

                    let boost = 1.0 + gamma * (max_bg_dist / MAX_PERCEPTUAL_DIST);
                    base_saliency[y][x] = d_temp * boost;
                }
            }
        }

        // 3. Aggregate base saliency map to target scale using p-norm (p=3)
        //    and quantize the normalised value to 4 bits (0..=15) linearly.
        let quantized_size = BASE_SIZE / factor;
        let mut delta_grid = vec![vec![0_u8; quantized_size]; quantized_size];
        let p = 3.0;

        for y in 0..quantized_size {
            for x in 0..quantized_size {
                let mut sum_power = 0.0;
                for dy in 0..factor {
                    for dx in 0..factor {
                        let val = base_saliency[y * factor + dy][x * factor + dx];
                        sum_power += val.powf(p);
                    }
                }

                let count = (factor * factor) as f64;
                let norm_val = (sum_power / count).powf(1.0 / p);

                let max_possible_saliency = MAX_PERCEPTUAL_DIST * (1.0 + gamma);
                let normalised = (norm_val / max_possible_saliency).min(1.0);
                delta_grid[y][x] = quantize_delta_value(normalised);
            }
        }

        delta_grid
    }

    fn multiscale_delta_grids_raw(&self, previous: &Frame64) -> Vec<Vec<Vec<u8>>> {
        let mut grids = Vec::with_capacity(1 + QUANTIZATION_FACTORS.len());
        // Base resolution (factor 1)
        grids.push(self.change_delta_grid(previous, 1));

        for factor in QUANTIZATION_FACTORS {
            grids.push(self.change_delta_grid(previous, factor));
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

fn load_action_seed() -> u64 {
    match env::var("ARC3_ACTION_SEED") {
        Ok(raw) => raw.parse::<u64>().unwrap_or_else(|error| {
            panic!("Failed to parse ARC3_ACTION_SEED='{}' as u64: {error}", raw)
        }),
        Err(_) => DEFAULT_ACTION_SEED,
    }
}

fn request_frame_for_action(
    client: &Client,
    endpoint: &str,
    action: &str,
    request_index: usize,
) -> Vec<Vec<u8>> {
    let response = client
        .post(endpoint)
        .json(&ActionRequest { action })
        .send()
        .unwrap_or_else(|error| {
            panic!(
                "Failed to request frame {} from {} with action {}: {error}",
                request_index + 1,
                endpoint,
                action
            )
        });

    let response = response.error_for_status().unwrap_or_else(|error| {
        panic!(
            "Frame request {} failed for {} with action {}: {error}",
            request_index + 1,
            endpoint,
            action
        )
    });

    let payload: ActionResponse = response.json().unwrap_or_else(|error| {
        panic!(
            "Failed to parse JSON for frame request {} from {}: {error}",
            request_index + 1,
            endpoint
        )
    });

    if payload.state.is_empty() {
        panic!(
            "Frame request {} from {} had empty state",
            request_index + 1,
            endpoint
        );
    }
    let _step = payload.step;
    let _reset_performed = payload.reset_performed;

    let frame = extract_first_frame(payload.frame);
    validate_frame_shape(
        &frame,
        &format!("REST API response.frame[0] request {}", request_index + 1),
    );
    frame
}

fn load_frame_sequence_from_api(
    base_url: &str,
    action_seed: u64,
) -> (Vec<Vec<Vec<u8>>>, Vec<String>) {
    let endpoint = format!("{}/action", base_url.trim_end_matches('/'));
    let client = Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap_or_else(|error| panic!("Failed to build HTTP client: {error}"));

    let mut rng = StdRng::seed_from_u64(action_seed);
    let mut frames = Vec::with_capacity(REQUEST_COUNT);
    let mut actions = Vec::with_capacity(REQUEST_COUNT);
    let mut frame_history = Vec::with_capacity(REQUEST_COUNT);
    let mut quantized_history = Vec::with_capacity(REQUEST_COUNT);
    let mut delta_history = Vec::with_capacity(REQUEST_COUNT);

    let mut action = "RESET";
    for request_index in 0..REQUEST_COUNT {
        actions.push(action.to_string());
        let frame = request_frame_for_action(&client, &endpoint, action, request_index);

        // Check there is a non-empty frame before calling Terry for the next action.
        if frame.is_empty() {
            panic!(
                "Received an empty frame for request {} from {}; cannot determine next action",
                request_index + 1,
                endpoint
            );
        } else {
            let current_frame = Frame64::new(frame.clone());
            let quantized_grids = current_frame.multiscale_grids_raw();
            let delta_grids = if let Some(previous_frame) = frame_history.last() {
                current_frame.multiscale_delta_grids_raw(previous_frame)
            } else {
                zeroed_delta_grids()
            };

            frame_history.push(current_frame);
            quantized_history.push(quantized_grids);
            delta_history.push(delta_grids);

            action = terry(&quantized_history, &delta_history, &mut rng);
        }

        frames.push(frame);
    }

    (frames, actions)
}

fn terry(
    _quantized_frames: &[Vec<Vec<Vec<u8>>>],
    _delta_frames: &[Vec<Vec<Vec<u8>>>],
    rng: &mut StdRng,
) -> &'static str {
    // Placeholder for the actual Terry logic that determines the next action using
    // quantized frame history and delta frame history.
    // For now, it just returns a random action from the predefined list.
    let _qframe = _quantized_frames
        .last()
        .and_then(|frame| frame.last())
        .unwrap_or_else(|| panic!("Terry was called with no quantized frames"));
    let _dframe = _delta_frames
        .last()
        .and_then(|frame| frame.last())
        .unwrap_or_else(|| panic!("Terry was called with no delta frames"));
    let action_index = rng.random_range(0..RANDOM_ACTIONS.len());
    RANDOM_ACTIONS[action_index]
}

fn render_frame_sequence(frame_grids: &[Vec<Vec<Vec<u8>>>]) {
    // Render each frame as one row and each quantization level as one panel in that row.

    if frame_grids.is_empty() {
        panic!("No frame grids provided for rendering");
    }
    if frame_grids[0].is_empty() {
        panic!("First frame has no quantization panels");
    }

    let panel_size = BASE_SIZE as u32;
    let panel_count = frame_grids[0].len() as u32;
    let frame_count = frame_grids.len() as u32;
    let output_width = panel_count * panel_size + (panel_count - 1) * PANEL_GAP_PX;
    let output_height = frame_count * panel_size + (frame_count - 1) * FRAME_ROW_GAP_PX;
    let mut image = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(output_width, output_height);

    for (frame_index, grids) in frame_grids.iter().enumerate() {
        if grids.len() as u32 != panel_count {
            panic!(
                "Frame {frame_index} has {} panels; expected {}",
                grids.len(),
                panel_count
            );
        }

        let frame_offset_y = frame_index as u32 * (panel_size + FRAME_ROW_GAP_PX);
        for (panel_index, grid) in grids.iter().enumerate() {
            let grid_size = grid.len();
            if grid_size == 0 || grid.iter().any(|row| row.len() != grid_size) {
                panic!("Grid at frame {frame_index}, panel {panel_index} is not square");
            }
            if BASE_SIZE % grid_size != 0 {
                panic!(
                    "Grid at frame {frame_index}, panel {panel_index} has unsupported size {grid_size}"
                );
            }

            let scale = (BASE_SIZE / grid_size) as u32;
            let panel_offset_x = panel_index as u32 * (panel_size + PANEL_GAP_PX);

            for (y, row) in grid.iter().enumerate() {
                for (x, value) in row.iter().enumerate() {
                    let color = PALETTE.get(*value as usize).unwrap_or_else(|| {
                        panic!(
                            "Cell value {} at frame {}, panel {}, ({}, {}) is out of palette range 0..=15",
                            value, frame_index, panel_index, x, y
                        )
                    });

                    let start_x = panel_offset_x + x as u32 * scale;
                    let start_y = frame_offset_y + y as u32 * scale;
                    for dy in 0..scale {
                        for dx in 0..scale {
                            image.put_pixel(start_x + dx, start_y + dy, Rgba(*color));
                        }
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

fn zeroed_delta_grids() -> Vec<Vec<Vec<u8>>> {
    let mut grids = Vec::with_capacity(1 + QUANTIZATION_FACTORS.len());
    // Base resolution (factor 1)
    grids.push(vec![vec![0_u8; BASE_SIZE]; BASE_SIZE]);

    // Quantized levels
    for factor in QUANTIZATION_FACTORS {
        let size = BASE_SIZE / factor;
        grids.push(vec![vec![0_u8; size]; size]);
    }
    grids
}

fn render_delta_sequence(delta_grids: &[Vec<Vec<Vec<u8>>>]) {
    if delta_grids.is_empty() {
        panic!("No delta grids provided for rendering");
    }
    if delta_grids[0].is_empty() {
        panic!("First delta frame has no quantization panels");
    }

    let panel_size = BASE_SIZE as u32;
    let panel_count = delta_grids[0].len() as u32;
    let frame_count = delta_grids.len() as u32;
    let output_width = panel_count * panel_size + (panel_count - 1) * PANEL_GAP_PX;
    let output_height = frame_count * panel_size + (frame_count - 1) * FRAME_ROW_GAP_PX;
    let mut image = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(output_width, output_height);

    for (frame_index, grids) in delta_grids.iter().enumerate() {
        if grids.len() as u32 != panel_count {
            panic!(
                "Delta frame {frame_index} has {} panels; expected {}",
                grids.len(),
                panel_count
            );
        }

        let frame_offset_y = frame_index as u32 * (panel_size + FRAME_ROW_GAP_PX);
        for (panel_index, grid) in grids.iter().enumerate() {
            let grid_size = grid.len();
            if grid_size == 0 || grid.iter().any(|row| row.len() != grid_size) {
                panic!("Delta grid at frame {frame_index}, panel {panel_index} is not square");
            }
            if BASE_SIZE % grid_size != 0 {
                panic!(
                    "Delta grid at frame {frame_index}, panel {panel_index} has unsupported size {grid_size}"
                );
            }

            let scale = (BASE_SIZE / grid_size) as u32;
            let panel_offset_x = panel_index as u32 * (panel_size + PANEL_GAP_PX);

            for (y, row) in grid.iter().enumerate() {
                for (x, &value) in row.iter().enumerate() {
                    // Grayscale mapping: quantized 4-bit index (0..=15) mapped linearly
                    // back to full 8-bit intensity so the rendered image mirrors the
                    // discrete state that is passed into terry().
                    let intensity = ((value as u16 * 255) / 15) as u8;
                    let color = [intensity, intensity, intensity, 0xFF];

                    let start_x = panel_offset_x + x as u32 * scale;
                    let start_y = frame_offset_y + y as u32 * scale;
                    for dy in 0..scale {
                        for dx in 0..scale {
                            image.put_pixel(start_x + dx, start_y + dy, Rgba(color));
                        }
                    }
                }
            }
        }
    }

    let output_path = Path::new(DELTA_OUTPUT_PATH);
    image.save(output_path).unwrap_or_else(|error| {
        panic!("Failed to save {}: {error}", output_path.display());
    });
}

fn main() {
    // Request a sequence of frames from REST API, then render each frame as a row over all quantizations.
    let base_url = env::var("ARC3_API_URL").unwrap_or_else(|_| DEFAULT_API_BASE_URL.to_string());
    let action_seed = load_action_seed();
    let (raw_frames, actions) = load_frame_sequence_from_api(&base_url, action_seed);

    let frames: Vec<Frame64> = raw_frames.into_iter().map(Frame64::new).collect();

    let frame_grids: Vec<Vec<Vec<Vec<u8>>>> = frames
        .iter()
        .map(|frame| frame.multiscale_grids_raw())
        .collect();

    println!(
        "Loaded {} frames from {} using seed {}",
        frame_grids.len(),
        base_url,
        action_seed
    );
    println!("Action sequence: {}", actions.join(", "));

    render_frame_sequence(&frame_grids);
    println!(
        "Rendered frame sequence panels (64, 32, 16, 8, 4) to {}",
        OUTPUT_PATH
    );

    // Compute consecutive deltas
    let mut delta_grids: Vec<Vec<Vec<Vec<u8>>>> = Vec::with_capacity(frames.len());
    for i in 0..frames.len() {
        if i == 0 {
            delta_grids.push(zeroed_delta_grids());
        } else {
            delta_grids.push(frames[i].multiscale_delta_grids_raw(&frames[i - 1]));
        }
    }

    render_delta_sequence(&delta_grids);
    println!(
        "Rendered frame delta sequence panels (64, 32, 16, 8, 4) to {}",
        DELTA_OUTPUT_PATH
    );
}
