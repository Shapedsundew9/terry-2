#!/usr/bin/env bash
set -euo pipefail

readonly ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly BINARY_NAME="maze-runner"
readonly OUTPUT_DIR="${CPU_VARIANT_OUTPUT_DIR:-${ROOT_DIR}/target/cpu-variants}"
readonly BUILD_ROOT="${CPU_VARIANT_BUILD_ROOT:-${ROOT_DIR}/target/cpu-builds}"

cd "${ROOT_DIR}"

host_triple="$(rustc -vV | sed -n 's/^host: //p')"
if [[ "${host_triple}" != x86_64-* ]]; then
    printf 'error: CPU variants require an x86_64 Rust host (found %s)\n' "${host_triple}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

build_variant() {
    local suffix="$1"
    local target_cpu="$2"
    local target_dir="${BUILD_ROOT}/${suffix}"
    local source_binary="${target_dir}/${host_triple}/release/${BINARY_NAME}"
    local output_binary="${OUTPUT_DIR}/${BINARY_NAME}-${suffix}"

    printf '\nBuilding %s with target-cpu=%s\n' "${output_binary##*/}" "${target_cpu}"
    CARGO_TARGET_DIR="${target_dir}" \
        RUSTFLAGS="-C target-cpu=${target_cpu}" \
        cargo build --release --locked --target "${host_triple}" --bin "${BINARY_NAME}"
    install -m 755 "${source_binary}" "${output_binary}"
}

build_variant "pentium-3825u" "x86-64-v2"
build_variant "i5-5200u" "broadwell"
build_variant "ryzen-6900hx" "znver3"

printf '\nCPU-specific binaries:\n'
find "${OUTPUT_DIR}" -maxdepth 1 -type f -name "${BINARY_NAME}-*" -printf '  %f\n' | sort