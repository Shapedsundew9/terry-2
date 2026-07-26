# CPU-specific Rust builds

Build all three optimized `maze-runner` variants from the repository root:

```bash
./scripts/build-cpu-variants.sh
```

The build uses the release profile (`opt-level = 3`) with thin link-time
optimization and one codegen unit. Each variant has a separate Cargo target
directory so switching CPU flags cannot reuse incompatible artifacts.

| Output | Rust `target-cpu` | Intended CPU |
| --- | --- | --- |
| `target/cpu-variants/maze-runner-pentium-3825u` | `x86-64-v2` | Intel Pentium 3825U |
| `target/cpu-variants/maze-runner-i5-5200u` | `broadwell` | Intel Core i5-5200U |
| `target/cpu-variants/maze-runner-ryzen-6900hx` | `znver3` | AMD Ryzen 9 6900HX |

The Pentium uses `x86-64-v2` because its Broadwell-derived core has AVX, AVX2,
FMA, BMI1, and BMI2 disabled. A normal `broadwell` build could execute an
unsupported instruction on that CPU. Zen 3 is the appropriate LLVM scheduling
model for the Zen 3+ Ryzen 6900HX core.

These executables are CPU-specific and should be run only on the named machine
or a compatible newer CPU. They are still dynamically linked Linux binaries, so
build them on a Linux distribution with a glibc version no newer than the target
machines use.

CPU-specific code generation can improve a CPU-bound workload, but the gain is
workload-dependent and should be measured. It is normally smaller than the
difference in core count and clock speed between these three machines.
