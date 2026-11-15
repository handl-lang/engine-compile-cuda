# compile-cuda

Lowers a kernel IR `Term` (ADR-0171 fragment, `lib-kernel-ir`) to CUDA C++
source: one templated `__global__` kernel plus an `extern "C"` launcher,
all real CUDA — no pseudocode.

The term is the input, the source is the output, and neither is a
template someone edited by hand for one shape:

```bash
echo '{"op":"store","args":[{"op":"silu","args":[...]},{"leaf":{"name":"out"}}]}' \
  | cargo run --quiet -- > k.cu
```

The emitted translation unit exposes
`launch_<slug>(dtype, in_ptrs, out_ptr, geom, block, stream)` plus a
small `cuda_emit_host::elem_*` helper section (broadcast canon shape,
effective strides, `geom` packing, and `transpose`/`permute` view
replay on shape/stride pairs — the IR prices layout ops at zero, and in
CUDA a view *is* a stride change, so the kernel only ever sees flat
stride-described loads). Six dtype instantiations are dispatched:
float, double, half, bfloat16, int32, int64.

## Coverage — honest scope

Lowered: the ADR-0171 elementwise/layout fragment — `store(body, out)`
over `mul`, `sub`, `neg`, `relu`, `exp`, `sigmoid`, `tanh`, `gelu`,
`silu`, `bias`, with `transpose`/`permute`/`tile_load`/`pipeline` as
views and identities.

Declined **by name** (error, never a stub): the qgemv ops — `dot`,
`dequant`, `unpack_i4`, `split`, `stage`, `reduce`. Batched GEMM /
attention is the measured wall in the coverage census
(`graphopt:docs/reports/kernel_ir_coverage.md`); this crate does not
claim it. A `transpose`/`permute` over a *computed* tile is also
refused — a view of non-leaf data has no stride-only transcription.

## Develop

```bash
cargo test                 # tests/codegen_test.rs — emitted-source contract
cargo run --quiet -- < term.json > k.cu
nvcc -c k.cu -o k.o        # if a CUDA toolkit is installed (not required)
bash ci.sh                 # the bupha gate: cargo check + cargo test
```

Tests are structural (kernel signature, launch config, dtype dispatch,
no unsupported op silently dropped) plus an `nvcc -c` syntax check that
skips honestly when no CUDA toolkit is present. No GPU execution in CI.

Dual-licensed MIT OR Apache-2.0 — see `LICENSE-MIT` / `LICENSE-APACHE`.
