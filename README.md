# engine-compile-cuda

Lowers a kernel IR `Term` to CUDA C++ source: one templated `__global__`
kernel plus an `extern "C"` launcher, all real CUDA — no pseudocode.

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

Two fragments lower; everything else is declined **by name** (error,
never a stub).

**Elementwise/layout**: `store(body, out)` over `mul`,
`sub`, `neg`, `relu`, `exp`, `sigmoid`, `tanh`, `gelu`, `silu`,
`bias`, with `transpose`/`permute`/`tile_load`/`pipeline` as views and
identities.

**Contract / qgemv**: `store(body, out)` where `body`
is `dot`, `split` of a dot, `reduce` of dot partials, and `bias`
wrappers, in any `pipeline`/`stage` nesting. Each dot's left operand is
either a leaf/view-chain `[k, n]` tile or the AWQ spelling
`dequant(unpack_i4(w[k, n/8], shifts[8]), z[g, n], s[g, n], group)`
with `kind:"g"` leaves (`g = k / div` per element — the denotation's
`repeat_interleave`). One fused kernel accumulates in fp32 — the
compute type `dot`/`stage`/`reduce` all denote — and casts on store;
`split`'s k literal is checked against `geom` at launch. This is the
`qgemv()` term `lib-kernel-ir` emits, lowered for real.

Still declined **by name**: `dequant`/`unpack_i4` outside a dot lhs
(they denote materialised tiles this crate doesn't stage), `reduce`
partials that aren't dots, `split` of a non-dot body (a k-axis
reduction whose k axis can't be identified from the term alone), and
`dot` over a computed tile. Batched GEMM / attention is the measured
wall in the coverage census; this crate does not
claim it. A `transpose`/`permute` over a *computed* tile is also
refused — a view of non-leaf data has no stride-only transcription.

## Development

```bash
cargo run --manifest-path xtask/Cargo.toml -- gate   # fmt → clippy → test → doc
cargo run --quiet -- < term.json > k.cu
nvcc -c k.cu -o k.o     # if a CUDA toolkit is installed (not required)
```

Tests are structural (kernel signature, launch config, dtype dispatch,
no unsupported op silently dropped) plus an `nvcc -c` syntax check that
skips honestly when no CUDA toolkit is present. No GPU execution in CI.
`xtask/` provides the gate tasks (`fmt`, `clippy`, `test`, `doc`, `check`,
`gate`); `ci.sh` is the same gate entry for the bupha shell runner. Deps
are git deps on `branch = "main"`; locally the workspace-root
`.cargo/config.toml` patches them to sibling checkouts (dev-only, never
committed).

## License

MIT.
