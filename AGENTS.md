# tool-compile-cuda

hnl IR terms → CUDA C++ — lowers a kernel IR `Term` to one templated
`__global__` kernel plus an `extern "C"` launcher (ADR-0171 fragment).
Rust 2024, `publish = false` (git-install only), crate `tool-compile-cuda`
(lib `tool_compile_cuda` + bins `tool-compile-cuda`, `kernel-lower`).

## Layout

| path | role |
| --- | --- |
| `src/lib.rs` | crate docs; `compile(node)` — tries `as_elementwise` first, falls through to `as_contract` on `store` refusal, names both errors |
| `src/main.rs` | `tool-compile-cuda` bin — term JSON on stdin → `.cu` translation unit on stdout |
| `src/bin/kernel-lower.rs` | `kernel-lower` bin — `Emitter<KernelTerm>` materialization to a change-set manifest record (PL-004 T-094); `--target triton\|cuda`, `--manifest-dir` for the apply-shim form |
| `src/kernel_term.rs` | `CudaEmitter` — `impl Emitter<KernelTerm>` for CUDA (`.cu` artifact via `artifact_name`); structural `verify` |
| `src/codegen.rs` | elementwise/layout emit — `emit(&Elementwise)`; flat grid-stride kernel, `launch_<slug>` dtype dispatch (f32/f64/f16/bf16/i32/i64), `cuda_emit_host::elem_*` helpers |
| `src/contract.rs` | contract/qgemv emit — `as_contract`/`emit_contract`, `Dot`/`Contract`; fused matvec kernel for `dot`/`split`/`reduce`/`bias`, plain `[k,n]` tile or AWQ `dequant(unpack_i4(…))` lhs |
| `tests/codegen_test.rs`, `tests/contract_test.rs` | emitted-source contract pins + `nvcc -c` check that skips when no toolkit |
| `xtask/`, `workflows/gate.yaml` | vendored repo gate + CI wiring (sibling convention) |

## Contract

- **Input**: hnl_codec Term JSON — the same `{"Op"/"Arr"/"Const"}` /
  `dag.program` wire `tool-compile-triton` parses (`tool_compile_triton::term::Node`).
- **Output**: one self-contained CUDA C++ TU — templated kernel,
  `extern "C" launch_<slug>(dtype, in_ptrs, out_ptr, geom, block,
  stream)`, six dtype instantiations. `geom` packs the broadcast canon
  shape + effective (post-view) strides; callers replay
  `transpose`/`permute` views on shape/stride pairs host-side.
- **Two fragments, named refusals**: elementwise/layout covers
  `store(body, out)` over the pointwise ops (`mul`, `sub`, `neg`,
  `relu`, `exp`, `sigmoid`, `tanh`, `gelu`, `silu`, `bias`) with
  `transpose`/`permute`/`tile_load`/`pipeline` as views. The contract
  fragment covers `dot` (plain tile or AWQ dequant lhs), `split` of a
  dot (k literal checked against `geom` at launch), `reduce` of dot
  partials, `bias` wrappers — all accumulating in fp32, casting on
  store. Everything else — `dequant`/`unpack_i4` outside a dot lhs,
  non-dot `reduce` partials, `split` of a non-dot, computed dot
  operands, batched GEMM/attention — is refused **by name**, never a
  stub. `compile()` surfaces both paths' reasons when both refuse.
- **Honest scope**: no GPU execution in CI; `Emitter::verify` is
  structural only (one `Put` on the `artifact_name` slug) — artifact
  verify stays producer-side until a CUDA runner exists.

## Conventions

- `tool-compile-triton` supplies `KernelTerm`/`Node`/`as_elementwise` and
  `artifact_name`; `hnl-tool-codegen` supplies `Emitter`/`Change`/
  `EmitContext`/`ContentHash`. Both are git deps on `branch = "main"`
  — no tag pins yet.
- No `[patch]` tables in the manifest — committed manifests resolve
  from git alone; local sibling overrides live in the gitignored
  workspace-root `.cargo/config.toml` (engine-public-readiness D7).
- `kernel-lower`'s manifest wire mirrors kernelopt's
  `lower_candidate`: `hash_algorithm: "blake3-256"`, `blobs/<hash>`
  payloads, inline `content_b64` record on stdout /
  `blob_ref` + `manifest.json` under `--manifest-dir`.
- Edition 2024, `#![forbid(unsafe_code)]`, `#![warn(missing_docs)]`.
  English everywhere.

## Verification

```bash
cargo run --manifest-path xtask/Cargo.toml -- gate   # fmt → clippy → test → doc
cargo run --quiet -- < term.json > k.cu              # stdin term → .cu
nvcc -c k.cu -o k.o     # if a CUDA toolkit is installed (optional; tests skip honestly without it)
```

Tests are structural (kernel signature, launch config, dtype dispatch,
named refusals) plus an `nvcc -c` syntax check that skips when no CUDA
toolkit is present. No GPU execution in CI.

<!-- myco:agent-awareness -->

## Agent awareness

- Session start / any work discussion: run `myco status --prompt`
  before proposing next steps — know the board first.
- User describes new work → offer to capture it (`myco todo add`,
  `issue new`, `proposal new`). Don't wait to be asked.
- Picking up a task: `myco resume <ID>` is the briefing.
- Wrapping up: `myco handoff <ID>` anything resumable; keep statuses
  honest (claimed → in-progress → blocked/done).

<!-- /myco:agent-awareness -->
