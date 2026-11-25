//! Elementwise term -> CUDA C++ source.
//!
//! The emitted translation unit is self-contained: one templated
//! `__global__` kernel over the broadcast output (a flat 1-D grid-stride
//! loop, each element's flat index decomposed into <=4 coordinates,
//! left-padded with size-1 dims — the same indexing scheme the Triton
//! emitter uses), one `extern "C"` launcher that dispatches on a dtype
//! enum, and a small host-helper section (`elem_*`) that computes the
//! broadcast canon shape, per-operand effective strides (broadcast and
//! absent dims contribute 0), packs the `geom` array, and replays the
//! term's `transpose`/`permute` views on a shape/stride pair — the IR
//! prices layout ops at zero because they are views, and in CUDA a view
//! *is* a stride change, so the kernel never sees them.
//!
//! Dtype honesty: the IR term carries no dtypes, so the kernel is
//! templated on the element type `T` and the launcher instantiates
//! float / double / half / bfloat16 / int32 / int64. Compute happens in
//! a "work" type `W`: the operand's own precision for float/double/int,
//! float for half/bf16 (matching torch's compute type for those dtypes).
//! Transcendentals evaluate in double and round back — the IR grades
//! `exp`/`sigmoid`/`tanh`/`gelu`/`silu` ROUNDS, so last-ulp differences
//! against libdevice are inside the grade.

use compile_triton::term::{Elementwise, Expr, Lit, Operand, View};

/// Emit a CUDA C++ translation unit for an elementwise/layout term.
///
/// The file exposes `launch_<slug>(dtype, in_ptrs, out_ptr, geom, block,
/// stream)` plus the `elem_*` host helpers and the `cuda_emit_dtype`
/// enum. `geom` layout (all `long long`):
/// `[numel, sh0, sh1, sh2, sh3, in0.s0..s3, in1.s0..s3, ..., out.s0..s3]`
/// where the sh* are the 4-dim left-padded broadcast canon shape and the
/// s* are element strides (0 for broadcast/absent dims).
///
/// Returns `Err` naming the op when the `Elementwise` was built by
/// hand with an op tag outside the lowered set — `as_elementwise`
/// cannot produce one, but `Expr` is public, so `emit` re-checks
/// rather than silently emitting the wrong math.
pub fn emit(e: &Elementwise) -> Result<String, String> {
    let n = e.operands.len();
    let slug = kernel_slug(e);

    // -- kernel parameters -------------------------------------------------
    // No `__restrict__`: `store(body, out)` may write a leaf the body
    // also reads (in-place `store(relu(x), x)`), and distinct leaf
    // names may still alias the same buffer — the IR carries no alias
    // information, so the kernel must not promise any.
    let in_params: Vec<String> = (0..n).map(|i| format!("const T* p{i}")).collect();
    let mut kparams: Vec<String> = in_params;
    kparams.push("T* out_p".to_string());
    kparams.push("long long nout".to_string());
    for d in 0..4 {
        kparams.push(format!("long long sh{d}"));
    }
    for i in 0..n {
        for d in 0..4 {
            kparams.push(format!("long long s{i}_{d}"));
        }
    }
    for d in 0..4 {
        kparams.push(format!("long long so_{d}"));
    }

    let res = gen_expr(&e.expr)?;
    let kbody = format!(
        r#"  using W = typename cuda_emit_detail::Work<T>::type;
  const long long gstride = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       i < nout; i += gstride) {{
    // decompose the flat row-major index into coords, sh3 fastest;
    // size-1 padding dims contribute coord 0 and stride 0
    long long c3 = i % sh3;
    long long r1 = i / sh3;
    long long c2 = r1 % sh2;
    long long r2 = r1 / sh2;
    long long c1 = r2 % sh1;
    long long c0 = r2 / sh1;
    W v = {res};
    out_p[c0*so_0 + c1*so_1 + c2*so_2 + c3*so_3] =
        cuda_emit_detail::store_cast<T>(v);
  }}"#
    );

    // -- launcher ------------------------------------------------------------
    // geom indices: 0=numel, 1..4=canon4, then n*4 input strides, then 4
    // output strides.
    let geom_idx_in = |i: usize, d: usize| 5 + i * 4 + d;
    let geom_idx_out = |d: usize| 5 + n * 4 + d;

    let mut cases = String::new();
    for (cname, ty) in [
        ("CUDA_EMIT_F32", "float"),
        ("CUDA_EMIT_F64", "double"),
        ("CUDA_EMIT_F16", "__half"),
        ("CUDA_EMIT_BF16", "__nv_bfloat16"),
        ("CUDA_EMIT_I32", "int"),
        ("CUDA_EMIT_I64", "long long"),
    ] {
        let ptrs: Vec<String> = (0..n)
            .map(|i| format!("(const {ty}*)in_ptrs[{i}]").to_string())
            .collect();
        let mut args: Vec<String> = ptrs;
        args.push(format!("({ty}*)out_ptr"));
        args.push("nout".to_string());
        args.extend([
            "sh0".to_string(),
            "sh1".to_string(),
            "sh2".to_string(),
            "sh3".to_string(),
        ]);
        for i in 0..n {
            for d in 0..4 {
                args.push(format!("geom[{}]", geom_idx_in(i, d)));
            }
        }
        for d in 0..4 {
            args.push(format!("geom[{}]", geom_idx_out(d)));
        }
        let arglist = args.join(",\n          ");
        cases.push_str(&format!(
            r#"    case {cname}:
      elem_kernel_{slug}<{ty}><<<grid, tb, 0, stream>>>(
          {arglist});
      break;
"#
        ));
    }

    let views_doc = operand_views_doc(&e.operands);

    Ok(format!(
        r#"// Generated by compile-cuda from a kernel IR term (ADR-0171 fragment).
// Elementwise/layout lowering: `store(body, out)` -> one flat CUDA
// kernel over the broadcast output. `transpose`/`permute` are views —
// they move no bytes in the IR — and here they are absorbed into the
// operand strides the caller passes (see elem_transpose/elem_permute).
// Do not edit: the expression tree below came from the term.
//
// Contract for the caller:
//   * in_ptrs/out_ptr are device pointers of `n_operands` read operands
//     and the store target, element type selected by `dtype`.
//   * geom is a long long[5 + 4*n_operands + 4] packed by elem_pack_geom:
//     [numel, canon0..3 (4-dim left-padded broadcast shape),
//      per-operand effective strides (4 each), output strides (4)].
//   * `block` is threads per block (0 -> 256, must be <= 1024);
//     `stream` may be NULL.
//   * The kernel is a grid-stride loop: any grid size >= 1 is correct,
//     the launcher caps grid.x at INT_MAX.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

namespace cuda_emit_detail {{

// Work type W: float/double/int compute in their own precision; half and
// bf16 compute in float (torch's compute type for those dtypes).
template <typename T>
struct Work {{ using type = T; }};
template <> struct Work<float> {{ using type = float; }};
template <> struct Work<double> {{ using type = double; }};
template <> struct Work<__half> {{ using type = float; }};
template <> struct Work<__nv_bfloat16> {{ using type = float; }};

// Transcendentals evaluate in double and round back to W. The IR grades
// these ops ROUNDS; this is inside the grade and keeps one code path
// for every element type (including integral T, where the denotation is
// already meaningless — the code still compiles).
template <typename W>
__device__ __forceinline__ W e_exp(W x) {{ return (W)exp((double)x); }}
template <typename W>
__device__ __forceinline__ W e_tanh(W x) {{ return (W)tanh((double)x); }}
template <typename W>
__device__ __forceinline__ W e_erf(W x) {{ return (W)erf((double)x); }}

// Store cast W -> T. Generic path is a C cast; half/bf16 get the
// explicit intrinsics so the rounding is the named one.
template <typename T, typename W>
__device__ __forceinline__ T store_cast(W v) {{ return (T)v; }}
template <>
__device__ __forceinline__ __half store_cast<__half, float>(float v) {{
  return __float2half(v);
}}
template <>
__device__ __forceinline__ __nv_bfloat16 store_cast<__nv_bfloat16, float>(
    float v) {{
  return __float2bfloat16(v);
}}

}}  // namespace cuda_emit_detail

namespace cuda_emit_host {{

// Broadcast the operand shapes into the canon shape (unpadded, up to 4
// dims). Returns 0 on success, -1 on rank > 4 or incompatible dims.
// shapes[i] has ranks[i] entries; canon gets canon_rank entries.
inline int elem_broadcast(
    const long long* const* shapes, const int* ranks, int n,
    long long* canon, int* canon_rank)
{{
  int maxrank = 0;
  for (int i = 0; i < n; ++i)
    if (ranks[i] > maxrank) maxrank = ranks[i];
  if (maxrank > 4) return -1;
  for (int j = 0; j < maxrank; ++j) {{
    long long d = 1;
    for (int i = 0; i < n; ++i) {{
      int od = j - (maxrank - ranks[i]);
      if (od < 0) continue;
      long long s = shapes[i][od];
      if (s == d || s == 1) continue;
      if (d == 1) {{ d = s; continue; }}
      return -1;  // neither dim is 1 and they differ
    }}
    canon[j] = d;
  }}
  *canon_rank = maxrank;
  return 0;
}}

// Left-pad `canon` (canon_rank dims) into the 4-dim kernel layout.
inline void elem_pad4(const long long* canon, int canon_rank,
                      long long* out4)
{{
  for (int d = 0; d < 4; ++d) {{
    int od = d - (4 - canon_rank);
    out4[d] = (od < 0) ? 1 : canon[od];
  }}
}}

// Effective per-dim element strides of one operand against the 4 padded
// canon dims: a dim absent from the operand (left-pad) or of size 1
// broadcasts, so it contributes stride 0. Returns 0 on success, -1 when
// rank is outside 0..=4 (the kernel's canon is 4-dim).
inline int elem_effective_strides(
    const long long* shape, int rank, const long long* stride,
    long long* out4)
{{
  if (rank < 0 || rank > 4) {{
    for (int d = 0; d < 4; ++d) out4[d] = 0;
    return -1;
  }}
  for (int d = 0; d < 4; ++d) {{
    int od = d - (4 - rank);
    if (od < 0 || shape[od] == 1) out4[d] = 0;
    else out4[d] = stride[od];
  }}
  return 0;
}}

// Replay a `transpose` view: swap axes d0/d1 of a shape/stride pair.
// Returns 0 on success, -1 when rank/d0/d1 are out of range.
inline int elem_transpose(
    long long* shape, long long* stride, int rank, int d0, int d1)
{{
  if (rank < 1 || d0 < 0 || d1 < 0 || d0 >= rank || d1 >= rank)
    return -1;
  long long ts = shape[d0]; shape[d0] = shape[d1]; shape[d1] = ts;
  long long tt = stride[d0]; stride[d0] = stride[d1]; stride[d1] = tt;
  return 0;
}}

// Replay a `permute` view (torch semantics: out dim i takes in dim
// dims[i]) on a shape/stride pair. dims has `rank` entries; rank must
// be <= 8 (the kernel broadcast caps effective rank at 4, so anything
// larger declines in elem_broadcast before this runs). Returns 0 on
// success, -1 on a bad rank or an out-of-range dim — never reads
// shape/stride out of bounds.
inline int elem_permute(
    long long* shape, long long* stride, int rank, const int* dims)
{{
  if (rank < 0 || rank > 8) return -1;
  for (int i = 0; i < rank; ++i)
    if (dims[i] < 0 || dims[i] >= rank) return -1;
  long long s[8], t[8];
  for (int i = 0; i < rank; ++i) {{ s[i] = shape[dims[i]]; t[i] = stride[dims[i]]; }}
  for (int i = 0; i < rank; ++i) {{ shape[i] = s[i]; stride[i] = t[i]; }}
  return 0;
}}

// Pack the launcher's geom array:
//   geom[0] = numel, geom[1..4] = canon4,
//   geom[5 + 4*i .. +3] = operand i's effective strides,
//   geom[5 + 4*n .. +3] = the output's effective strides.
inline void elem_pack_geom(
    long long* geom, long long numel, const long long* canon4,
    const long long* const* in_strides4, int n,
    const long long* out_strides4)
{{
  geom[0] = numel;
  for (int d = 0; d < 4; ++d) geom[1 + d] = canon4[d];
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < 4; ++d) geom[5 + 4 * i + d] = in_strides4[i][d];
  for (int d = 0; d < 4; ++d) geom[5 + 4 * n + d] = out_strides4[d];
}}

}}  // namespace cuda_emit_host

// Element type selector for launch_{slug}. The term carries no dtype;
// the caller picks the instantiation its tensors actually are.
enum cuda_emit_dtype : int {{
  CUDA_EMIT_F32 = 0,
  CUDA_EMIT_F64 = 1,
  CUDA_EMIT_F16 = 2,
  CUDA_EMIT_BF16 = 3,
  CUDA_EMIT_I32 = 4,
  CUDA_EMIT_I64 = 5,
}};

template <typename T>
__global__ void elem_kernel_{slug}({kparams})
{{
{kbody}
}}

// Read operands (first-appearance order) and their layout views —
// views are replayed host-side on shape/stride pairs via
// cuda_emit_host::elem_transpose / elem_permute before the strides are
// computed; the kernel only ever sees flat stride-described loads.
{views_doc}

extern "C" cudaError_t launch_{slug}(
    int dtype,
    const void* const* in_ptrs,   // device pointers, one per read operand
    const void* out_ptr,          // device pointer to the store target
    const long long* geom,        // packed by cuda_emit_host::elem_pack_geom
    unsigned int block,           // threads per block; 0 -> 256
    cudaStream_t stream)
{{
  const long long nout = geom[0];
  if (block != 0u && block > 1024u) return cudaErrorInvalidValue;
  if (nout <= 0) return cudaSuccess;
  const unsigned int tb = block ? block : 256u;
  unsigned long long want = ((unsigned long long)nout + tb - 1) / tb;
  const unsigned int grid =
      want > 0x7fffffffull ? 0x7fffffffu : (unsigned int)want;
  const long long sh0 = geom[1], sh1 = geom[2], sh2 = geom[3], sh3 = geom[4];
  // nout > 0 implies every canon dim >= 1; a packed 0/negative dim is a
  // caller geom bug — refuse rather than divide by zero in the kernel.
  if (sh0 < 1 || sh1 < 1 || sh2 < 1 || sh3 < 1)
    return cudaErrorInvalidValue;
  switch (dtype) {{
{cases}    default:
      return cudaErrorInvalidValue;
  }}
}}
"#,
        slug = slug,
        kparams = kparams.join(", "),
        kbody = kbody,
        views_doc = views_doc,
        cases = cases,
    ))
}

/// Triton-style expression over `W` work values. `p{i}` loads use the
/// coordinate/stride locals the kernel body binds.
fn gen_expr(e: &Expr) -> Result<String, String> {
    Ok(match e {
        Expr::Load(i) => format!("((W)(p{i}[c0*s{i}_0 + c1*s{i}_1 + c2*s{i}_2 + c3*s{i}_3]))"),
        Expr::Lit(Lit::Int(v)) => format!("(W){v}"),
        Expr::Lit(Lit::Float(v)) => format!("(W){v:e}"),
        Expr::Unary(op, a) => {
            let ga = gen_expr(a)?;
            match *op {
                "neg" => format!("(-({ga}))"),
                "relu" => format!("(({ga}) > (W)0 ? ({ga}) : (W)0)"),
                "exp" => format!("cuda_emit_detail::e_exp({ga})"),
                "sigmoid" => {
                    format!("((W)1 / ((W)1 + cuda_emit_detail::e_exp(-({ga}))))")
                }
                "tanh" => format!("cuda_emit_detail::e_tanh({ga})"),
                "gelu" => format!(
                    "((W)0.5 * ({ga}) * ((W)1 + cuda_emit_detail::e_erf(\
                     ({ga}) * (W)0.70710678118654752440)))"
                ),
                // silu = x * sigmoid(x)
                "silu" => {
                    format!("({ga}) * ((W)1 / ((W)1 + cuda_emit_detail::e_exp(-({ga}))))")
                }
                _ => return Err(format!("unary op {op:?} has no CUDA lowering")),
            }
        }
        Expr::Binary(op, a, b) => {
            let (ga, gb) = (gen_expr(a)?, gen_expr(b)?);
            match *op {
                "mul" => format!("(({ga}) * ({gb}))"),
                "sub" => format!("(({ga}) - ({gb}))"),
                "bias" => format!("(({ga}) + ({gb}))"),
                _ => return Err(format!("binary op {op:?} has no CUDA lowering")),
            }
        }
    })
}

/// A stable, filesystem/identifier-safe name for the kernel and its
/// launcher: the op sequence plus a hash of the whole lowered term, so
/// two different terms never share a `launch_*` symbol.
fn kernel_slug(e: &Elementwise) -> String {
    let mut ops: Vec<&'static str> = Vec::new();
    collect_ops(&e.expr, &mut ops);
    let mut names: Vec<String> = e.operands.iter().map(|o| o.name.clone()).collect();
    names.sort();
    let mut raw = ops.join("_");
    for n in &names {
        raw.push('_');
        raw.push_str(n);
    }
    let clean: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    // cap the cosmetic prefix — the fnv hash keeps the symbol unique
    let clean: String = clean.chars().take(48).collect();
    let clean = clean.trim_matches('_');
    let clean = if clean.is_empty() || clean.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        format!("k_{clean}")
    } else {
        clean.to_string()
    };
    format!("{}_{}", clean, fnv1a(&term_key(e)))
}

fn collect_ops(e: &Expr, out: &mut Vec<&'static str>) {
    match e {
        Expr::Unary(op, a) => {
            out.push(op);
            collect_ops(a, out);
        }
        Expr::Binary(op, a, b) => {
            out.push(op);
            collect_ops(a, out);
            collect_ops(b, out);
        }
        Expr::Load(_) | Expr::Lit(_) => {}
    }
}

/// Canonical serialization of the lowered term — the hash input.
fn term_key(e: &Elementwise) -> String {
    let mut s = String::new();
    for o in &e.operands {
        s.push_str(&format!("{}|{:?};", o.name, o.views));
    }
    s.push('|');
    expr_key(&e.expr, &mut s);
    s
}

fn expr_key(e: &Expr, s: &mut String) {
    match e {
        Expr::Load(i) => s.push_str(&format!("L{i}")),
        Expr::Lit(Lit::Int(v)) => s.push_str(&format!("i{v}")),
        Expr::Lit(Lit::Float(v)) => s.push_str(&format!("f{v:e}")),
        Expr::Unary(op, a) => {
            s.push_str(op);
            s.push('(');
            expr_key(a, s);
            s.push(')');
        }
        Expr::Binary(op, a, b) => {
            s.push_str(op);
            s.push('(');
            expr_key(a, s);
            s.push(',');
            expr_key(b, s);
            s.push(')');
        }
    }
}

pub(crate) fn fnv1a(s: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:08x}")
}

/// Per-operand view documentation emitted above the launcher.
fn operand_views_doc(operands: &[Operand]) -> String {
    if operands.iter().all(|o| o.views.is_empty()) {
        return String::new();
    }
    let mut lines = String::from(
        "// Layout views per read operand (replayed innermost-first on the\n\
         // caller's shape/stride pair before elem_effective_strides):\n",
    );
    for (i, o) in operands.iter().enumerate() {
        if o.views.is_empty() {
            continue;
        }
        let vs: Vec<String> = o.views.iter().map(view_doc).collect();
        lines.push_str(&format!(
            "//   operand {i} \"{}\": {}\n",
            o.name,
            vs.join(", ")
        ));
    }
    lines
}

fn view_doc(v: &View) -> String {
    match v {
        View::Transpose(d0, d1) => format!("transpose({d0},{d1})"),
        View::Permute(dims) => {
            let ds: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
            format!("permute({})", ds.join(","))
        }
    }
}
