//! Contract term -> CUDA C++ source: the `dot`/`split`/`reduce` fragment.
//!
//! This is the second lowering path beside `codegen`'s elementwise one.
//! It recognises `store(body, out)` where `body` is a *contraction*
//! tree — the qgemv vocabulary of ADR-0165/0166:
//!
//! ```text
//! store( dot(A[k,n], x[k]), out )                          -- matvec
//! store( split(k, parts, [pipeline] dot(A, x)), out )      -- split-k
//! store( reduce(p1, ..., pm), out )                        -- partial sum
//! store( bias(<any of the above>, b[n]), out )             -- + bias
//! ```
//!
//! where each dot's left operand is either
//!
//! * a leaf (or `transpose`/`permute`/`tile_load` view chain over a
//!   leaf) denoting the `[k, n]` tile directly, or
//! * the AWQ qgemv spelling
//!   `dequant(unpack_i4(w[k, n/8], shifts[8]), z[g, n], s[g, n], group)`
//!   with `z`/`s` carried as `kind: "g"` leaves whose `div` is the
//!   group size — the kernel indexes `g = k / div` per element, exactly
//!   the `repeat_interleave` the IR denotation applies.
//!
//! ## What the denotations pin down (ops.py / term.py)
//!
//! `dot` casts both operands to fp32 before the einsum, `stage` casts
//! the partial to fp32, `reduce` sums the staged fp32 partials, and
//! `store` casts to `out.dtype`. So the emitted kernel *always*
//! accumulates in `float` and casts once on store — for every dtype
//! instantiation. `split` and `pipeline` are schedule annotations: the
//! fused kernel computes the whole contraction in one launch, which is
//! the same denotation (the partial sums of a partition are summed
//! anyway; reassociation is inside the ROUNDS grade). The `split`
//! literal `k` is checked against the caller-packed `geom` klen of the
//! (single) dot at launch — a mismatch is `cudaErrorInvalidValue`,
//! never a silent wrong answer.
//!
//! `stage` unwraps transparently anywhere inside the contract body
//! (it denotes an fp32 materialisation the fused kernel already
//! performs); `pipeline` unwraps likewise.
//!
//! ## Wire contract
//!
//! `geom` (all `long long`):
//! `[nout, <per-dot stride block>..., <per-bias stride>...]`
//! where each dot's block *leads* with its own `klen` — the dots in a
//! `reduce` may contract different k axes, so k is per-dot, not
//! global: a plain dot contributes `(klen, ak, an, kx)` — k-stride,
//! n-stride of the `[k,n]` tile, k-stride of `x`; a dequant dot
//! contributes `(klen, wk, wn, zg, zn, zdiv, sg, sn, sdiv, shs, kx)`;
//! each bias contributes `(bs)`. `in_ptrs` order is the kernel's
//! pointer parameter order: per dot `a, x` (plain) or `w, shifts, z,
//! s, x` (dequant), then each bias operand; `out_ptr` is separate.
//!
//! Strides are *effective* (post-view) element strides computed by the
//! caller with `cuda_emit_host::contract_strides2` / `_strides1`,
//! which the caller applies after replaying the term's
//! `transpose`/`permute` views on the shape/stride pair — the same
//! scheme the elementwise emitter uses. Packed `w`, `shifts`, and zero
//! `z` are read as `int` (int32 — `unpack_i4`'s signature pins the
//! packed input and `shifts()` to int32, and AWQ zero-points are
//! int32); `s`, `x`, plain tiles, biases and `out` share the launcher's
//! `dtype` `T`.

use compile_triton::term::{Node, Operand, View};

use crate::codegen::fnv1a;

/// One matvec partial: `dot(lhs, x)`.
#[derive(Debug, Clone, PartialEq)]
pub enum Dot {
    /// `dot(tile[k,n], x[k])` — the tile is a leaf/view-chain operand.
    Plain {
        /// the `[k, n]` weight tile operand (leaf + views).
        tile: Operand,
        /// the `[k]` activation operand (leaf + views).
        x: Operand,
    },
    /// `dot(dequant(unpack_i4(w, shifts), z, s, group), x)` — the AWQ
    /// qgemv tile. `z_div`/`s_div` are the leaves' `div` (k-elements
    /// per g-row).
    Dequant {
        /// packed `[k, n/8]` int32 weights operand.
        w: Operand,
        /// the 8 nibble-shift values operand (int32, rank-1).
        shifts: Operand,
        /// `[g, n]` zero-point operand, `kind:"g"` with `div` = z_div.
        z: Operand,
        /// `[g, n]` scale operand, `kind:"g"` with `div` = s_div.
        s: Operand,
        /// the `[k]` activation operand (leaf + views).
        x: Operand,
        /// k-elements per z row (`g = k / z_div` per element).
        z_div: i64,
        /// k-elements per s row (`g = k / s_div` per element).
        s_div: i64,
    },
}

/// A recognised contract term: `store(body, out)` lowered to one
/// kernel computing `sum_j dot_j[n] + sum_b bias_b[n]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Contract {
    /// the matvec partials summed into the `[n]` output, in term order.
    pub dots: Vec<Dot>,
    /// rank-1 `[n]` bias operands (outermost bias peeled first).
    pub biases: Vec<Operand>,
    /// the `split` k literal, when the term spelled one — checked
    /// against `geom`'s dot-0 klen at launch.
    pub k_literal: Option<i64>,
}

/// Read a term as a contract kernel, or say precisely why not.
pub fn as_contract(root: &Node) -> Result<Contract, String> {
    if root.op() != Some("store") {
        return Err(format!("expected a store term, got {:?}", root.op()));
    }
    let args = root.args().ok_or("store has no args")?;
    if args.len() != 2 {
        return Err("store wants (tile, out)".into());
    }
    if args[1].leaf_name().is_none() {
        return Err("store's second argument must be a leaf".into());
    }

    // Peel scheduling annotations (`pipeline`/`stage` — both denote
    // the identity here) and `bias` wrappers in any nesting order:
    // bias(<contract>, b) is a contract result plus a rank-1 operand.
    let mut body = &args[0];
    let mut biases = Vec::new();
    loop {
        body = unwrap_sched(body)?;
        if body.op() != Some("bias") {
            break;
        }
        let bargs = body.args().ok_or("bias has no args")?;
        if bargs.len() != 2 {
            return Err("bias wants (tile, bias)".into());
        }
        biases.push(operand_of(&bargs[1])?);
        body = &bargs[0];
    }

    let (dots, k_literal) = contract_core(body)?;
    if dots.is_empty() {
        return Err("a contract term needs at least one dot partial".into());
    }
    Ok(Contract {
        dots,
        biases,
        k_literal,
    })
}

/// The contraction core: `dot`, `reduce` of dots, or `split` of a dot.
fn contract_core(node: &Node) -> Result<(Vec<Dot>, Option<i64>), String> {
    match node.op() {
        Some("dot") => Ok((vec![dot_of(node)?], None)),
        Some("reduce") => {
            let args = node.args().ok_or("reduce has no args")?;
            if args.is_empty() {
                return Err("reduce wants at least one partial".into());
            }
            let mut dots = Vec::new();
            for (i, p) in args.iter().enumerate() {
                dots.push(partial_dot(p).map_err(|e| format!("reduce partial {i}: {e}"))?);
            }
            Ok((dots, None))
        }
        Some("split") => {
            let args = node.args().ok_or("split has no args")?;
            if args.len() != 3 {
                return Err("split wants (k_len, n_parts, body)".into());
            }
            let (Some(k), Some(parts)) = (args[0].as_int(), args[1].as_int()) else {
                return Err("split's k and parts must be integer literals".into());
            };
            if k <= 0 || parts <= 0 {
                return Err(format!("split geometry must be positive, got {k}/{parts}"));
            }
            // split is a k-partition annotation: the fused kernel runs
            // the whole range, so `parts` is schedule documentation and
            // `k` becomes a launch-time geom check.
            Ok((vec![dot_of(unwrap_sched(&args[2])?)?], Some(k)))
        }
        _ => Err(format!(
            "{:?} is not a contract core — expected dot, reduce of dots, \
             or split of a dot",
            node.op()
        )),
    }
}

/// Peel `pipeline`/`stage` wrappers — both denote the identity (a
/// schedule annotation and an fp32 materialisation the fused kernel
/// already performs).
fn unwrap_sched(mut node: &Node) -> Result<&Node, String> {
    loop {
        match node.op() {
            Some("pipeline") => {
                let p = node.args().ok_or("pipeline has no args")?;
                if p.len() != 2 {
                    return Err("pipeline wants (depth, body)".into());
                }
                node = &p[1];
            }
            Some("stage") => {
                let s = node.args().ok_or("stage has no args")?;
                if s.len() != 1 {
                    return Err("stage wants (partial)".into());
                }
                node = &s[0];
            }
            _ => return Ok(node),
        }
    }
}

/// One reduce partial: `stage`/`pipeline` wrappers around a `dot`.
fn partial_dot(node: &Node) -> Result<Dot, String> {
    let body = unwrap_sched(node)?;
    if body.op() != Some("dot") {
        return Err(format!(
            "partial must be a dot (optionally stage/pipeline-wrapped), \
             got {:?}",
            body.op()
        ));
    }
    dot_of(body)
}

/// `dot(lhs, x)` — lhs is the `[k,n]` tile (plain or dequant), x is `[k]`.
fn dot_of(node: &Node) -> Result<Dot, String> {
    if node.op() != Some("dot") {
        return Err(format!(
            "expected a dot term, got {:?} — split/reduce bodies must \
             bottom out on dot",
            node.op()
        ));
    }
    let args = node.args().ok_or("dot has no args")?;
    if args.len() != 2 {
        return Err("dot wants (tile, x)".into());
    }
    let x = operand_of(&args[1])?;

    // Unwrap scheduling pass-throughs and view-pass tile_loads on the
    // lhs, then classify.
    let mut cur = &args[0];
    loop {
        match cur.op() {
            Some("pipeline") | Some("stage") => cur = unwrap_sched(cur)?,
            Some("tile_load") => {
                let t = cur.args().ok_or("tile_load has no args")?;
                if t.len() != 1 {
                    return Err("tile_load wants 1 argument".into());
                }
                cur = &t[0];
            }
            _ => break,
        }
    }

    if cur.op() == Some("dequant") {
        return dequant_dot(cur, x);
    }
    // Plain tile: a leaf under a transpose/permute/tile_load chain.
    let (name, views, _kind, _div) = leaf_views(cur)?;
    Ok(Dot::Plain {
        tile: Operand { name, views },
        x,
    })
}

/// `dequant(unpack_i4(w, shifts), z, s, group)` as a dot lhs.
fn dequant_dot(node: &Node, x: Operand) -> Result<Dot, String> {
    let args = node.args().ok_or("dequant has no args")?;
    if args.len() != 4 {
        return Err("dequant wants (q, zero, scale, group)".into());
    }
    let group = args[3]
        .as_int()
        .ok_or("dequant's group must be an integer")?;
    if group <= 0 {
        return Err(format!("group must be positive, got {group}"));
    }
    // `group` itself is not emitted: the IR denotation passes it to
    // `_dequant` only for its error message — the actual indexing is
    // the `kind:"g"` leaves' `div` (`repeat_interleave`), which the
    // kernel reads per element as `g = k / div`.

    // q must be unpack_i4(w, shifts), modulo pass-through wrappers.
    let q = unwrap_sched(&args[0])?;
    let q = if q.op() == Some("tile_load") {
        let t = q.args().ok_or("tile_load has no args")?;
        if t.len() != 1 {
            return Err("tile_load wants 1 argument".into());
        }
        unwrap_sched(&t[0])?
    } else {
        q
    };
    if q.op() != Some("unpack_i4") {
        return Err(format!(
            "dequant's first argument must be an unpack_i4, got {:?}",
            q.op()
        ));
    }
    let uargs = q.args().ok_or("unpack_i4 has no args")?;
    if uargs.len() != 2 {
        return Err("unpack_i4 wants (packed, shifts)".into());
    }
    let (w_name, w_views, _wk, _wdiv) = leaf_views(&uargs[0])?;
    let (sh_name, sh_views, _sk, _sdiv0) = leaf_views(&uargs[1])?;

    // zero/scale are group-indexed leaves: the kernel computes
    // g = k / div per element (the denotation's repeat_interleave).
    let (z_name, z_views, z_kind, z_div) = leaf_views(&args[1])?;
    let (s_name, s_views, s_kind, s_div) = leaf_views(&args[2])?;
    if z_kind != "g" || s_kind != "g" {
        return Err(format!(
            "dequant's zero/scale leaves must be kind \"g\" \
             (group-indexed), got {z_kind:?}/{s_kind:?}"
        ));
    }
    if z_div < 1 || s_div < 1 {
        return Err(format!(
            "dequant group-indexed leaves need div >= 1, got {z_div}/{s_div}"
        ));
    }

    Ok(Dot::Dequant {
        w: Operand {
            name: w_name,
            views: w_views,
        },
        shifts: Operand {
            name: sh_name,
            views: sh_views,
        },
        z: Operand {
            name: z_name,
            views: z_views,
        },
        s: Operand {
            name: s_name,
            views: s_views,
        },
        x,
        z_div,
        s_div,
    })
}

/// A leaf under `transpose`/`permute`/`tile_load` wrappers — the same
/// view-chain rule the elementwise path uses (a chain bottoming out on
/// a computed tile is refused). Returns name, views (innermost-first),
/// leaf kind and div.
fn leaf_views(node: &Node) -> Result<(String, Vec<View>, String, i64), String> {
    let mut collected: Vec<View> = Vec::new();
    let mut cur = node;
    loop {
        match cur {
            Node::Leaf { leaf } => {
                collected.reverse();
                return Ok((leaf.name.clone(), collected, leaf.kind.clone(), leaf.div));
            }
            Node::Term { op, args } => match op.as_str() {
                "tile_load" if args.len() == 1 => cur = &args[0],
                "transpose" => {
                    if args.len() != 3 {
                        return Err("transpose wants (tensor, dim0, dim1)".into());
                    }
                    let (Some(d0), Some(d1)) = (args[1].as_int(), args[2].as_int()) else {
                        return Err("transpose dims must be integer literals".into());
                    };
                    collected.push(View::Transpose(d0, d1));
                    cur = &args[0];
                }
                "permute" => {
                    if args.len() < 2 {
                        return Err("permute wants (tensor, dims...)".into());
                    }
                    let mut dims = Vec::new();
                    for a in &args[1..] {
                        match a.as_int() {
                            Some(d) => dims.push(d),
                            None => return Err("permute dims must be integer literals".into()),
                        }
                    }
                    collected.push(View::Permute(dims));
                    cur = &args[0];
                }
                _ => {
                    return Err(format!(
                        "a view chain bottoms out on {op:?}, not a leaf — \
                         dot operands lower only over leaf tensors"
                    ));
                }
            },
            _ => {
                return Err("a dot operand must be a leaf (or view chain over one), \
                     not a literal"
                    .into());
            }
        }
    }
}

/// Bias operand: a leaf/view-chain — anything else is refused (a
/// computed bias would need a materialisation pass this crate
/// doesn't emit).
fn operand_of(node: &Node) -> Result<Operand, String> {
    let (name, views, _kind, _div) = leaf_views(node)?;
    Ok(Operand { name, views })
}

// -- emission ---------------------------------------------------------------

/// C type of one `in_ptrs` entry.
#[derive(Clone, Copy)]
enum PtrTy {
    /// `const T*` — float-side operands (s, x, plain tile, bias, out)
    T,
    /// `const int*` — packed w, shifts, zero z (int32-pinned)
    I,
}

struct PtrParam {
    name: String,
    ty: PtrTy,
    /// caller-facing role, e.g. `dot0.tile "w"` — used in the emitted
    /// in_ptrs-order doc.
    doc: String,
}

/// Emit a CUDA C++ translation unit for a contract term.
///
/// One templated `__global__` kernel: a grid-stride loop over the `n`
/// outputs, each thread accumulating `sum_j dot_j + sum_b bias` in
/// `float` (the denotations' compute type) and storing through
/// `store_cast<T>`. See the module doc for the `geom`/`in_ptrs`
/// contract.
///
/// Returns `Err` when the `Contract` is degenerate (no dot partials) —
/// `as_contract` cannot produce one, but the fields are public.
pub fn emit_contract(c: &Contract) -> Result<String, String> {
    if c.dots.is_empty() {
        return Err("a contract needs at least one dot partial".into());
    }
    let slug = contract_slug(c);

    // in_ptrs order == kernel pointer-param order == geom stride-block
    // order.
    let mut ptrs: Vec<PtrParam> = Vec::new();
    let mut geom_names: Vec<String> = Vec::new(); // stride-block entries
    // geom entries the kernel *divides by* or loops to — packed by the
    // caller, so the launcher validates them (geom index -> minimum).
    let mut geom_min: Vec<(usize, i64, &str)> = Vec::new();
    let mut body_parts: Vec<String> = Vec::new();

    for (j, d) in c.dots.iter().enumerate() {
        // geom index of this dot's block lead (klen{j}) — geom[0] is
        // nout, so block entry `geom_names.len()` lands at geom[1+len].
        let block0 = geom_names.len();
        match d {
            Dot::Plain { tile, x } => {
                for (param, ty, doc) in [
                    (
                        format!("a{j}"),
                        PtrTy::T,
                        format!("dot {j} tile \"{}\"", tile.name),
                    ),
                    (
                        format!("x{j}"),
                        PtrTy::T,
                        format!("dot {j} x \"{}\"", x.name),
                    ),
                ] {
                    ptrs.push(PtrParam {
                        name: param,
                        ty,
                        doc,
                    });
                }
                geom_names.extend([
                    format!("klen{j}"),
                    format!("ak{j}"),
                    format!("an{j}"),
                    format!("kx{j}"),
                ]);
                geom_min.push((1 + block0, 0, "klen"));
                body_parts.push(format!(
                    r#"  // dot {j}: plain tile[k,n] . x[k]
  for (long long k = 0; k < klen{j}; ++k) {{
    acc += cuda_emit_detail::to_f32(a{j}[k*ak{j} + i*an{j}]) *
           cuda_emit_detail::to_f32(x{j}[k*kx{j}]);
  }}"#
                ));
            }
            Dot::Dequant {
                w, shifts, z, s, x, ..
            } => {
                for (param, ty, doc) in [
                    (
                        format!("w{j}"),
                        PtrTy::I,
                        format!("dot {j} w \"{}\"", w.name),
                    ),
                    (
                        format!("sh{j}"),
                        PtrTy::I,
                        format!("dot {j} shifts \"{}\"", shifts.name),
                    ),
                    (
                        format!("z{j}"),
                        PtrTy::I,
                        format!("dot {j} z \"{}\"", z.name),
                    ),
                    (
                        format!("s{j}"),
                        PtrTy::T,
                        format!("dot {j} s \"{}\"", s.name),
                    ),
                    (
                        format!("x{j}"),
                        PtrTy::T,
                        format!("dot {j} x \"{}\"", x.name),
                    ),
                ] {
                    ptrs.push(PtrParam {
                        name: param,
                        ty,
                        doc,
                    });
                }
                geom_names.extend([
                    format!("klen{j}"),
                    format!("wk{j}"),
                    format!("wn{j}"),
                    format!("zg{j}"),
                    format!("zn{j}"),
                    format!("zdiv{j}"),
                    format!("sg{j}"),
                    format!("sn{j}"),
                    format!("sdiv{j}"),
                    format!("shs{j}"),
                    format!("kx{j}"),
                ]);
                // klen >= 0; the group divisors must be >= 1 — the
                // kernel computes k / zdiv and k / sdiv per element.
                geom_min.push((1 + block0, 0, "klen"));
                geom_min.push((1 + block0 + 5, 1, "zdiv"));
                geom_min.push((1 + block0 + 8, 1, "sdiv"));
                body_parts.push(format!(
                    r#"  // dot {j}: dequant(unpack_i4(w, shifts), z, s) . x[k]
  //   g = k / div per element — the denotation's repeat_interleave;
  //   nibble n%8 of packed word n/8, shifted by shifts[n%8].
  {{
    const int shv{j} = sh{j}[(i & 7) * shs{j}];
    const long long wc{j} = i >> 3;
    for (long long k = 0; k < klen{j}; ++k) {{
      const int word{j} = w{j}[k*wk{j} + wc{j}*wn{j}];
      const int q{j} = (word{j} >> (shv{j} & 31)) & 0xF;
      const long long gz{j} = k / zdiv{j};
      const long long gs{j} = k / sdiv{j};
      const float zv{j} = (float)z{j}[gz{j}*zg{j} + i*zn{j}];
      const float sv{j} = cuda_emit_detail::to_f32(s{j}[gs{j}*sg{j} + i*sn{j}]);
      const float xv{j} = cuda_emit_detail::to_f32(x{j}[k*kx{j}]);
      acc += (((float)q{j} - zv{j}) * sv{j}) * xv{j};
    }}
  }}"#
                ));
            }
        }
    }
    for (b, bias) in c.biases.iter().enumerate() {
        ptrs.push(PtrParam {
            name: format!("b{b}"),
            ty: PtrTy::T,
            doc: format!("bias {b} \"{}\"", bias.name),
        });
        geom_names.push(format!("bs{b}"));
        body_parts.push(format!(
            "  acc += cuda_emit_detail::to_f32(b{b}[i*bs{b}]);  // bias {b}"
        ));
    }

    // -- kernel parameters -------------------------------------------------
    // No `__restrict__`: `store(body, out)` may write a leaf the body
    // also reads (in-place store), and distinct leaf names may alias —
    // the IR carries no alias information, so the kernel must not
    // promise any.
    let mut kparams: Vec<String> = vec!["long long nout".to_string()];
    for p in &ptrs {
        let ty = match p.ty {
            PtrTy::T => "const T*",
            PtrTy::I => "const int*",
        };
        kparams.push(format!("{ty} {}", p.name));
    }
    for s in &geom_names {
        kparams.push(format!("long long {s}"));
    }
    kparams.push("T* out_p".to_string());

    let kbody = format!(
        r#"  // fp32 accumulation: dot/stage/reduce all denote fp32 compute
  const long long gstride = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       i < nout; i += gstride) {{
    float acc = 0.0f;
{body}
    out_p[i] = cuda_emit_detail::store_cast<T>(acc);
  }}"#,
        body = body_parts.join("\n"),
    );

    // -- launcher ----------------------------------------------------------
    // geom indices: 0=nout, then the per-dot stride blocks (each leads
    // with its klen) and the per-bias strides, in order.
    let mut cases = String::new();
    for (cname, ty) in [
        ("CUDA_CONTRACT_F32", "float"),
        ("CUDA_CONTRACT_F64", "double"),
        ("CUDA_CONTRACT_F16", "__half"),
        ("CUDA_CONTRACT_BF16", "__nv_bfloat16"),
        ("CUDA_CONTRACT_I32", "int"),
        ("CUDA_CONTRACT_I64", "long long"),
    ] {
        let mut args: Vec<String> = vec!["nout".to_string()];
        for (pi, p) in ptrs.iter().enumerate() {
            let cast = match p.ty {
                PtrTy::T => format!("(const {ty}*)in_ptrs[{pi}]"),
                PtrTy::I => format!("(const int*)in_ptrs[{pi}]"),
            };
            args.push(cast);
        }
        for (gi, s) in geom_names.iter().enumerate() {
            args.push(format!("geom[{}] /* {s} */", 1 + gi));
        }
        args.push(format!("({ty}*)out_ptr"));
        let arglist = args.join(",\n          ");
        cases.push_str(&format!(
            r#"    case {cname}:
      contract_kernel_{slug}<{ty}><<<grid, tb, 0, stream>>>(
          {arglist});
      break;
"#
        ));
    }

    let views_doc = contract_views_doc(c);
    let ptr_doc = ptr_order_doc(&ptrs);
    let kcheck = c
        .k_literal
        .map(|k| format!("  if (geom[1] != {k}) return cudaErrorInvalidValue;\n"))
        .unwrap_or_default();
    // caller-packed geom the kernel divides by or loops to — a 0/negative
    // entry is a caller bug, refused at launch rather than a device fault.
    let geom_checks: String = geom_min
        .iter()
        .map(|(idx, min, name)| {
            format!("  if (geom[{idx}] < {min}) return cudaErrorInvalidValue; // {name}\n")
        })
        .collect();

    Ok(format!(
        r#"// Generated by compile-cuda from a kernel IR contract term
// (dot / split / reduce — the ADR-0165/0166 qgemv fragment).
// One fused kernel: grid-stride over the n outputs, float accumulation
// (the denotations' compute type — dot/stage/reduce are all fp32),
// store_cast<T> on write. `split`'s k literal is checked against
// geom's dot-0 klen at launch. Do not edit: the op tree came from the term.
//
// Contract for the caller:
//   * in_ptrs holds the read operands in this order (device pointers):
{ptr_doc}//   * geom is a long long[] packed by the caller:
//     [0]=nout, then per-dot stride blocks — each *leads* with its own
//     klen (dots in a reduce may contract different k axes):
//     plain dot: klen,ak,an,kx — the k length, then k-stride and
//      n-stride of the [k,n] tile and k-stride of x; dequant dot:
//      klen,wk,wn,zg,zn,zdiv,sg,sn,sdiv,shs,kx — then per-bias stride
//      (bs). Strides are post-view element
//     strides — replay the term's transpose/permute views on the
//     shape/stride pair first (cuda_emit_host::elem_transpose /
//     elem_permute), then contract_strides2 for rank-2 operands
//     (tile/w/z/s) and contract_strides1 for rank-1 operands
//     (biases, shifts). x's k-stride kx comes from the same rank-1
//     helper on the [k] activation.
//   * `block` is threads per block (0 -> 256, must be <= 1024);
//     `stream` may be NULL.
//   * dtype selects T for the float-side operands and out; packed w,
//     shifts and zero z are always read as int (int32 — the op
//     signatures pin them).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

namespace cuda_emit_detail {{

// T -> float for the fp32 compute the denotations specify. Generic
// path is a C cast; half/bf16 get the named intrinsics.
template <typename T>
__device__ __forceinline__ float to_f32(T v) {{ return (float)v; }}
template <>
__device__ __forceinline__ float to_f32<__half>(__half v) {{
  return __half2float(v);
}}
template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {{
  return __bfloat162float(v);
}}

// float -> T on store; half/bf16 get the explicit rounding intrinsics.
template <typename T>
__device__ __forceinline__ T store_cast(float v) {{ return (T)v; }}
template <>
__device__ __forceinline__ __half store_cast<__half>(float v) {{
  return __float2half(v);
}}
template <>
__device__ __forceinline__ __nv_bfloat16 store_cast<__nv_bfloat16>(
    float v) {{
  return __float2bfloat16(v);
}}

}}  // namespace cuda_emit_detail

namespace cuda_emit_host {{

// Replay a `transpose` view on a shape/stride pair. Returns 0 on
// success, -1 when rank/d0/d1 are out of range.
inline int elem_transpose(
    long long* shape, long long* stride, int rank, int d0, int d1)
{{
  if (rank < 1 || d0 < 0 || d1 < 0 || d0 >= rank || d1 >= rank)
    return -1;
  long long ts = shape[d0]; shape[d0] = shape[d1]; shape[d1] = ts;
  long long tt = stride[d0]; stride[d0] = stride[d1]; stride[d1] = tt;
  return 0;
}}

// Replay a `permute` view (torch semantics) on a shape/stride pair.
// dims has `rank` entries; rank must be <= 8 and every dim in range —
// returns 0 on success, -1 otherwise (never reads out of bounds).
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

// Effective (stride[0], stride[1]) of a rank-2 operand — the [k, n]
// (or [k, n/8], [g, n]) tile layouts. Size-1 dims broadcast to stride
// 0, matching the elementwise convention. Returns 0 on success, -1
// when rank != 2.
inline int contract_strides2(
    const long long* shape, const long long* stride, int rank,
    long long* out2)
{{
  if (rank != 2) return -1;
  for (int d = 0; d < 2; ++d) out2[d] = (shape[d] == 1) ? 0 : stride[d];
  return 0;
}}

// Effective stride of a rank-1 operand (x, biases, shifts). Returns
// 0 on success, -1 when rank != 1.
inline int contract_strides1(
    const long long* shape, const long long* stride, int rank,
    long long* out1)
{{
  if (rank != 1) return -1;
  out1[0] = (shape[0] == 1) ? 0 : stride[0];
  return 0;
}}

}}  // namespace cuda_emit_host

// dtype selector for launch_{slug}: T for the float-side operands and
// out. The int-side operands (packed w, shifts, zero z) are always
// int — the op signatures pin them to int32.
enum cuda_contract_dtype : int {{
  CUDA_CONTRACT_F32 = 0,
  CUDA_CONTRACT_F64 = 1,
  CUDA_CONTRACT_F16 = 2,
  CUDA_CONTRACT_BF16 = 3,
  CUDA_CONTRACT_I32 = 4,
  CUDA_CONTRACT_I64 = 5,
}};

template <typename T>
__global__ void contract_kernel_{slug}({kparams})
{{
{kbody}
}}

{views_doc}

extern "C" cudaError_t launch_contract_{slug}(
    int dtype,
    const void* const* in_ptrs,   // device pointers, order documented above
    const void* out_ptr,          // device pointer to the [n] output
    const long long* geom,        // packed as documented above
    unsigned int block,           // threads per block; 0 -> 256
    cudaStream_t stream)
{{
  const long long nout = geom[0];
  if (block != 0u && block > 1024u) return cudaErrorInvalidValue;
{kcheck}{geom_checks}  if (nout <= 0) return cudaSuccess;
  const unsigned int tb = block ? block : 256u;
  unsigned long long want = ((unsigned long long)nout + tb - 1) / tb;
  const unsigned int grid =
      want > 0x7fffffffull ? 0x7fffffffu : (unsigned int)want;
  switch (dtype) {{
{cases}    default:
      return cudaErrorInvalidValue;
  }}
}}
"#,
        slug = slug,
        kparams = kparams.join(", "),
        kbody = kbody,
        ptr_doc = ptr_doc,
        kcheck = kcheck,
        cases = cases,
        views_doc = views_doc,
        geom_checks = geom_checks,
    ))
}

/// Stable kernel/launcher name: op summary + hash of the contract.
fn contract_slug(c: &Contract) -> String {
    let mut raw = String::new();
    for d in &c.dots {
        raw.push_str(match d {
            Dot::Plain { .. } => "dot_plain_",
            Dot::Dequant { .. } => "dot_dequant_",
        });
    }
    for b in &c.biases {
        raw.push_str(&format!("b{}", b.name));
    }
    raw.push_str(&fnv1a(&contract_key(c)));
    let clean: String = raw
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    // cap the cosmetic prefix — the fnv hash keeps the symbol unique
    let clean: String = clean.chars().take(48).collect();
    clean.trim_matches('_').to_string()
}

fn contract_key(c: &Contract) -> String {
    let mut s = String::new();
    for d in &c.dots {
        match d {
            Dot::Plain { tile, x } => {
                s.push_str(&operand_key(tile));
                s.push('|');
                s.push_str(&operand_key(x));
            }
            Dot::Dequant {
                w,
                shifts,
                z,
                s: sc,
                x,
                z_div,
                s_div,
            } => {
                for o in [w, shifts, z, sc, x] {
                    s.push_str(&operand_key(o));
                    s.push('|');
                }
                s.push_str(&format!("{z_div}/{s_div}"));
            }
        }
        s.push(';');
    }
    for b in &c.biases {
        s.push_str(&operand_key(b));
        s.push(';');
    }
    if let Some(k) = c.k_literal {
        s.push_str(&format!("k{k}"));
    }
    s
}

fn operand_key(o: &Operand) -> String {
    let mut s = o.name.clone();
    for v in &o.views {
        match v {
            View::Transpose(a, b) => s.push_str(&format!("t{a},{b}")),
            View::Permute(ds) => {
                s.push('p');
                for d in ds {
                    s.push_str(&format!("{d}."));
                }
            }
        }
    }
    s
}

/// Document the in_ptrs order for the emitted caller contract.
fn ptr_order_doc(ptrs: &[PtrParam]) -> String {
    if ptrs.is_empty() {
        return String::new();
    }
    let mut s = String::from("//     ");
    for (i, p) in ptrs.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!("in_ptrs[{i}] = {} ({})", p.doc, p.name));
    }
    s.push('\n');
    s
}

/// Per-operand view documentation (same scheme as the elementwise
/// emitter): views are replayed host-side before strides are packed.
fn contract_views_doc(c: &Contract) -> String {
    let mut lines = String::new();
    let note = |role: &str, o: &Operand, lines: &mut String| {
        if o.views.is_empty() {
            return;
        }
        let vs: Vec<String> = o.views.iter().map(view_doc).collect();
        lines.push_str(&format!("//   {role} \"{}\": {}\n", o.name, vs.join(", ")));
    };
    for (j, d) in c.dots.iter().enumerate() {
        match d {
            Dot::Plain { tile, x } => {
                note(&format!("dot {j} tile"), tile, &mut lines);
                note(&format!("dot {j} x"), x, &mut lines);
            }
            Dot::Dequant {
                w, shifts, z, s, x, ..
            } => {
                note(&format!("dot {j} w"), w, &mut lines);
                note(&format!("dot {j} shifts"), shifts, &mut lines);
                note(&format!("dot {j} z"), z, &mut lines);
                note(&format!("dot {j} s"), s, &mut lines);
                note(&format!("dot {j} x"), x, &mut lines);
            }
        }
    }
    for (b, bias) in c.biases.iter().enumerate() {
        note(&format!("bias {b}"), bias, &mut lines);
    }
    if lines.is_empty() {
        String::new()
    } else {
        format!(
            "// Layout views per operand (replayed innermost-first on the\n\
             // caller's shape/stride pair before contract_strides*):\n{lines}"
        )
    }
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
