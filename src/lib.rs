//! Lower a kernel IR `Term` to CUDA C++ source (ADR-0171 fragment).
//!
//! ```text
//! term (JSON from lib-kernel-ir)
//!   -> as_elementwise -> emit          (the pointwise/layout fragment:
//!                                       one flat kernel over the
//!                                       broadcast output)
//!   -> as_contract   -> emit_contract  (the qgemv fragment: dot /
//!                                       split-k / reduce-of-partials,
//!                                       fused into one matvec kernel)
//! ```
//!
//! Scope is deliberately two fragments. The elementwise/layout one
//! covers the pointwise ops (`mul`, `sub`, `neg`, `relu`, `exp`,
//! `sigmoid`, `tanh`, `gelu`, `silu`, `bias`) over leaf operands inside
//! `store(body, out)`, with `transpose`/`permute`/`tile_load`/
//! `pipeline` as views and identities. The contract one covers
//! `store(body, out)` where `body` is `dot` (plain `[k,n]` tile or the
//! `dequant(unpack_i4(...))` AWQ spelling), `split` of a dot,
//! `reduce` of dot partials, and `bias` wrappers — see
//! `contract`'s module doc for the denotation-pinned semantics.
//!
//! What still has no CUDA lowering and is refused **by name**: a
//! computed (non-leaf) operand inside `dot`, `reduce` partials that
//! are not dots, `split` of a non-dot body, `dequant`/`unpack_i4`
//! outside the dot lhs (they denote materialised tiles this crate
//! doesn't stage), and `stage` outside the contract body. A term that
//! fits neither fragment is an error, never a stub.

pub mod codegen;
pub mod contract;

pub use codegen::emit;
pub use compile_triton::as_elementwise;
pub use compile_triton::term::Node;
pub use contract::{as_contract, emit_contract, Contract, Dot};

/// Compile a parsed term to CUDA C++ source, or say precisely why not.
///
/// The elementwise/layout fragment is tried first (it is the broader
/// surface); a `store` term it refuses falls through to the contract
/// fragment. When both refuse, the error names both paths' reasons.
pub fn compile(node: &Node) -> Result<String, String> {
    match as_elementwise(node) {
        Ok(e) => Ok(emit(&e)),
        Err(elem_err) => {
            if node.op() != Some("store") {
                return Err(elem_err);
            }
            match as_contract(node) {
                Ok(c) => Ok(emit_contract(&c)),
                Err(contract_err) => Err(format!("{elem_err}; contract path: {contract_err}")),
            }
        }
    }
}
