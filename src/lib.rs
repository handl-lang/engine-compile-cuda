//! Lower a kernel IR `Term` to CUDA C++ source (ADR-0171 fragment).
//!
//! ```text
//! term (JSON from lib-kernel-ir)
//!   -> as_elementwise -> emit   (one flat CUDA kernel over the
//!                                broadcast output)
//! ```
//!
//! Scope is deliberately the elementwise/layout fragment only: the
//! pointwise ops (`mul`, `sub`, `neg`, `relu`, `exp`, `sigmoid`, `tanh`,
//! `gelu`, `silu`, `bias`) over leaf operands inside `store(body, out)`.
//! The qgemv ops (`tile_load` as a real load, `unpack_i4`, `dequant`,
//! `dot`, `split`, `stage`, `reduce`) have no CUDA lowering in this
//! crate and are refused **by name** — a term that does not fit is an
//! error, never a stub.

pub mod codegen;

pub use codegen::emit;
pub use compile_triton::as_elementwise;
pub use compile_triton::term::Node;

/// Compile a parsed term to CUDA C++ source, or say precisely why not.
pub fn compile(node: &Node) -> Result<String, String> {
    let e = as_elementwise(node)?;
    Ok(emit(&e))
}
