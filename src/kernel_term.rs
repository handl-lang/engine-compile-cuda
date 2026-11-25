//! `Emitter<KernelTerm>` for the CUDA backend (PL-004 T-094).
//!
//! The IR is [`tool_compile_triton::KernelTerm`] — the `lib-kernel-ir` wire
//! term, defined next to [`tool_compile_triton::Node`] because that is where
//! the wire format is parsed. `impl Emitter<KernelTerm>` can live here
//! (the orphan rule only needs a local *self* type): [`CudaEmitter`]
//! is local, the trait and the IR are foreign. This mirrors how
//! `tool-compile-cuda` already reuses `tool_compile_triton::term::Node` for its
//! `compile()` entry point.
//!
//! `Emitter::verify` stays structural, same ceiling as the Triton
//! emitter and kernelopt's `verify_lowered`: there is no CUDA runner
//! in this repo, so artifact-level verify remains producer-side
//! (term-denotation) until one exists.

use hnl_tool_codegen::{Change, CodegenError, EmitContext, Emitter};
use tool_compile_triton::KernelTerm;
use tool_compile_triton::kernel_term::artifact_name;

use crate::compile;

/// `Emitter<KernelTerm>` for CUDA C++ (`.cu`) output.
///
/// Stateless — `compile()` picks between the elementwise/layout
/// fragment and the contract fragment; both refusals are named in the
/// `Emit` error, verbatim, exactly like the `tool-compile-cuda` CLI.
#[derive(Debug, Clone, Default)]
pub struct CudaEmitter;

impl Emitter<KernelTerm> for CudaEmitter {
    fn materialize(
        &self,
        ir: &KernelTerm,
        _req: &EmitContext,
    ) -> Result<Vec<Change>, CodegenError> {
        let source = compile(ir.node()).map_err(CodegenError::Emit)?;
        Ok(vec![Change::text(
            artifact_name("cuda", ir.raw(), "cu"),
            source,
        )])
    }

    /// Structural verify: exactly one `Put` on the artifact this
    /// emitter would have named.
    fn verify(&self, ir: &KernelTerm, changes: &[Change]) -> Result<(), CodegenError> {
        let want = artifact_name("cuda", ir.raw(), "cu");
        match changes {
            [change] if change.artifact() == &want && matches!(change, Change::Put { .. }) => {
                Ok(())
            }
            _ => Err(CodegenError::Verify(format!(
                "cuda emitter produced {} change(s), expected one Put on {want}",
                changes.len()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tool_compile_triton::TritonEmitter;

    const ELEMENTWISE: &str = r#"{"op":"store","args":[{"op":"neg","args":[{"leaf":{"name":"x"}}]},
            {"leaf":{"name":"out"}}]}"#;

    #[test]
    fn materialize_elementwise_put_shape() {
        let t = KernelTerm::parse(ELEMENTWISE).unwrap();
        let changes = CudaEmitter
            .materialize(&t, &EmitContext::new())
            .expect("elementwise fixture lowers");
        assert_eq!(changes.len(), 1);
        let Change::Put {
            artifact, content, ..
        } = &changes[0]
        else {
            panic!("expected a Put");
        };
        assert!(artifact.as_str().starts_with("kernel_cuda_"));
        assert!(artifact.as_str().ends_with(".cu"));
        let text = String::from_utf8(content.clone()).unwrap();
        assert!(text.contains("cuda"));
        CudaEmitter.verify(&t, &changes).unwrap();
    }

    #[test]
    fn triton_and_cuda_slugs_share_the_term_hash() {
        // Same term bytes → same 12-hex digest, different target/extension.
        let t = KernelTerm::parse(ELEMENTWISE).unwrap();
        let triton = TritonEmitter::default()
            .materialize(&t, &EmitContext::new())
            .unwrap();
        let cuda = CudaEmitter.materialize(&t, &EmitContext::new()).unwrap();
        let ta = triton[0].artifact().as_str();
        let ca = cuda[0].artifact().as_str();
        assert!(ta.starts_with("kernel_triton_"));
        assert!(ca.starts_with("kernel_cuda_"));
        assert_eq!(
            &ta["kernel_triton_".len()..ta.len() - 3],
            &ca["kernel_cuda_".len()..ca.len() - 3]
        );
    }

    #[test]
    fn materialize_refusal_is_named() {
        // `dot` of a computed lhs: refused by the elementwise fragment
        // (dot is contract-only) and by the contract fragment (dot
        // operands must be leaves/views) — both reasons surface.
        let bad = r#"{"op":"store","args":[{"op":"dot","args":[
            {"op":"neg","args":[{"leaf":{"name":"a"}}]},{"leaf":{"name":"b"}}]},
            {"leaf":{"name":"out"}}]}"#;
        let t = KernelTerm::parse(bad).unwrap();
        let err = CudaEmitter
            .materialize(&t, &EmitContext::new())
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("contract path"), "got: {msg}");
    }
}
