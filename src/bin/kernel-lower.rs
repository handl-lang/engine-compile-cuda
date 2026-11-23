//! `kernel-lower` — term JSON on stdin, a change-set manifest record on
//! stdout (PL-004 T-094).
//!
//! The Rust-side twin of kernelopt's Python `mlopt_kernel.lower`
//! (`lower_candidate`): same `Emitter<KernelTerm>` materialization,
//! same artifact slugs (`kernel_<target>_<blake3(term-json)[:12]>`),
//! same manifest wire shape — so a Rust consumer gets lowering without
//! a Python subprocess, and the emitted `manifest.json` is consumable
//! by the same apply shim mlopt uses (`kit_mlopt::manifest::
//! apply_manifest` — `hash_algorithm: "blake3-256"`, `blobs/<hash>`
//! payloads, `{"op","artifact","before_hash","blob_ref"|"content_b64",
//! "hash"}` changes).
//!
//! ```text
//! kernel-lower --target triton|cuda [--block-n N] [--block-k N]
//!              [--run-id ID] [--manifest-dir DIR] < term.json
//! ```
//!
//! stdout always carries the inline manifest *record* (changes embed
//! `content_b64` — self-contained JSON). `--manifest-dir DIR`
//! additionally writes the apply-shim consumable form: `DIR/blobs/
//! <hash>` payloads + `DIR/manifest.json` whose changes carry
//! `blob_ref` — exactly what `mlopt.manifest.write_manifest` produces.
//!
//! Refusals are named on stderr with exit 1, verbatim from the
//! emitter — never a silent fallback.

use std::io::Read;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use compile_cuda::{CudaEmitter, KernelTerm, TritonEmitter};
use compile_triton::artifact_name;
use hnl_tool_codegen::{Change, ContentHash, EmitContext, Emitter};
use serde_json::{json, Value};

const GENERATOR_NAME: &str = "mlopt-kernel-lower";
const GENERATOR_VERSION: &str = "0.1.0dev";
const MANIFEST_VERSION: u64 = 1;
const HASH_ALGORITHM: &str = "blake3-256";
const BLOB_DIRNAME: &str = "blobs";
const MANIFEST_FILENAME: &str = "manifest.json";

fn usage() -> ! {
    eprintln!(
        "usage: kernel-lower --target triton|cuda [--block-n N] [--block-k N]\n\
         \x20             [--run-id ID] [--manifest-dir DIR] < term.json"
    );
    std::process::exit(2);
}

/// Provenance run id when the caller doesn't supply one — time+pid,
/// unique enough for a manifest record (journaling is apply-side).
fn default_run_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:x}-{:x}", std::process::id())
}

/// One materialized artifact → the wire change dict (inline form).
fn change_wire(change: &Change) -> Value {
    match change {
        Change::Put {
            artifact,
            before_hash,
            content,
        } => json!({
            "op": "put",
            "artifact": artifact.as_str(),
            "before_hash": before_hash.map(|h| h.to_hex()),
            "content_b64": base64::engine::general_purpose::STANDARD.encode(content),
            "hash": ContentHash::of(content).to_hex(),
        }),
        Change::Delete { artifact } => json!({
            "op": "delete",
            "artifact": artifact.as_str(),
        }),
    }
}

/// The verify-semantics record — mirrors kernelopt `lower.py`'s
/// `verify_manifest_block` field-for-field.
fn verify_block() -> Value {
    json!({
        "level": "term-denotation",
        "executor_gate": "KernelSink.verify (in-process torch rel-diff)",
        "artifact_gate": Value::Null,
        "artifact_gate_reason": "no Triton/CUDA runner exists yet — emitted source is unverified beyond the term's denotation",
        "referee": "measured wall-time in search(), pre-materialize",
    })
}

/// Write `DIR/blobs/<hash>` for every put + `DIR/manifest.json` in the
/// apply-shim wire form (blob_ref changes). Returns the manifest path.
fn write_manifest_dir(
    dir: &Path,
    target: &str,
    run_id: &str,
    term_hash: &str,
    changes: &[Change],
) -> Result<std::path::PathBuf, String> {
    let blobs = dir.join(BLOB_DIRNAME);
    std::fs::create_dir_all(&blobs)
        .map_err(|e| format!("cannot create {}: {e}", blobs.display()))?;

    let mut wire_changes = Vec::with_capacity(changes.len());
    let mut artifacts = Vec::new();
    for ch in changes {
        match ch {
            Change::Put {
                artifact,
                before_hash,
                content,
            } => {
                let hash = ContentHash::of(content).to_hex();
                let blob_path = blobs.join(&hash);
                if !blob_path.exists() {
                    let tmp = blob_path.with_file_name(format!(".{hash}.tmp"));
                    std::fs::write(&tmp, content)
                        .map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
                    std::fs::rename(&tmp, &blob_path)
                        .map_err(|e| format!("cannot rename {}: {e}", blob_path.display()))?;
                }
                wire_changes.push(json!({
                    "op": "put",
                    "artifact": artifact.as_str(),
                    "before_hash": before_hash.map(|h| h.to_hex()),
                    "blob_ref": format!("{BLOB_DIRNAME}/{hash}"),
                    "hash": hash,
                }));
                artifacts.push(json!({"artifact": artifact.as_str(), "hash": hash}));
            }
            Change::Delete { artifact } => {
                wire_changes.push(json!({"op": "delete", "artifact": artifact.as_str()}));
            }
        }
    }
    artifacts.sort_by(|a, b| a["artifact"].as_str().cmp(&b["artifact"].as_str()));

    let doc = json!({
        "v": MANIFEST_VERSION,
        "hash_algorithm": HASH_ALGORITHM,
        "target": target,
        "generator": GENERATOR_NAME,
        "generator_version": format!("{GENERATOR_NAME}/{GENERATOR_VERSION}"),
        "run_id": run_id,
        "source_hashes": {"term": term_hash},
        "artifacts": artifacts,
        "changes": wire_changes,
        "verify": verify_block(),
        "manifest_dir": dir.to_string_lossy(),
    });
    let path = dir.join(MANIFEST_FILENAME);
    let tmp = dir.join(format!(".{MANIFEST_FILENAME}.tmp"));
    std::fs::write(&tmp, serde_json::to_string_pretty(&doc).unwrap() + "\n")
        .map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &path).map_err(|e| format!("cannot rename {}: {e}", path.display()))?;
    Ok(path)
}

fn main() -> std::process::ExitCode {
    let mut target: Option<String> = None;
    let mut block_n = 64usize;
    let mut block_k = 128usize;
    let mut run_id: Option<String> = None;
    let mut manifest_dir: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--target" => target = args.next(),
            "--block-n" => block_n = args.next().and_then(|v| v.parse().ok()).unwrap_or(64),
            "--block-k" => block_k = args.next().and_then(|v| v.parse().ok()).unwrap_or(128),
            "--run-id" => run_id = args.next(),
            "--manifest-dir" => manifest_dir = args.next(),
            other => {
                eprintln!("unknown flag {other}");
                usage();
            }
        }
    }
    let Some(target) = target else {
        eprintln!("kernel-lower: --target triton|cuda is required");
        usage();
    };

    let mut raw = String::new();
    if let Err(err) = std::io::stdin().read_to_string(&mut raw) {
        eprintln!("cannot read the term: {err}");
        return std::process::ExitCode::FAILURE;
    }
    let term = match KernelTerm::parse(raw.trim()) {
        Ok(t) => t,
        Err(err) => {
            eprintln!("the term is not valid JSON: {err}");
            return std::process::ExitCode::FAILURE;
        }
    };

    let changes = match target.as_str() {
        "triton" => TritonEmitter { block_n, block_k }.materialize(&term, &EmitContext::new()),
        "cuda" => CudaEmitter.materialize(&term, &EmitContext::new()),
        other => {
            eprintln!("kernel-lower: unknown target {other:?} — expected triton|cuda");
            return std::process::ExitCode::FAILURE;
        }
    };
    let changes = match changes {
        Ok(c) => c,
        Err(err) => {
            eprintln!("kernel-lower: {target} emitter refused the term: {err}");
            return std::process::ExitCode::FAILURE;
        }
    };

    let run_id = run_id.unwrap_or_else(default_run_id);
    let term_hash = ContentHash::of(term.raw().as_bytes()).to_hex();
    let wire_changes: Vec<Value> = changes.iter().map(change_wire).collect();
    let artifacts: Vec<Value> = changes
        .iter()
        .filter_map(|c| match c {
            Change::Put {
                artifact, content, ..
            } => Some(json!({
                "artifact": artifact.as_str(),
                "hash": ContentHash::of(content).to_hex(),
            })),
            Change::Delete { .. } => None,
        })
        .collect();

    // Sanity: the emitter-named artifact must equal the shared slug —
    // catches a backend that drifts from the kernelopt wire name.
    for c in &changes {
        if let Change::Put { artifact, .. } = c {
            let ext = if target == "triton" { "py" } else { "cu" };
            let want = artifact_name(&target, term.raw(), ext);
            if *artifact != want {
                eprintln!("kernel-lower: emitter artifact {artifact} != expected slug {want}");
                return std::process::ExitCode::FAILURE;
            }
        }
    }

    if let Some(dir) = manifest_dir {
        let dir = Path::new(&dir);
        match write_manifest_dir(dir, &target, &run_id, &term_hash, &changes) {
            Ok(path) => eprintln!("manifest: {}", path.display()),
            Err(err) => {
                eprintln!("kernel-lower: cannot write manifest dir: {err}");
                return std::process::ExitCode::FAILURE;
            }
        }
    }

    let doc = json!({
        "v": MANIFEST_VERSION,
        "hash_algorithm": HASH_ALGORITHM,
        "target": target,
        "generator": GENERATOR_NAME,
        "generator_version": format!("{GENERATOR_NAME}/{GENERATOR_VERSION}"),
        "run_id": run_id,
        "source_hashes": {"term": term_hash},
        "artifacts": artifacts,
        "changes": wire_changes,
        "verify": verify_block(),
    });
    println!("{}", serde_json::to_string_pretty(&doc).unwrap());
    std::process::ExitCode::SUCCESS
}
