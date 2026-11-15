//! CLI: a term on stdin, a CUDA C++ translation unit on stdout.

use std::io::Read;

use compile_cuda::{compile, Node};

fn main() -> std::process::ExitCode {
    let mut args = std::env::args().skip(1);
    if let Some(flag) = args.next() {
        eprintln!("unknown flag {flag}");
        return std::process::ExitCode::FAILURE;
    }

    let mut raw = String::new();
    if let Err(err) = std::io::stdin().read_to_string(&mut raw) {
        eprintln!("cannot read the term: {err}");
        return std::process::ExitCode::FAILURE;
    }
    let node: Node = match serde_json::from_str(&raw) {
        Ok(node) => node,
        Err(err) => {
            eprintln!("the term is not valid JSON: {err}");
            return std::process::ExitCode::FAILURE;
        }
    };
    match compile(&node) {
        Ok(src) => {
            print!("{src}");
            std::process::ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("not a lowerable term: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}
