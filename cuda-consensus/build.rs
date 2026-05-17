// build.rs — Conditionally compiles CUDA kernels to PTX.
//
// If nvcc is available on the build host, compiles kernels/*.cu → kernels/*.ptx.
// If nvcc is NOT available, the crate builds with CPU-only fallback.

use std::path::Path;
use std::process::Command;

fn main() {
    // Rebuild if kernels changed
    println!("cargo:rerun-if-changed=kernels/distance.cu");
    println!("cargo:rerun-if-changed=kernels/reduce.cu");

    let out_dir = std::env::var("OUT_DIR").unwrap_or_else(|_| "target/out".into());

    // Check if nvcc is available
    let nvcc_check = Command::new("nvcc")
        .arg("--version")
        .output();

    let cuda_toolkit = match nvcc_check {
        Ok(output) if output.status.success() => {
            let version_str = String::from_utf8_lossy(&output.stdout);
            let version_line = version_str.lines().find(|l| l.contains("release"));
            eprintln!("[build.rs] CUDA toolkit detected: {}", version_line.unwrap_or("unknown"));
            true
        }
        _ => {
            eprintln!("[build.rs] nvcc not found — building with CPU-only fallback.");
            println!("cargo:rustc-cfg=cpu_only_fallback");
            false
        }
    };

    if cuda_toolkit {
        let kernels_dir = Path::new("kernels");
        let kernel_files = vec!["distance.cu", "reduce.cu"];

        for kernel_file in &kernel_files {
            let src = kernels_dir.join(kernel_file);
            let ptx_name = kernel_file.replace(".cu", ".ptx");
            let dst = Path::new(&out_dir).join(&ptx_name);

            eprintln!("[build.rs] Compiling {} -> {}", kernel_file, dst.display());

            let status = Command::new("nvcc")
                .args([
                    "-ptx",
                    "-O3",
                    "-arch=sm_75",          // Turing+ (RTX 20xx, 30xx, 40xx)
                    "-use_fast_math",
                    "-o",
                ])
                .arg(dst.to_str().unwrap())
                .arg(src.to_str().unwrap())
                .status()
                .expect("nvcc execution failed");

            if !status.success() {
                eprintln!(
                    "[build.rs] WARNING: nvcc compilation failed for {}. \
                     Trying compute capability 6.0 (Pascal+)",
                    kernel_file
                );

                // Retry with older arch
                let dst_old = Path::new(&out_dir).join(format!("{}_sm60.ptx", ptx_name.replace(".ptx", "")));
                let status = Command::new("nvcc")
                    .args([
                        "-ptx",
                        "-O3",
                        "-arch=sm_60",
                        "-use_fast_math",
                        "-o",
                    ])
                    .arg(dst_old.to_str().unwrap())
                    .arg(src.to_str().unwrap())
                    .status()
                    .expect("nvcc retry failed");

                if status.success() {
                    // Copy fallback ptx to expected name
                    std::fs::copy(&dst_old, &dst).ok();
                } else {
                    panic!(
                        "[build.rs] FATAL: Could not compile {} with sm_75 or sm_60. \
                         Install CUDA toolkit >=11.0 or remove CUDA feature flag.",
                        kernel_file
                    );
                }
            }
        }

        println!("cargo:rustc-cfg=cuda_available");
    }
}
