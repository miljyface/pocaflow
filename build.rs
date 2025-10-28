use std::env;
use std::process::Command;

fn detect_gpu_arch() -> String {
    // Try to detect GPU architecture automatically
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=compute_cap")
        .arg("--format=csv,noheader")
        .output();
    
    if let Ok(output) = output {
        if let Ok(compute_cap) = String::from_utf8(output.stdout) {
            let compute_cap = compute_cap.trim().replace('.', "");
            if !compute_cap.is_empty() {
                println!("cargo:warning=Detected GPU compute capability: sm_{}", compute_cap);
                return format!("sm_{}", compute_cap);
            }
        }
    }
    
    // Fallback: use safe default for most modern GPUs
    println!("cargo:warning=Could not detect GPU, using sm_75 (Turing) as default");
    "sm_75".to_string()
}

fn main() {
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let arch = detect_gpu_arch();
    
    println!("cargo:warning=Building CUDA kernel with architecture: {}", arch);

    // Compile CUDA kernel
    cc::Build::new()
        .cuda(true)
        .flag(&format!("-arch={}", arch))
        .flag("-O3")
        .flag("--use_fast_math")
        .flag("-Xcompiler")
        .flag("-fPIC")
        .file("src/cuda/matmul_kernel.cu")
        .compile("matmul_kernel");

    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rerun-if-changed=src/cuda/matmul_kernel.cu");
}
