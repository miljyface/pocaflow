use std::env;
use std::process::Command;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    
    match target_os.as_str() {
        "linux" => build_cuda(),
        "macos" => println!("cargo:warning=Building Metal backend"),
        _ => println!("cargo:warning=CPU-only mode"),
    }
}

fn build_cuda() {
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    
    let arch = detect_gpu_arch();
    
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let kernel_path = format!("{}/src/kernels/cuda/matmul_kernel.cu", manifest_dir);

    assert!(std::path::Path::new(&kernel_path).exists(), "Kernel file not found: {:?}", kernel_path);

    cc::Build::new()
        .cuda(true)
        .flag(&format!("-arch={}", arch))
        .flag("-O3")
        .flag("--use_fast_math")
        .flag("-Xcompiler")
        .flag("-fPIC")
        .file(&kernel_path)
        .compile("matmul_kernel");
    
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rerun-if-changed=kernels/cuda/matmul_kernel.cu");
}

fn detect_gpu_arch() -> String {
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=compute_cap")
        .arg("--format=csv,noheader")
        .output();
    
    if let Ok(output) = output {
        if let Ok(compute_cap) = String::from_utf8(output.stdout) {
            let compute_cap = compute_cap.trim();
            println!("cargo:warning=Detected GPU compute capability: {}", compute_cap);
            
            let arch = match compute_cap {
                "8.9" => {
                    println!("cargo:warning=RTX 4090 detected, using sm_86 for compatibility");
                    "sm_86"
                },
                "8.6" => "sm_86",
                "8.0" => "sm_80",
                "7.5" => "sm_75",
                "7.0" => "sm_70",
                _ => {
                    println!("cargo:warning=Unknown compute capability {}, using sm_75", compute_cap);
                    "sm_75"
                }
            };
            
            println!("cargo:warning=Using architecture: {}", arch);
            return arch.to_string();
        }
    }
    
    println!("cargo:warning=Could not detect GPU, using sm_75");
    "sm_75".to_string()
}