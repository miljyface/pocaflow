use std::env;
use std::process::Command;

fn detect_and_map_gpu_arch() -> String {
    // Try to detect GPU architecture
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=compute_cap")
        .arg("--format=csv,noheader")
        .output();
    
    if let Ok(output) = output {
        if let Ok(compute_cap) = String::from_utf8(output.stdout) {
            let compute_cap = compute_cap.trim();
            println!("cargo:warning=Detected GPU compute capability: {}", compute_cap);
            
            // Map compute capability to architecture supported by CUDA 11.5
            let arch = match compute_cap {
                "8.9" => {
                    println!("cargo:warning=RTX 4090 detected, but CUDA 11.5 doesn't support sm_89");
                    println!("cargo:warning=Using sm_86 (Ampere) for compatibility");
                    "sm_86"
                },
                "8.6" => "sm_86", // RTX 3090/3080
                "8.0" => "sm_80", // A100
                "7.5" => "sm_75", // RTX 2080/Turing
                "7.0" => "sm_70", // V100
                "6.1" => "sm_61", // GTX 1080
                "6.0" => "sm_60", // Pascal P100
                _ => {
                    println!("cargo:warning=Unknown compute capability {}, using sm_75", compute_cap);
                    "sm_75"
                }
            };
            
            println!("cargo:warning=Mapped to architecture: {}", arch);
            return arch.to_string();
        }
    }
    
    // Fallback
    println!("cargo:warning=Could not detect GPU, using sm_75 as default");
    "sm_75".to_string()
}

fn main() {
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let arch = detect_and_map_gpu_arch();
    
    println!("cargo:warning=Final architecture for nvcc: {}", arch);

    // Compile CUDA kernel with detected architecture
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
