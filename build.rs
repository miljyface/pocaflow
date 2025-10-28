use std::env;

fn main() {
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    // Compile CUDA kernel
    cc::Build::new()
        .cuda(true)
        .flag("-arch=sm_89")  // Change based on your GPU
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
