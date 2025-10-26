use metal::*;
use ndarray::Array2;
use std::mem;
use std::sync::Mutex;
use std::collections::HashMap;

pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    shader_cache: Mutex<HashMap<String, ComputePipelineState>>,
}

impl MetalContext {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or_else(|| "No Metal device".to_string())?;
        let queue = device.new_command_queue();
        Ok(MetalContext {
            device,
            queue,
            shader_cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn matmul_f32(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>, String> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        
        if k != k2 {
            return Err("Dimension mismatch".to_string());
        }

        // create key for cache
        let tile_size = 64usize;
        let cache_key = format!("matmul_tiled_{}", tile_size);

        // get or compile shader
        let pipeline = {
            let mut cache = self.shader_cache.lock().unwrap();
            
            if let Some(cached) = cache.get(&cache_key) {
                cached.clone()  // use cached pipeline
            } else {
                // shader compile
                let shader = format!(r#"
                #include <metal_stdlib>
                using namespace metal;

                kernel void matmul_tiled(
                    device const float* A [[buffer(0)]],
                    device const float* B [[buffer(1)]],
                    device float* C [[buffer(2)]],
                    constant uint& M [[buffer(3)]],
                    constant uint& N [[buffer(4)]],
                    constant uint& K [[buffer(5)]],
                    uint3 gid [[threadgroup_position_in_grid]],
                    uint3 tid [[thread_position_in_threadgroup]]
                ) {{
                    threadgroup float a_tile[{0}][{0}];
                    threadgroup float b_tile[{0}][{0}];
                    
                    uint row = gid.y * {0} + tid.y;
                    uint col = gid.x * {0} + tid.x;
                    
                    float sum = 0.0f;
                    
                    for (uint t = 0; t < K; t += {0}) {{
                        if (row < M && t + tid.x < K)
                            a_tile[tid.y][tid.x] = A[row * K + t + tid.x];
                        if (t + tid.y < K && col < N)
                            b_tile[tid.y][tid.x] = B[(t + tid.y) * N + col];
                        
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        
                        for (uint k = 0; k < {0}; ++k) {{
                            if (t + k < K)
                                sum += a_tile[tid.y][k] * b_tile[k][tid.x];
                        }}
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }}
                    
                    if (row < M && col < N)
                        C[row * N + col] = sum;
                }}
                "#, tile_size);

                let lib = self.device.new_library_with_source(&shader, &CompileOptions::new())
                    .map_err(|e| format!("Compile: {:?}", e))?;
                let func = lib.get_function("matmul_tiled", None)
                    .map_err(|e| format!("Func: {:?}", e))?;
                let pipeline = self.device.new_compute_pipeline_state_with_function(&func)
                    .map_err(|e| format!("Pipeline: {:?}", e))?;
                
                // Cache it
                cache.insert(cache_key, pipeline.clone());
                pipeline
            }
        };

        // Create buffers
        let a_buf = self.device.new_buffer_with_data(
            a.as_ptr() as *const _, (m * k * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let b_buf = self.device.new_buffer_with_data(
            b.as_ptr() as *const _, (k * n * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let c_buf = self.device.new_buffer(
            (m * n * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Encode compute
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&a_buf), 0);
        enc.set_buffer(1, Some(&b_buf), 0);
        enc.set_buffer(2, Some(&c_buf), 0);
        enc.set_bytes(3, 4, &(m as u32) as *const _ as *const _);
        enc.set_bytes(4, 4, &(n as u32) as *const _ as *const _);
        enc.set_bytes(5, 4, &(k as u32) as *const _ as *const _);

        let tg_size = MTLSize::new(tile_size as u64, tile_size as u64, 1);
        let tg_count = MTLSize::new(
            (n as u64 + tile_size as u64 - 1) / tile_size as u64,
            (m as u64 + tile_size as u64 - 1) / tile_size as u64,
            1,
        );

        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        unsafe {
            let ptr = c_buf.contents() as *const f32;
            let vec = std::slice::from_raw_parts(ptr, m * n).to_vec();
            Array2::from_shape_vec((m, n), vec).map_err(|e| e.to_string())
        }
    }
}
