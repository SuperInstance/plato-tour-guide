//! CUDA Runtime — GPU kernel loading and execution.
//!
//! This module is only compiled when the `cuda` feature is enabled.
//! It wraps the `cust` crate for safe CUDA driver interaction.
//!
//! ## Architecture
//!
//! 1. **Build time**: `build.rs` compiles `kernels/*.cu` → PTX via nvcc
//! 2. **Compile time**: PTX is embedded via `include_bytes!`
//! 3. **Runtime**: `cust` loads the PTX into CUDA modules and launches kernels
//!
//! ## Kernel Loading
//!
//! PTX strings are loaded from the OUT_DIR at compile time. Kernel functions
//! are looked up by name in the CUDA module.

use std::ffi::CString;
use std::sync::OnceLock;

/// Global CUDA context — initialized once and cached.
static CUDA_CTX: OnceLock<CudaContext> = OnceLock::new();

/// Safe wrapper around a CUDA context with loaded kernels.
pub struct CudaContext {
    /// The CUDA device
    device: cust::device::Device,
    /// The CUDA context
    context: cust::ctx::Context,
    /// Loaded kernel modules
    modules: CudaModules,
    /// Device properties
    pub device_props: cust::memory::DeviceProp,
}

/// Loaded kernel modules (PTX artifacts).
struct CudaModules {
    /// Distance computation module
    distance: cust::module::Module,
    /// Reduction module
    reduce: cust::module::Module,
}

impl CudaContext {
    /// Initialize CUDA context — loads PTX kernels.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize CUDA driver
        cust::quick_init()?;

        let device = cust::device::Device::get_device(0)?;
        let context = device.create_context()?;
        let device_props = device.get_properties()?;

        // Load PTX modules
        let distance_ptx = include_bytes!(concat!(env!("OUT_DIR"), "/distance.ptx"));
        let reduce_ptx = include_bytes!(concat!(env!("OUT_DIR"), "/reduce.ptx"));

        let distance = cust::module::Module::from_ptx(
            distance_ptx,
            &CString::new("distance.cu").unwrap(),
        )?;

        let reduce = cust::module::Module::from_ptx(
            reduce_ptx,
            &CString::new("reduce.cu").unwrap(),
        )?;

        Ok(Self {
            device,
            context,
            modules: CudaModules { distance, reduce },
            device_props,
        })
    }

    /// Get or create the global CUDA context.
    pub fn global() -> Result<&'static CudaContext, Box<dyn std::error::Error>> {
        CUDA_CTX.get_or_try_init(Self::new)
    }

    /// Upload a slice of f32 data to the GPU.
    pub fn upload(&self, data: &[f32]) -> Result<CudaBuffer, ConsensusCudaError> {
        let n_bytes = data.len() * std::mem::size_of::<f32>();
        let device_ptr = unsafe {
            cust::memory::DeviceBuffer::<f32>::new(data.len())
                .map_err(|e| ConsensusCudaError::Allocation(e.to_string()))?
        };
        device_ptr.copy_from(data)?;
        Ok(CudaBuffer { inner: device_ptr })
    }

    /// Allocate a zero-initialized buffer on the GPU.
    pub fn alloc_zeros(&self, len: usize) -> Result<CudaBuffer, ConsensusCudaError> {
        let device_ptr = unsafe {
            cust::memory::DeviceBuffer::<f32>::new(len)
                .map_err(|e| ConsensusCudaError::Allocation(e.to_string()))?
        };
        Ok(CudaBuffer { inner: device_ptr })
    }

    /// Download data from GPU to CPU.
    pub fn download(&self, buffer: &CudaBuffer, len: usize) -> Result<Vec<f32>, ConsensusCudaError> {
        let mut host = vec![0.0f32; len];
        buffer.inner.copy_to(&mut host)?;
        Ok(host)
    }

    /// Access the distance module.
    pub fn distance_module(&self) -> &cust::module::Module {
        &self.modules.distance
    }

    /// Access the reduce module.
    pub fn reduce_module(&self) -> &cust::module::Module {
        &self.modules.reduce
    }

    /// Get device properties.
    pub fn device_name(&self) -> String {
        let name = self.device_props.name();
        name.to_string_lossy().into_owned()
    }

    /// Get number of SMs on the device.
    pub fn sm_count(&self) -> i32 {
        self.device_props.multi_processor_count
    }

    /// Get max threads per block.
    pub fn max_threads_per_block(&self) -> i32 {
        self.device_props.max_threads_per_block
    }
}

/// A buffer of f32 data on the GPU.
pub struct CudaBuffer {
    inner: cust::memory::DeviceBuffer<f32>,
}

/// CUDA-specific errors.
#[derive(Debug)]
pub enum ConsensusCudaError {
    Allocation(String),
    KernelLaunch(String),
    Download(String),
    Upload(String),
    ModuleLoad(String),
}

impl std::fmt::Display for ConsensusCudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Allocation(msg) => write!(f, "CUDA allocation error: {msg}"),
            Self::KernelLaunch(msg) => write!(f, "CUDA kernel launch error: {msg}"),
            Self::Download(msg) => write!(f, "CUDA download error: {msg}"),
            Self::Upload(msg) => write!(f, "CUDA upload error: {msg}"),
            Self::ModuleLoad(msg) => write!(f, "CUDA module load error: {msg}"),
        }
    }
}

impl std::error::Error for ConsensusCudaError {}

impl From<cust::error::CudaError> for ConsensusCudaError {
    fn from(e: cust::error::CudaError) -> Self {
        ConsensusCudaError::KernelLaunch(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// GPU Distance Computation
// ---------------------------------------------------------------------------

/// GPU-accelerated distance operations.
pub struct CudaDistance;

impl CudaDistance {
    /// Compute the full pairwise cosine distance matrix on GPU.
    ///
    /// Launches a 2D grid, one thread per (i,j) pair.
    ///
    /// # Args
    ///
    /// * `ctx` - CUDA context
    /// * `embeddings` - Embedding buffer on GPU [n, dim]
    /// * `n` - Number of agents
    /// * `dim` - Embedding dimension
    ///
    /// # Returns
    ///
    /// A GPU buffer of size [n, n] with pairwise cosine distances.
    pub fn cosine_distance_matrix(
        ctx: &CudaContext,
        embeddings: &CudaBuffer,
        n: usize,
        dim: usize,
    ) -> Result<CudaBuffer, ConsensusCudaError> {
        let output = ctx.alloc_zeros(n * n)?;

        let distance_kernel = ctx
            .distance_module()
            .get_function(&CString::new("cosine_distance_kernel").unwrap())
            .map_err(|e| ConsensusCudaError::ModuleLoad(e.to_string()))?;

        // Grid/block configuration
        let threads_per_block = 16; // 16×16 = 256 threads per block
        let blocks_x = (n + threads_per_block - 1) / threads_per_block;
        let blocks_y = (n + threads_per_block - 1) / threads_per_block;

        let grid = (blocks_x as u32, blocks_y as u32, 1);
        let block = (threads_per_block as u32, threads_per_block as u32, 1);

        // Launch kernel
        unsafe {
            distance_kernel.launch(
                grid,
                block,
                0, // shared memory bytes
                None, // default stream
                &[
                    &embeddings.inner.as_device_ptr() as *const _ as *const std::ffi::c_void,
                    &n as *const _ as *const std::ffi::c_void,
                    &dim as *const _ as *const std::ffi::c_void,
                    &output.inner.as_device_ptr() as *const _ as *const std::ffi::c_void,
                ],
            )
            .map_err(|e| ConsensusCudaError::KernelLaunch(e.to_string()))?;
        }

        Ok(output)
    }

    /// Compute partial distance matrix for incremental swarm updates.
    ///
    /// Only computes the new rows/columns when agents join an existing swarm.
    ///
    /// # Args
    ///
    /// * `ctx` - CUDA context
    /// * `all_embeddings` - Full embedding buffer [total, dim]
    /// * `total` - Total number of agents (old + new)
    /// * `old_count` - Number of existing agents
    /// * `dim` - Embedding dimension
    /// * `dist_matrix` - Existing distance matrix [total, total] (in/out, partial update)
    pub fn batch_update(
        ctx: &CudaContext,
        all_embeddings: &CudaBuffer,
        total: usize,
        old_count: usize,
        dim: usize,
        dist_matrix: &CudaBuffer,
    ) -> Result<(), ConsensusCudaError> {
        let update_kernel = ctx
            .distance_module()
            .get_function(&CString::new("batch_update_kernel").unwrap())
            .map_err(|e| ConsensusCudaError::ModuleLoad(e.to_string()))?;

        let threads = 16;
        let blocks = (total + threads - 1) / threads;

        unsafe {
            update_kernel.launch(
                (blocks as u32, blocks as u32, 1),
                (threads as u32, threads as u32, 1),
                0,
                None,
                &[
                    &all_embeddings.inner.as_device_ptr() as *const _ as *const std::ffi::c_void,
                    &total as *const _ as *const std::ffi::c_void,
                    &old_count as *const _ as *const std::ffi::c_void,
                    &dim as *const _ as *const std::ffi::c_void,
                    &dist_matrix.inner.as_device_ptr() as *const _ as *const std::ffi::c_void,
                ],
            )
            .map_err(|e| ConsensusCudaError::KernelLaunch(e.to_string()))?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GPU Reduction
// ---------------------------------------------------------------------------

/// GPU-accelerated reduction operations.
pub struct CudaReduce;

impl CudaReduce {
    /// Compute the maximum (spread) across a GPU buffer using tree reduction.
    ///
    /// Recursively reduces until a single value remains.
    ///
    /// # Args
    ///
    /// * `ctx` - CUDA context
    /// * `buffer` - GPU buffer of f32 values
    /// * `n` - Number of elements
    ///
    /// # Returns
    ///
    /// The maximum value
    pub fn max_reduce(
        ctx: &CudaContext,
        buffer: &CudaBuffer,
        n: usize,
    ) -> Result<f32, ConsensusCudaError> {
        let reduce_kernel = ctx
            .reduce_module()
            .get_function(&CString::new("max_reduce_kernel").unwrap())
            .map_err(|e| ConsensusCudaError::ModuleLoad(e.to_string()))?;

        let threads = 256;

        // Single-pass for small arrays, multi-pass for large
        if n <= threads {
            // Single block
            let mut output = unsafe {
                cust::memory::DeviceBuffer::<f32>::new(1)
                    .map_err(|e| ConsensusCudaError::Allocation(e.to_string()))?
            };

            unsafe {
                reduce_kernel.launch(
                    (1, 1, 1),
                    (threads as u32, 1, 1),
                    threads * std::mem::size_of::<f32>() as u32, // shared memory
                    None,
                    &[
                        &buffer.inner.as_device_ptr() as *const _ as *const std::ffi::c_void,
                        &n as *const _ as *const std::ffi::c_void,
                        &output.as_device_ptr() as *const _ as *const std::ffi::c_void,
                    ],
                )
                .map_err(|e| ConsensusCudaError::KernelLaunch(e.to_string()))?;
            }

            let mut host = vec![0.0f32; 1];
            output.copy_to(&mut host)?;
            return Ok(host[0]);
        }

        // Multi-pass reduction: recursively reduce n -> n/threads -> ...
        let mut current = buffer.inner.as_device_ptr();
        let mut current_n = n;
        let mut temp_buffers: Vec<cust::memory::DeviceBuffer<f32>> = Vec::new();

        loop {
            let n_blocks = (current_n + threads - 1) / threads;
            let mut output = unsafe {
                cust::memory::DeviceBuffer::<f32>::new(n_blocks)
                    .map_err(|e| ConsensusCudaError::Allocation(e.to_string()))?
            };

            unsafe {
                reduce_kernel.launch(
                    (n_blocks as u32, 1, 1),
                    (threads as u32, 1, 1),
                    threads * std::mem::size_of::<f32>() as u32,
                    None,
                    &[
                        &current as *const _ as *const std::ffi::c_void,
                        &current_n as *const _ as *const std::ffi::c_void,
                        &output.as_device_ptr() as *const _ as *const std::ffi::c_void,
                    ],
                )
                .map_err(|e| ConsensusCudaError::KernelLaunch(e.to_string()))?;
            }

            if n_blocks == 1 {
                let mut host = vec![0.0f32; 1];
                output.copy_to(&mut host)?;
                return Ok(host[0]);
            }

            current = output.as_device_ptr();
            temp_buffers.push(output);
            current_n = n_blocks;
        }
    }

    /// Early termination spread check — returns true if any pair exceeds threshold.
    ///
    /// Much faster than full reduction when you just need a yes/no answer.
    ///
    /// # Args
    ///
    /// * `ctx` - CUDA context
    /// * `dist_matrix` - Distance matrix on GPU [n, n]
    /// * `n` - Matrix size
    /// * `threshold` - Spread threshold to check
    pub fn spread_exceeds_threshold(
        ctx: &CudaContext,
        dist_matrix: &CudaBuffer,
        n: usize,
        threshold: f32,
    ) -> Result<bool, ConsensusCudaError> {
        let early_kernel = ctx
            .reduce_module()
            .get_function(&CString::new("early_termination_spread_kernel").unwrap())
            .map_err(|e| ConsensusCudaError::ModuleLoad(e.to_string()))?;

        let mut gpu_result = unsafe {
            cust::memory::DeviceBuffer::<i32>::new(1)
                .map_err(|e| ConsensusCudaError::Allocation(e.to_string()))?
        };
        // Initialize to 0
        gpu_result.copy_from(&[0i32])?;

        let threads = 16;
        let blocks = (n + threads - 1) / threads;

        unsafe {
            early_kernel.launch(
                (blocks as u32, blocks as u32, 1),
                (threads as u32, threads as u32, 1),
                0,
                None,
                &[
                    &dist_matrix.inner.as_device_ptr() as *const _ as *const std::ffi::c_void,
                    &n as *const _ as *const std::ffi::c_void,
                    &n as *const _ as *const std::ffi::c_void,
                    &threshold as *const _ as *const std::ffi::c_void,
                    &gpu_result.as_device_ptr() as *const _ as *const std::ffi::c_void,
                ],
            )
            .map_err(|e| ConsensusCudaError::KernelLaunch(e.to_string()))?;
        }

        let mut host = vec![0i32; 1];
        gpu_result.copy_to(&mut host)?;
        Ok(host[0] != 0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_init() {
        // This test may be skipped if no CUDA driver
        let ctx = CudaContext::new();
        match ctx {
            Ok(ctx) => {
                println!("CUDA device: {}", ctx.device_name());
                println!("  SMs: {}", ctx.sm_count());
                println!("  Max threads/block: {}", ctx.max_threads_per_block());
            }
            Err(e) => {
                eprintln!("CUDA not available (expected in CI): {e}");
                return;
            }
        }
    }

    #[test]
    fn test_gpu_cosine_distance() {
        let ctx = match CudaContext::new() {
            Ok(c) => c,
            Err(_) => return, // Skip if no GPU
        };

        // Create 3 normalized vectors: v0=[1,0], v1=[0,1] (orthogonal), v2=[-1,0] (opposite to v0)
        let embeddings = vec![1.0f32, 0.0, 0.0, 1.0, -1.0, 0.0];
        let gpu_emb = ctx.upload(&embeddings).unwrap();

        let dist = CudaDistance::cosine_distance_matrix(&ctx, &gpu_emb, 3, 2).unwrap();
        let host_dist = ctx.download(&dist, 9).unwrap();

        // v0-v0: 0.0
        assert!((host_dist[0 * 3 + 0]).abs() < 1e-5);
        // v0-v1: 1.0 (orthogonal)
        assert!((host_dist[0 * 3 + 1] - 1.0).abs() < 1e-5);
        // v0-v2: 2.0 (opposite)
        assert!((host_dist[0 * 3 + 2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_gpu_spread_reduce() {
        let ctx = match CudaContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };

        // Simple array with known max
        let data = vec![0.1f32, 0.5, 0.9, 0.3, 0.7];
        let gpu_data = ctx.upload(&data).unwrap();

        let max_val = CudaReduce::max_reduce(&ctx, &gpu_data, 5).unwrap();
        assert!((max_val - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_gpu_early_termination() {
        let ctx = match CudaContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };

        // Distance matrix where max is 0.8
        let data = vec![
            0.0, 0.3, 0.5,
            0.3, 0.0, 0.8,
            0.5, 0.8, 0.0,
        ];
        let gpu_data = ctx.upload(&data).unwrap();

        // Should find value > 0.7
        let exceeds = CudaReduce::spread_exceeds_threshold(&ctx, &gpu_data, 3, 0.7).unwrap();
        assert!(exceeds, "max=0.8 should exceed threshold=0.7");

        // Should NOT find value > 0.9
        let exceeds = CudaReduce::spread_exceeds_threshold(&ctx, &gpu_data, 3, 0.9).unwrap();
        assert!(!exceeds, "max=0.8 should not exceed threshold=0.9");
    }
}
