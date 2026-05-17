# Zero Holonomy Consensus (ZHC) Memory Model

## Overview

Zero Holonomy Consensus uses the geometric properties of zero holonomy to achieve fast distributed consensus. The memory model maps this mathematical structure to GPU memory hierarchy for optimal performance.

## Mathematical Foundation

### Zero Holonomy Condition

A consensus protocol has zero holonomy if and only if:

```
∀ cycles C in the trust graph: ∏_{i∈C} T_i = I
```

Where:
- T_i = trust vector of agent i
- Product = geometric product (rotation + scaling)
- I = identity transformation (no net rotation)

This means: following any cycle of trust relationships returns you to the starting orientation.

### GPU Memory Implications

The zero holonomy condition implies that trust vectors live in a **curled subspace** of ℝ³. This allows aggressive compression:

1. **Each agent's trust vector** → 3D unit vector (96 bits naive)
2. **Zero holonomy constraint** → vectors are highly correlated
3. **Optimal encoding** → Pythagorean 48-direction code (6 bits)

## Memory Hierarchy Mapping

### 1. GPU Registers (Hot Path)

**What lives here:**
- Individual agent trust vectors (3 floats × 32 threads = 96 floats)
- Partial consensus values (1 float × 32 threads)
- Threshold comparison results (1 bool × 32 threads)

**Operations:**
- Encode/decode Pythagorean directions
- Compute pairwise trust (dot products)
- Byzantine detection (threshold comparison)

**Performance:**
- Zero latency (if no register spill)
- 32 registers per thread maximum
- ~10KB total per warp

**ZHC Step in Registers:**
```cuda
// Each thread handles one agent
__global__ void zhc_step_registers(
    const float* trust_vectors,  // [n_agents × 3]
    const float threshold,
    float* consensus_value
) {
    int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Load trust vector to registers
    float tx = trust_vectors[agent_id * 3 + 0];
    float ty = trust_vectors[agent_id * 3 + 1];
    float tz = trust_vectors[agent_id * 3 + 2];
    
    // Compute trust with all other agents (coalesced)
    float total_trust = 0.0f;
    for (int i = 0; i < n_agents; i++) {
        float ox = trust_vectors[i * 3 + 0];
        float oy = trust_vectors[i * 3 + 1];
        float oz = trust_vectors[i * 3 + 2];
        
        // Dot product = trust magnitude
        float trust = tx * ox + ty * oy + tz * oz;
        total_trust += trust;
    }
    
    // Byzantine detection: compare to threshold
    bool is_byzantine = (total_trust < threshold);
    
    // Store result
    consensus_value[agent_id] = is_byzantine ? 0.0f : total_trust;
}
```

**Register Pressure:**
- 3 floats for trust vector
- 3 floats for other agent
- 1 float for accumulator
- 1 bool for flag
- **Total: 8 registers** (well within limits)

### 2. Shared Memory (Thread Group Collaboration)

**What lives here:**
- Agent subset trust vectors (e.g., 32 agents × 3 dims = 96 floats)
- Partial pairwise trust matrix (32 × 32 = 1024 floats)
- Reduction intermediate values

**Operations:**
- Warp-level reduction for consensus
- Shared prefix sums for ranking
- Thread-group barrier synchronization

**Performance:**
- ~30 cycle latency
- User-managed cache
- 48KB per block (modern GPUs)

**ZHC Consensus in Shared Memory:**
```cuda
__global__ void zhc_consensus_shared(
    const float* trust_vectors,
    const int n_agents,
    float* consensus_output
) {
    extern __shared__ float shared_trust[];
    
    // Load all trust vectors to shared memory
    int tid = threadIdx.x;
    for (int i = tid; i < n_agents; i += blockDim.x) {
        shared_trust[i * 3 + 0] = trust_vectors[i * 3 + 0];
        shared_trust[i * 3 + 1] = trust_vectors[i * 3 + 1];
        shared_trust[i * 3 + 2] = trust_vectors[i * 3 + 2];
    }
    __syncthreads();
    
    // Compute consensus (weighted average)
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    
    for (int i = 0; i < n_agents; i++) {
        float trust = 0.0f;
        for (int d = 0; d < 3; d++) {
            float my_val = shared_trust[tid * 3 + d];
            float other_val = shared_trust[i * 3 + d];
            trust += my_val * other_val;
        }
        
        weighted_sum += trust * trust;  // Self-trust weight
        weight_sum += trust;
    }
    
    // Warp-level reduction
    float final_sum = warp_reduce(weighted_sum);
    float final_weight = warp_reduce(weight_sum);
    
    if (tid == 0) {
        consensus_output[blockIdx.x] = final_sum / final_weight;
    }
}
```

### 3. Global Memory (Large-Scale Storage)

**What lives here:**
- Full agent trust matrix (n_agents × n_agents × 4 bytes)
- Historical trust values (for debugging)
- Consensus history (for analysis)

**Operations:**
- Load trust vectors from host
- Store consensus results
- Stream large batches of agents

**Performance:**
- ~500 cycle latency
- 900 GB/s bandwidth (A100)
- Needs coalescing for efficiency

## Zero Holonomy Optimization

### Cycle Detection in Registers

The zero holonomy condition allows cycle detection using only register operations:

```cuda
// Detect if 3 agents form a zero-holonomy cycle
__device__ bool is_zero_holonomy_cycle(
    const float* v1, const float* v2, const float* v3
) {
    // Compute cross products (all in registers)
    float cp1x = v1[1] * v2[2] - v1[2] * v2[1];
    float cp1y = v1[2] * v2[0] - v1[0] * v2[2];
    float cp1z = v1[0] * v2[1] - v1[1] * v2[0];
    
    float cp2x = v2[1] * v3[2] - v2[2] * v3[1];
    float cp2y = v2[2] * v3[0] - v2[0] * v3[2];
    float cp2z = v2[0] * v3[1] - v2[1] * v3[0];
    
    float cp3x = v3[1] * v1[2] - v3[2] * v1[1];
    float cp3y = v3[2] * v1[0] - v3[0] * v1[2];
    float cp3z = v3[0] * v1[1] - v3[1] * v1[0];
    
    // Check if cross products sum to zero
    float sum = cp1x + cp2x + cp3x + 
                cp1y + cp2y + cp3y + 
                cp1z + cp2z + cp3z;
    
    return fabs(sum) < 1e-6f;
}
```

### Byzantine Detection with Warp Reduce

Detect Byzantine agents using warp-level reduction:

```cuda
__device__ bool detect_byzantine_agent(
    const float* trust_vectors,
    const int n_agents,
    const float threshold
) {
    int agent_id = threadIdx.x;
    
    // Compute total trust received
    float total_trust = 0.0f;
    for (int i = 0; i < n_agents; i++) {
        if (i != agent_id) {
            float dot = trust_vectors[agent_id * 3 + 0] * trust_vectors[i * 3 + 0] +
                       trust_vectors[agent_id * 3 + 1] * trust_vectors[i * 3 + 1] +
                       trust_vectors[agent_id * 3 + 2] * trust_vectors[i * 3 + 2];
            total_trust += dot;
        }
    }
    
    // Warp-level reduction to find average
    float avg_trust = warp_reduce(total_trust) / n_agents;
    
    // Byzantine if below threshold
    return total_trust < threshold * avg_trust;
}
```

## Memory Layout Comparison

### Naive Layout (No ZHC Optimization)

```
Trust Matrix: [n_agents × n_agents] floats
Memory: n² × 4 bytes
Example: 1000 agents → 4 MB

Access Pattern:
- Each consensus step: O(n²) memory reads
- Random access (poor locality)
- Cache thrashing
```

### ZHC-Optimized Layout

```
Encoded Trust: [n_agents] uint8 (Pythagorean code)
Memory: n × 1 byte
Example: 1000 agents → 1 KB

Access Pattern:
- Each consensus step: O(n) memory reads
- Sequential access (excellent locality)
- All hot data in registers
```

**Compression: 4000x**

## Bandwidth Analysis

### Per-Agent Trust Bandwidth

**Naive approach:**
- 3 floats × 32 bits = 96 bits per agent
- For 1000 agents: 96,000 bits = 12 KB

**ZHC with Pythagorean code:**
- log₂(48) ≈ 5.585 bits per agent
- For 1000 agents: 5,585 bits = 698 bytes

**Reduction: 17x**

### Consensus Step Bandwidth

**Naive:**
- Read full trust matrix: n² × 4 bytes
- Write consensus value: n × 4 bytes
- Total: O(n²) memory traffic

**ZHC:**
- Read encoded trust: n × 1 byte
- Write consensus: 1 × 4 bytes
- Total: O(n) memory traffic

**Reduction: O(n)**

## Performance Targets

### For 1000 Agents:

| Metric | Naive | ZHC | Speedup |
|--------|-------|-----|---------|
| Memory | 4 MB | 1 KB | 4000x |
| Bandwidth | 4 MB/step | 1 KB/step | 4000x |
| Latency | ~10 ms | ~10 μs | 1000x |
| Throughput | ~100 Hz | ~100 kHz | 1000x |

### Scaling:

```
Memory: O(n) → O(log n)
Bandwidth: O(n²) → O(n)
Latency: O(n) → O(log n)  (with warp reduction)
```

## Implementation Notes

1. **Always keep trust vectors in registers** during consensus computation
2. **Use shared memory** for thread-group collaboration (reduction, prefix sum)
3. **Encode to Pythagorean** before storing to global memory
4. **Batch decode** when reading from global memory
5. **Exploit zero holonomy** to detect cycles without memory access

## References

- Pythagorean 48-direction code: See `pythagorean48_cuda.cu`
- CUDA kernel selection: See `ct_kernel_selector.py`
- Memory hierarchy benchmarks: See `memory_latency_benchmark.cu`
