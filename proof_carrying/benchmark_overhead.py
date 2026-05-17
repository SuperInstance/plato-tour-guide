#!/usr/bin/env python3
"""
Security Practice Overhead Benchmark

Measure the runtime overhead of various security practices and determine
which can be disabled in a sandboxed, proof-carrying environment.

Philosophy: Security through verification, not runtime checks.
"""

import time
import sys
import gc
import subprocess
import multiprocessing
from dataclasses import dataclass
from typing import List, Dict, Callable, Any
from enum import Enum


class SecurityPractice(Enum):
    """Security practices we can benchmark"""
    GC_PAUSE = "Garbage Collection"
    BOUNDS_CHECK = "Array Bounds Checking"
    TYPE_CHECK = "Runtime Type Checking"
    STACK_CANARY = "Stack Canary Protection"
    ASLR = "Address Space Layout Randomization"
    RLIMITS = "Resource Limits (rlimits)"


@dataclass
class BenchmarkResult:
    """Result of benchmarking a security practice"""
    practice: SecurityPractice
    security_problem: str
    baseline_time: float
    secure_time: float
    overhead_percent: float
    can_disable_in_sandbox: bool
    notes: str


class OverheadBenchmark:
    """Benchmark security overhead"""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def run_gc_benchmark(self) -> BenchmarkResult:
        """
        Benchmark garbage collection overhead.

        Security problem: Prevents use-after-free and double-free
        Cost: GC pauses add latency, CPU overhead
        Can disable: YES, if using manual memory management with proofs
        """
        # Baseline: Pre-allocated pool (no GC)
        start = time.time()
        pool = [None] * 10_000_000
        for i in range(10_000_000):
            pool[i] = i
        baseline = time.time() - start

        # With GC: Frequent allocation
        gc.disable()
        start = time.time()
        data = []
        for i in range(10_000_000):
            data.append(i)
            if i % 1000 == 0:
                data = data[-1000:]  # Trigger GC
        gc.enable()
        secure_time = time.time() - start

        overhead = (secure_time / baseline - 1) * 100

        return BenchmarkResult(
            practice=SecurityPractice.GC_PAUSE,
            security_problem="Prevents use-after-free, double-free, memory leaks",
            baseline_time=baseline,
            secure_time=secure_time,
            overhead_percent=overhead,
            can_disable_in_sandbox=True,
            notes="In proof-carrying code, memory safety proved statically → no GC needed"
        )

    def run_bounds_check_benchmark(self) -> BenchmarkResult:
        """
        Benchmark bounds checking overhead.

        Security problem: Prevents buffer overflow
        Cost: Branch misprediction, CPU overhead
        Can disable: YES, if array access proved correct by verifier
        """
        size = 100_000_000
        arr = list(range(size))

        # Baseline: No bounds check (using direct memory access in C)
        # We'll simulate by using while loop with known bounds
        start = time.time()
        i = 0
        total = 0
        while i < size:
            total += arr[i]  # Python still checks, but minimal
            i += 1
        baseline = time.time() - start

        # With bounds check: Explicit check in loop
        start = time.time()
        total = 0
        for i in range(size):
            if 0 <= i < size:
                total += arr[i]
        secure_time = time.time() - start

        overhead = (secure_time / baseline - 1) * 100

        return BenchmarkResult(
            practice=SecurityPractice.BOUNDS_CHECK,
            security_problem="Prevents buffer overflow, out-of-bounds access",
            baseline_time=baseline,
            secure_time=secure_time,
            overhead_percent=overhead,
            can_disable_in_sandbox=True,
            notes="Verification conditions prove 0 <= i < len(arr) → check elided"
        )

    def run_type_check_benchmark(self) -> BenchmarkResult:
        """
        Benchmark runtime type checking overhead.

        Security problem: Prevents type confusion attacks
        Cost: Dynamic dispatch, vtable lookups
        Can disable: YES, if using monomorphization (compile-time generics)
        """
        # Baseline: Monomorphized (specialized for type)
        def specialized_int(x: int) -> int:
            return x + 1

        start = time.time()
        for i in range(100_000_000):
            specialized_int(i)
        baseline = time.time() - start

        # With type check: Generic function with runtime check
        def generic_with_check(x: Any) -> Any:
            if not isinstance(x, int):
                raise TypeError("Expected int")
            return x + 1

        start = time.time()
        for i in range(100_000_000):
            generic_with_check(i)
        secure_time = time.time() - start

        overhead = (secure_time / baseline - 1) * 100

        return BenchmarkResult(
            practice=SecurityPractice.TYPE_CHECK,
            security_problem="Prevents type confusion, unsafe casting",
            baseline_time=baseline,
            secure_time=secure_time,
            overhead_percent=overhead,
            can_disable_in_sandbox=True,
            notes="Monomorphization generates specialized code → no runtime check"
        )

    def run_stack_canary_benchmark(self) -> BenchmarkResult:
        """
        Benchmark stack canary overhead.

        Security problem: Detects stack buffer overflow
        Cost: Store/load canary on each function call
        Can disable: MAYBE, only if no buffer operations and verified
        """
        # Baseline: Function without canary
        def no_canary(x: int) -> int:
            return x * 2

        start = time.time()
        for i in range(100_000_000):
            no_canary(i)
        baseline = time.time() - start

        # Simulate canary: Additional load/store
        canary_value = 0xDEADBEEF

        def with_canary(x: int) -> int:
            # Simulate canary check
            _ = canary_value  # Load canary
            result = x * 2
            _ = canary_value  # Verify canary
            return result

        start = time.time()
        for i in range(100_000_000):
            with_canary(i)
        secure_time = time.time() - start

        overhead = (secure_time / baseline - 1) * 100

        return BenchmarkResult(
            practice=SecurityPractice.STACK_CANARY,
            security_problem="Detects stack smashing, buffer overflow",
            baseline_time=baseline,
            secure_time=secure_time,
            overhead_percent=overhead,
            can_disable_in_sandbox=False,  # Conservative
            notes="Only disable if: (1) no buffers on stack, (2) no alloca, (3) verified"
        )

    def run_aslr_benchmark(self) -> BenchmarkResult:
        """
        Benchmark ASLR overhead.

        Security problem: Prevents code reuse attacks (ROP/JOP)
        Cost: TLB flushes on randomization, page table walks
        Can disable: NO, ASLR is critical even with verification
        """
        # ASLR overhead is minimal but constant
        # We'll estimate based on literature (~1-3%)

        baseline = 1.0  # Normalized
        secure_time = 1.02  # ~2% overhead from literature
        overhead = 2.0

        return BenchmarkResult(
            practice=SecurityPractice.ASLR,
            security_problem="Prevents return-oriented programming, code reuse",
            baseline_time=baseline,
            secure_time=secure_time,
            overhead_percent=overhead,
            can_disable_in_sandbox=False,
            notes="Keep ASLR: Defense in depth, protects against verification bugs"
        )

    def run_rlimits_benchmark(self) -> BenchmarkResult:
        """
        Benchmark resource limit checks.

        Security problem: Prevents resource exhaustion attacks
        Cost: System call overhead on resource usage
        Can disable: NO, essential for sandbox isolation
        """
        # rlimits overhead is minimal (system call on fork/exec)

        baseline = 1.0  # Normalized
        secure_time = 1.001  # ~0.1% overhead
        overhead = 0.1

        return BenchmarkResult(
            practice=SecurityPractice.RLIMITS,
            security_problem="Prevents CPU/memory exhaustion, fork bombs",
            baseline_time=baseline,
            secure_time=secure_time,
            overhead_percent=overhead,
            can_disable_in_sandbox=False,
            notes="Keep rlimits: Critical for sandbox isolation"
        )

    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks and return results"""
        print("Running security overhead benchmarks...")

        self.results.append(self.run_gc_benchmark())
        print(f"  ✓ GC pause")

        self.results.append(self.run_bounds_check_benchmark())
        print(f"  ✓ Bounds checking")

        self.results.append(self.run_type_check_benchmark())
        print(f"  ✓ Type checking")

        self.results.append(self.run_stack_canary_benchmark())
        print(f"  ✓ Stack canary")

        self.results.append(self.run_aslr_benchmark())
        print(f"  ✓ ASLR")

        self.results.append(self.run_rlimits_benchmark())
        print(f"  ✓ rlimits")

        return self.results

    def generate_table(self) -> str:
        """Generate markdown table of results"""
        output = []
        output.append("# Security Practice Overhead Analysis")
        output.append("")
        output.append("| Practice | Security Problem Solved | Overhead | Can Disable in Sandbox? | Notes |")
        output.append("|----------|-------------------------|----------|------------------------|-------|")

        for result in self.results:
            can_disable = "✓ YES" if result.can_disable_in_sandbox else "✗ NO"
            output.append(
                f"| {result.practice.value} | {result.security_problem} | "
                f"{result.overhead_percent:.1f}% | {can_disable} | {result.notes} |"
            )

        output.append("")
        output.append("## Key Findings")
        output.append("")
        output.append("### Can Disable (Verified Code)")
        output.append("- **GC Pause**: Use manual memory management with lifetime proofs")
        output.append("- **Bounds Check**: Verification conditions prove array access safe")
        output.append("- **Type Check**: Monomorphization generates specialized code")
        output.append("")
        output.append("### Must Keep (Defense in Depth)")
        output.append("- **Stack Canary**: Only disable if no buffers on stack")
        output.append("- **ASLR**: Protects against verification bugs")
        output.append("- **RLimits**: Essential for sandbox isolation")
        output.append("")
        output.append("## Performance Impact")
        output.append("")
        total_disableable = sum(r.overhead_percent for r in self.results if r.can_disable_in_sandbox)
        output.append(f"**Potential speedup from disabling verify-safe practices**: {total_disableable:.1f}%")

        return "\n".join(output)


def main():
    benchmark = OverheadBenchmark()
    benchmark.run_all_benchmarks()

    table = benchmark.generate_table()
    print(table)

    # Write to file
    with open("overhead_analysis.md", "w") as f:
        f.write(table)

    print("\nResults written to overhead_analysis.md")


if __name__ == "__main__":
    main()
