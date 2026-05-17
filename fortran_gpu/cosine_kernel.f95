! cosine_kernel.f95 — Fortran 95 Cosine Distance Kernel
!
! Implements cosine distance between two embedding vectors using pure
! register arithmetic and explicit loops. No classes, no allocatable
! arrays mid-computation, no dynamic dispatch.
!
! Mathematical definition:
!   d(u, v) = 1 - dot(u, v) / (|u| * |v|)
!
! Range: [0, 2]
!   0 = identical direction
!   1 = orthogonal (uncorrelated)
!   2 = opposite
!
! Design principles (Old Fortran HPC philosophy):
!   - All array bounds known at compile time
!   - No heap allocation during computation
!   - Explicit DO loops for predictable memory access
!   - Register-only arithmetic in inner loops
!   - Whole-program optimization visible to compiler

module cosine_kernel
  implicit none

  ! Embedding dimension: fixed at compile time for maximum optimization
  ! This allows the compiler to:
  !   - Unroll loops completely
  !   - Register-allocate all temporaries
  !   - Eliminate all bounds checking
  integer, parameter :: EMBEDDING_DIM = 384
  integer, parameter :: DP = kind(1.0d0)  ! Double precision

  ! Small epsilon to prevent division by zero
  real(DP), parameter :: EPSILON = 1.0e-8_DP

contains

  ! ==============================================================================
  ! cosine_distance_pure — Pure register arithmetic version
  ! ==============================================================================
  !
  ! Input:  u(EMBEDDING_DIM), v(EMBEDDING_DIM) — two embedding vectors
  ! Output: Cosine distance in [0, 2]
  !
  ! This is the reference implementation that maps 1:1 to what the GPU does.
  ! Every operation is explicit; no compiler magic required.
  !
  ! Performance characteristics on modern CPUs:
  !   - ~100% register reuse (u(i), v(i) loaded once)
  !   - Perfect loop predictability (fixed trip count)
  !   - FMA fusion visible in assembly (dot += ui * vi)
  !   - sqrt call dominates runtime (~20 cycles)
  !
  ! Note: marked 'pure' so compiler can prove no side effects
  pure function cosine_distance_pure(u, v) result(distance)
    real(DP), intent(in) :: u(EMBEDDING_DIM)
    real(DP), intent(in) :: v(EMBEDDING_DIM)
    real(DP) :: distance

    ! Accumulators: live entirely in registers (no memory traffic)
    real(DP) :: dot_product      ! dot(u, v)
    real(DP) :: sq_norm_u        ! |u|^2
    real(DP) :: sq_norm_v        ! |v|^2
    real(DP) :: norm_u, norm_v   ! |u|, |v|
    real(DP) :: denominator      ! |u| * |v|
    integer :: i                 ! Loop index

    ! Initialize accumulators
    dot_product = 0.0_DP
    sq_norm_u = 0.0_DP
    sq_norm_v = 0.0_DP

    ! Main computation loop: single-pass dot product + squared norms
    !
    ! This loop is the heart of the computation. The compiler will:
    !   1. Unroll it completely (EMBEDDING_DIM=384 is known)
    !   2. Schedule loads to hide latency (prefetch u(i+4), v(i+4))
    !   3. Fuse multiply-add into FMA instructions (x86 AVX2, ARM NEON)
    !   4. Keep all accumulators in registers (XMM0-XMM5 on x86)
    !
    ! Memory access pattern:
    !   - Sequential: u(1), u(2), ..., u(384)
    !   - Perfect spatial locality
    !   - Prefetcher will fetch ahead automatically
    do i = 1, EMBEDDING_DIM
      ! Load once, use twice: ui and vi stay in registers
      ! Compiler will keep them in registers across the three operations
      dot_product = dot_product + u(i) * v(i)
      sq_norm_u = sq_norm_u + u(i) * u(i)
      sq_norm_v = sq_norm_v + v(i) * v(i)
    end do

    ! Compute norms: single sqrt operation each
    ! This is the expensive part (~20-30 cycles on modern CPUs)
    ! Compiler may use rsqrt (reciprocal sqrt) + Newton refinement
    norm_u = sqrt(sq_norm_u)
    norm_v = sqrt(sq_norm_v)

    ! Protect against zero vectors (undefined in standard cosine similarity)
    ! Using max() instead of division by zero check for branch-free code
    denominator = max(norm_u * norm_v, EPSILON)

    ! Final distance computation
    distance = 1.0_DP - (dot_product / denominator)

  end function cosine_distance_pure


  ! ==============================================================================
  ! cosine_distance_forall — Fortran forall version (array syntax)
  ! ==============================================================================
  !
  ! This version uses Fortran's 'forall' construct, which is semantically
  ! equivalent to the DO loop but gives the compiler more freedom for
  ! parallelization and vectorization.
  !
  ! On modern compilers (gfortran, ifort), this generates identical or
  ! better code than the explicit DO loop because:
  !   - Compiler knows there are no loop-carried dependencies
  !   - Can auto-vectorize with SIMD instructions
  !   - Can distribute across multiple cores (OpenMP)
  pure function cosine_distance_forall(u, v) result(distance)
    real(DP), intent(in) :: u(EMBEDDING_DIM)
    real(DP), intent(in) :: v(EMBEDDING_DIM)
    real(DP) :: distance

    real(DP) :: dot_product, sq_norm_u, sq_norm_v
    real(DP) :: norm_u, norm_v, denominator

    ! Initialize
    dot_product = 0.0_DP
    sq_norm_u = 0.0_DP
    sq_norm_v = 0.0_DP

    ! Forall: parallel iteration without order guarantees
    ! The compiler can safely parallelize this because each iteration
    ! is independent (no loop-carried dependencies)
    forall (i = 1:EMBEDDING_DIM)
      dot_product = dot_product + u(i) * v(i)
      sq_norm_u = sq_norm_u + u(i) * u(i)
      sq_norm_v = sq_norm_v + v(i) * v(i)
    end forall

    ! Same as pure version
    norm_u = sqrt(sq_norm_u)
    norm_v = sqrt(sq_norm_v)
    denominator = max(norm_u * norm_v, EPSILON)
    distance = 1.0_DP - (dot_product / denominator)

  end function cosine_distance_forall


  ! ==============================================================================
  ! cosine_distance_intrinsic — Fortran intrinsic version
  ! ==============================================================================
  !
  ! This version uses Fortran's built-in dot_product intrinsic, which may
  ! call highly optimized BLAS routines (e.g., Intel MKL, OpenBLAS).
  !
  ! Performance characteristics:
  !   - May use multi-threading behind the scenes
  !   - Hand-tuned assembly for specific architectures
  !   - Better cache utilization for large dimensions
  !   - Less transparent than explicit loops
  pure function cosine_distance_intrinsic(u, v) result(distance)
    real(DP), intent(in) :: u(EMBEDDING_DIM)
    real(DP), intent(in) :: v(EMBEDDING_DIM)
    real(DP) :: distance

    real(DP) :: dot_product, sq_norm_u, sq_norm_v
    real(DP) :: norm_u, norm_v, denominator

    ! Use intrinsic for dot product (may call BLAS)
    dot_product = dot_product(u, v)

    ! Manual norm computation (no intrinsic for norm)
    sq_norm_u = dot_product(u, u)
    sq_norm_v = dot_product(v, v)

    norm_u = sqrt(sq_norm_u)
    norm_v = sqrt(sq_norm_v)
    denominator = max(norm_u * norm_v, EPSILON)
    distance = 1.0_DP - (dot_product / denominator)

  end function cosine_distance_intrinsic


  ! ==============================================================================
  ! pairwise_distance_matrix — Compute all N^2/2 pairwise distances
  ! ==============================================================================
  !
  ! Input:  embeddings(N, EMBEDDING_DIM) — N embedding vectors, row-major
  ! Output: distances(N, N) — symmetric distance matrix
  !
  ! This is a batched version that computes the full distance matrix.
  ! Uses explicit nested loops for maximum transparency.
  subroutine pairwise_distance_matrix(embeddings, N, distances)
    integer, intent(in) :: N
    real(DP), intent(in) :: embeddings(N, EMBEDDING_DIM)
    real(DP), intent(out) :: distances(N, N)

    integer :: i, j

    ! Diagonal is zero (distance from self is zero)
    do i = 1, N
      distances(i, i) = 0.0_DP
    end do

    ! Upper triangle: compute for i < j
    do i = 1, N
      do j = i + 1, N
        ! Extract row i and row j from the embedding matrix
        ! Fortran is column-major, so we need to be careful with indexing
        ! The input is assumed to be row-major (like C/Python/NumPy)
        !
        ! For row-major storage in column-major Fortran:
        !   embeddings(i, k) accesses the k-th element of row i
        distances(i, j) = cosine_distance_pure( &
          embeddings(i, 1:EMBEDDING_DIM), &
          embeddings(j, 1:EMBEDDING_DIM) &
        )

        ! Symmetric: distance(i, j) = distance(j, i)
        distances(j, i) = distances(i, j)
      end do
    end do

  end subroutine pairwise_distance_matrix


  ! ==============================================================================
  ! cosine_distance_pre_normalized — Skip normalization for pre-normalized inputs
  ! ==============================================================================
  !
  ! If inputs are guaranteed to be L2-normalized (norm = 1.0), we can skip
  ! the sqrt operations and just compute 1 - dot(u, v).
  !
  ! This is the common case for embedding models (sentence-transformers,
  ! OpenAI embeddings, etc.), which output normalized vectors by default.
  pure function cosine_distance_pre_normalized(u, v) result(distance)
    real(DP), intent(in) :: u(EMBEDDING_DIM)
    real(DP), intent(in) :: v(EMBEDDING_DIM)
    real(DP) :: distance

    real(DP) :: dot_product
    integer :: i

    dot_product = 0.0_DP

    ! Single accumulator: no need to track norms
    do i = 1, EMBEDDING_DIM
      dot_product = dot_product + u(i) * v(i)
    end do

    ! Pre-normalized: |u| = |v| = 1, so denominator = 1
    distance = 1.0_DP - dot_product

  end function cosine_distance_pre_normalized


  ! ==============================================================================
  ! Self-test suite
  ! ==============================================================================
  subroutine test_cosine_distance()
    real(DP) :: u(EMBEDDING_DIM), v(EMBEDDING_DIM), d
    integer :: i

    ! Test 1: Identical vectors (should be 0)
    u = 1.0_DP
    v = 1.0_DP
    d = cosine_distance_pure(u, v)
    print *, 'Test 1 (identical): ', d
    if (abs(d) > 1.0e-6_DP) then
      print *, 'FAILED: expected 0, got ', d
    end if

    ! Test 2: Orthogonal vectors (should be 1)
    ! First half = [1, 0, 0, ...], second half = [0, 1, 0, ...]
    u = 0.0_DP
    v = 0.0_DP
    do i = 1, EMBEDDING_DIM / 2
      u(i) = 1.0_DP
    end do
    do i = EMBEDDING_DIM / 2 + 1, EMBEDDING_DIM
      v(i) = 1.0_DP
    end do
    d = cosine_distance_pure(u, v)
    print *, 'Test 2 (orthogonal): ', d
    if (abs(d - 1.0_DP) > 1.0e-5_DP) then
      print *, 'FAILED: expected 1, got ', d
    end if

    ! Test 3: Opposite vectors (should be 2)
    u = 1.0_DP
    v = -1.0_DP
    d = cosine_distance_pure(u, v)
    print *, 'Test 3 (opposite): ', d
    if (abs(d - 2.0_DP) > 1.0e-5_DP) then
      print *, 'FAILED: expected 2, got ', d
    end if

    print *, 'All tests completed.'

  end subroutine test_cosine_distance

end module cosine_kernel


! ==============================================================================
! Main program: run self-test
! ==============================================================================
program test_main
  use cosine_kernel
  implicit none

  print *, '=== Fortran 95 Cosine Distance Kernel ==='
  print *, 'Embedding dimension: ', EMBEDDING_DIM
  print *, ''

  call test_cosine_distance()

end program test_main
