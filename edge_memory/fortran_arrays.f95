! fortran_arrays.f95
! Fortran 95 showing the ideal memory model for tile-based systems.
!
! Philosophy: Known sizes at compile time. No heap allocation.
! Whole-array operations. Intent contracts. Save attribute for persistence.
!
! This is what our Rust should compile to on the LLVM level.

module tile_memory
    implicit none
    
    ! =============================================================================
    ! CONSTANTS — Compile-time known sizes
    ! =============================================================================
    
    integer, parameter :: MAX_TILES = 1024
    integer, parameter :: TILE_HEADER_SIZE = 32
    integer, parameter :: MAX_ROOM_ID = 65535
    
    ! =============================================================================
    ! TILE_HEADER — Fixed-size, stack-allocated
    ! =============================================================================
    
    type :: TileHeader
        integer(kind=8) :: tile_id
        integer(kind=8) :: room_id
        integer(kind=8) :: sequence
        integer(kind=4) :: flags
        integer(kind=8) :: payload_size
    end type TileHeader
    
    ! =============================================================================
    ! FIXED-SIZE ARRAYS — No allocatable, no pointers
    ! =============================================================================
    
    ! Fixed-size tile storage (no heap)
    type :: TileStore
        type(TileHeader), dimension(MAX_TILES) :: headers
        real(kind=8), dimension(MAX_TILES, 128) :: embeddings  ! 128-dim embeddings
        integer(kind=4), dimension(MAX_TILES) :: active_flags
    end type TileStore
    
    ! =============================================================================
    ! INTENT CONTRACTS — Compiler enforces correct usage
    ! =============================================================================
    
contains

    ! -----------------------------------------------------------------------------
    ! Subroutine: create_tile
    ! Intent(in):    tile_id, room_id, sequence — input parameters
    ! Intent(out):   header — output only, caller provides storage
    ! Intent(inout): store — modified by the operation
    ! -----------------------------------------------------------------------------
    
    subroutine create_tile(store, tile_id, room_id, sequence, header_idx, err)
        type(TileStore), intent(inout) :: store
        integer(kind=8), intent(in) :: tile_id, room_id, sequence
        integer, intent(out) :: header_idx
        integer, intent(out) :: err
        
        ! Find first inactive slot
        header_idx = 0
        err = 0
        
        do header_idx = 1, MAX_TILES
            if (store%active_flags(header_idx) == 0) then
                exit
            end if
        end do
        
        if (header_idx > MAX_TILES) then
            err = 1  ! No space
            return
        end if
        
        ! Initialize header (whole-array assignment)
        store%headers(header_idx)%tile_id = tile_id
        store%headers(header_idx)%room_id = room_id
        store%headers(header_idx)%sequence = sequence
        store%headers(header_idx)%flags = 0
        store%headers(header_idx)%payload_size = 0
        
        ! Mark active
        store%active_flags(header_idx) = 1
        
    end subroutine create_tile
    
    ! -----------------------------------------------------------------------------
    ! Subroutine: set_embedding
    ! Intent(inout): store — modified in place
    ! Purpose: Set embedding vector for a tile (whole-array operation)
    ! -----------------------------------------------------------------------------
    
    subroutine set_embedding(store, idx, embedding)
        type(TileStore), intent(inout) :: store
        integer, intent(in) :: idx
        real(kind=8), dimension(128), intent(in) :: embedding
        
        ! Whole-array assignment — no loop needed
        store%embeddings(idx, :) = embedding(:)
        
    end subroutine set_embedding
    
    ! -----------------------------------------------------------------------------
    ! Function: distance
    ! Intent(in): vec1, vec2 — input only, not modified
    ! Purpose: Compute Euclidean distance between two embedding vectors
    ! -----------------------------------------------------------------------------
    
    pure function distance(vec1, vec2) result(dist)
        real(kind=8), dimension(128), intent(in) :: vec1, vec2
        real(kind=8) :: dist
        
        ! Vector subtraction and multiplication — Fortran handles arrays
        real(kind=8), dimension(128) :: diff
        diff = vec1 - vec2  ! Whole-array operation
        dist = sqrt(sum(diff * diff))  ! Sum of squares, then sqrt
        
    end function distance
    
    ! -----------------------------------------------------------------------------
    ! Subroutine: room_reset
    ! Intent(inout): store — modified to reset room
    ! Purpose: Clear all tiles in a room (O(1) operation)
    ! -----------------------------------------------------------------------------
    
    subroutine room_reset(store, room_id)
        type(TileStore), intent(inout) :: store
        integer(kind=8), intent(in) :: room_id
        
        ! Vector operation: find all tiles with this room_id and deactivate
        where (store%headers%room_id == room_id)
            store%active_flags = 0
        end where
        
        ! Note: No individual deallocation. Bulk reset via where clause.
        
    end subroutine room_reset
    
    ! =============================================================================
    ! PERSISTENT STATE — The save attribute
    ! =============================================================================
    
    ! Module-level persistent variables
    ! These survive across subroutine calls without heap allocation
    
    subroutine init_global_store(global_store)
        type(TileStore), intent(out) :: global_store
        
        ! Initialize all to zero — done once at startup
        global_store%headers%tile_id = 0
        global_store%headers%room_id = 0
        global_store%headers%sequence = 0
        global_store%headers%flags = 0
        global_store%headers%payload_size = 0
        global_store%embeddings = 0.0d0
        global_store%active_flags = 0
        
    end subroutine init_global_store
    
    ! =============================================================================
    ! RING BUFFER IMPLEMENTATION
    ! =============================================================================
    
    type :: RingBuffer
        type(TileHeader), dimension(MAX_TILES) :: buffer
        integer :: write_idx
        integer :: read_idx
    contains
        procedure :: push => ringbuffer_push
        procedure :: pop => ringbuffer_pop
        procedure :: is_empty => ringbuffer_is_empty
        procedure :: is_full => ringbuffer_is_full
    end type RingBuffer
    
contains

    subroutine ringbuffer_push(self, header, success)
        class(RingBuffer), intent(inout) :: self
        type(TileHeader), intent(in) :: header
        logical, intent(out) :: success
        
        integer :: next_write
        
        next_write = mod(self%write_idx + 1, MAX_TILES)
        
        if (next_write == self%read_idx) then
            success = .false.  ! Buffer full
            return
        end if
        
        self%buffer(self%write_idx + 1) = header  ! Fortran arrays are 1-indexed
        self%write_idx = next_write
        success = .true.
        
    end subroutine ringbuffer_push
    
    subroutine ringbuffer_pop(self, header, success)
        class(RingBuffer), intent(inout) :: self
        type(TileHeader), intent(out) :: header
        logical, intent(out) :: success
        
        if (self%read_idx == self%write_idx) then
            success = .false.  ! Buffer empty
            return
        end if
        
        self%read_idx = mod(self%read_idx + 1, MAX_TILES)
        header = self%buffer(self%read_idx + 1)
        success = .true.
        
    end subroutine ringbuffer_pop
    
    pure function ringbuffer_is_empty(self) result(empty)
        class(RingBuffer), intent(in) :: self
        logical :: empty
        empty = (self%read_idx == self%write_idx)
    end function ringbuffer_is_empty
    
    pure function ringbuffer_is_full(self) result(full)
        class(RingBuffer), intent(in) :: self
        logical :: full
        full = (mod(self%write_idx + 1, MAX_TILES) == self%read_idx)
    end function ringbuffer_is_full

end module tile_memory


! =============================================================================
! MAIN PROGRAM — Example usage
! =============================================================================

program tile_example
    use tile_memory
    implicit none
    
    type(TileStore) :: store
    type(RingBuffer) :: tile_queue
    type(TileHeader) :: header
    real(kind=8), dimension(128) :: embedding
    integer :: idx, err
    logical :: success
    integer :: i
    
    ! Initialize
    call init_global_store(store)
    tile_queue%write_idx = 0
    tile_queue%read_idx = 0
    
    ! Create 5 tiles
    do i = 1, 5
        call create_tile(store, int(i, 8), int(1, 8), int(i, 8), idx, err)
        if (err == 0) then
            print '(A,I2,A)', 'Created tile ', i, ' at index ', idx
            
            ! Set embedding (vector assignment)
            embedding = real(i, 8)  ! Broadcast scalar to vector
            call set_embedding(store, idx, embedding)
        end if
    end do
    
    ! Push some tiles to the queue
    do i = 1, 3
        header%tile_id = int(i, 8)
        header%room_id = int(1, 8)
        header%sequence = int(i, 8)
        header%flags = 0
        header%payload_size = 128 * 8  ! 128 float64s
        
        call tile_queue%push(header, success)
        if (success) then
            print '(A,I2)', 'Enqueued tile ', i
        end if
    end do
    
    ! Pop and process tiles
    print '(A)', 'Processing tile queue:'
    do while (.not. tile_queue%is_empty())
        call tile_queue%pop(header, success)
        if (success) then
            print '(A,I2,A,I2)', '  Tile ID=', header%tile_id, &
                                  ' Room=', header%room_id
        end if
    end do
    
    ! Compute distance between two tiles
    if (store%active_flags(1) == 1 .and. store%active_flags(2) == 1) then
        print '(A,F10.4)', 'Distance between tile 1 and 2: ', &
            distance(store%embeddings(1, :), store%embeddings(2, :))
    end if
    
    ! Reset room (clear all tiles from room 1)
    call room_reset(store, 1)
    print '(A)', 'Room 1 reset'
    
end program tile_example