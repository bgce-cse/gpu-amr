# Parallelization Plan

## Goal
Patch data (`m_data_buffers`) lives permanently on GPU and is never copied to CPU
except for visualization output. All tree metadata stays on CPU.

## Memory Layout

**CPU (`malloc`):**
- `m_index_map`, `m_linear_index_map`, `m_active_slots`, `m_free_slots`
- `m_neighbors`, `m_refine_status_buffer`, `m_slot_active`

**GPU (`cudaMalloc`):**
- `m_data_buffers` — all field data, slot-indexed, never moves

## CPU Operations (serial, integer only)
- `apply_refine_coarsen()` — reads refine status, pure morton bit ops
- `balancing()` — neighbor graph traversal on integer tree
- `fragment()` / `recombine()` metadata — alloc_slot, write_slot_metadata, enforce_symmetry
- Note: `enforce_symmetry` is potentially a bottleneck at large patch counts but non-trivial to parallelize

## GPU Kernels
- **Solver kernel** — embarrassingly parallel, one block per patch, purely local
- **Halo exchange kernel** — one block per patch, reads `d_neighbors`, no race conditions since each patch only writes its own halo
- **Interpolation kernel** — one launch per fragmentation batch, reads parent slot, writes child slots
- **Restriction kernel** — one launch per recombination batch, reads child slots, writes parent slot
- **Refinement criterion kernel** — runs on `m_data_buffers`, writes `d_refine_status` (e.g. abs rho based), result transferred back to CPU

## CPU↔GPU Transfers

**Each tree reconstruction (low frequency):**
- `d_refine_status` → CPU: result of criterion kernel, ~1 byte per patch, negligible
- `d_active_slots` → GPU: updated after reshape, ~8 bytes per patch
- `d_neighbors` → GPU: updated after reshape, ~64 bytes per patch

**Each timestep (high frequency):**
- No transfers — solver and halo exchange run entirely on GPU

## Open Questions
- Refinement criterion API: `update_refine_flags` needs a GPU variant that accepts a kernel