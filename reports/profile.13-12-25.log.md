# Performance Analysis

## Initial Tree-Solver integration

### Notes

iTLB misses are way too high.This could potentially be due to the high amount of
template generated code. Further analysis is required to troubleshoot the issue.

High LLC misses must be analyzed further too. Could be caused by issues in th
memory layout and access.

### Data

2D
10x10 patches
halo_width=2
t_end=400
refine_period=5
intel i7-9750h

perf stat -ddd ./bin/Release/benchmark_fvm_solver_integration

    11,586,929,309      task-clock:u                     #    0.997 CPUs utilized
                 0      context-switches:u               #    0.000 /sec
                 0      cpu-migrations:u                 #    0.000 /sec
             1,702      page-faults:u                    #  146.890 /sec
    65,355,739,487      instructions:u                   #    1.73  insn per cycle              (38.46%)
    37,746,549,477      cycles:u                         #    3.258 GHz                         (46.16%)
     2,949,986,325      branches:u                       #  254.596 M/sec                       (46.16%)
        10,816,679      branch-misses:u                  #    0.37% of all branches             (46.16%)
    15,694,067,401      L1-dcache-loads:u                #    1.354 G/sec                       (46.16%)
       729,617,379      L1-dcache-load-misses:u          #    4.65% of all L1-dcache accesses   (46.17%)
       167,481,023      LLC-loads:u                      #   14.454 M/sec                       (30.77%)
        41,167,486      LLC-load-misses:u                #   24.58% of all LL-cache accesses    (30.77%)
         2,302,454      L1-icache-load-misses:u                                                 (30.77%)
    15,734,516,209      dTLB-loads:u                     #    1.358 G/sec                       (30.76%)
           134,828      dTLB-load-misses:u               #    0.00% of all dTLB cache accesses  (30.76%)
            75,017      iTLB-loads:u                     #    6.474 K/sec                       (30.76%)
            39,148      iTLB-load-misses:u               #   52.19% of all iTLB cache accesses  (30.76%)


Callgraph analysis:

Incl Self Function\\
83.24 82.30 amr_solver<>::time_step_2d(double)\\
2.21 1.75 void amr::ndt::utils::patches::detail::halo_apply_unroll_impl<>(amr::ndt::tree::ndtree<>&, std::remove_cvref_t::linear_index_t)\\
1.45 0.13 amr::ndt::tree::ndtree<>::sort_buffers()\\
