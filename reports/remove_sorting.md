# Refactor: Remove Sorted Order from Slot Manager

## Summary

Removed `m_sorted_order`, `sort_buffers()`, `compact()` and `is_sorted()` from `ndtree`.
Patch data buffers are now never moved during AMR reshape. 


## Benchmark Results (Release build, depth=5, 1024 patches)

```
=== AMR reshape benchmark ===
patch_flat_size = 196  |  repetitions = 50

[fragment      ] depth=2  patches_in=    16  mean=  123.41 us  min=  108.18 us  max=  218.25 us
[recombine     ] depth=2  patches_in=    64  mean=   43.58 us  min=   37.92 us  max=  110.92 us
[full_amr_cycle] depth=2  patches=    16  mean=    0.37 us  min=    0.35 us  max=    1.10 us

[fragment      ] depth=3  patches_in=    64  mean=  403.84 us  min=  350.06 us  max=  966.27 us
[recombine     ] depth=3  patches_in=   256  mean=  153.29 us  min=  121.45 us  max= 1249.69 us
[full_amr_cycle] depth=3  patches=    64  mean=    0.81 us  min=    0.78 us  max=    1.81 us

[fragment      ] depth=4  patches_in=   256  mean=  565.90 us  min=  448.72 us  max= 2332.34 us
[recombine     ] depth=4  patches_in=  1024  mean=  650.09 us  min=  576.39 us  max=  997.36 us
[full_amr_cycle] depth=4  patches=   256  mean=    5.37 us  min=    3.58 us  max=   16.26 us

[fragment      ] depth=5  patches_in=  1024  mean= 2534.96 us  min= 2202.48 us  max= 6820.17 us
[recombine     ] depth=5  patches_in=  4096  mean= 4213.52 us  min= 3944.02 us  max= 5953.59 us
[full_amr_cycle] depth=5  patches=  1024  mean=   19.05 us  min=   15.78 us  max=   64.20 us
```

## Comparison vs Old Tree

| operation | depth=5 old | depth=5 new | speedup |
|-----------|-------------|-------------|---------|
| `fragment` | 6459 µs | 2535 µs | **2.5x** |
| `recombine` | 7650 µs | 4214 µs | **1.8x** |
| `full_amr_cycle` | 71 µs | 19 µs | **3.7x** |

