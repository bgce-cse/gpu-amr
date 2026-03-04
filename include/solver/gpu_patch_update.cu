#include "gpu_patch_update.cuh"
#include <cstdio>
#include <cuda_runtime.h>

constexpr int PATCH_SIZE = 196;
constexpr int NXH = 14;
constexpr int NYH = 14;

__global__
void patch_update_kernel(
    double* rho,
    double* rhou,
    double* rhov,
    double* E,
    int num_patches,
    double dt,
    double gamma)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("GPU kernel running\n");
    }
    int patch = blockIdx.x;
    int t     = threadIdx.x;

    if (patch >= num_patches || t >= PATCH_SIZE)
        return;

    int ix = t % NXH;
    int iy = t / NXH;

    // skip halo
    if (ix < 2 || ix >= NXH-2 || iy < 2 || iy >= NYH-2)
        return;

    int base = patch * PATCH_SIZE;

    // example placeholder update
    rho[base + t] += 0.0;
}

void launch_patch_update(
    double* rho,
    double* rhou,
    double* rhov,
    double* E,
    int num_patches,
    double dt,
    double gamma)
{
    dim3 blocks(num_patches);
    dim3 threads(256);

    patch_update_kernel<<<blocks, threads>>>(
        rho, rhou, rhov, E,
        num_patches,
        dt,
        gamma
    );
}