#pragma once
#include <cstddef>

void launch_patch_update(
    double* rho,
    double* rhou,
    double* rhov,
    double* E,
    int num_patches,
    double dt,
    double gamma
);