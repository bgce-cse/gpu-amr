#pragma once
#include <vector>
#include <cuda_runtime.h>

struct DeviceSoA2D
{
    double* rho;
    double* rhou;
    double* rhov;
    double* E;
};

// allocate gpu memory
inline DeviceSoA2D cuda_alloc_soa(std::size_t n)
{
    DeviceSoA2D d;

    cudaMalloc(&d.rho,  n*sizeof(double));
    cudaMalloc(&d.rhou, n*sizeof(double));
    cudaMalloc(&d.rhov, n*sizeof(double));
    cudaMalloc(&d.E,    n*sizeof(double));

    return d;
}

// copy from cpu -> gpu
inline void cuda_h2d(
    const DeviceSoA2D& d,
    const std::vector<double>& rho,
    const std::vector<double>& rhou,
    const std::vector<double>& rhov,
    const std::vector<double>& E)
{
    cudaMemcpy(d.rho,  rho.data(),  rho.size()*sizeof(double),  cudaMemcpyHostToDevice);
    cudaMemcpy(d.rhou, rhou.data(), rhou.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d.rhov, rhov.data(), rhov.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d.E,    E.data(),    E.size()*sizeof(double),    cudaMemcpyHostToDevice);
}


// copy from gpu -> cpu
inline void cuda_d2h(
    const DeviceSoA2D& d,
    std::vector<double>& rho,
    std::vector<double>& rhou,
    std::vector<double>& rhov,
    std::vector<double>& E)
{
    cudaMemcpy(rho.data(),  d.rho,  rho.size()*sizeof(double),  cudaMemcpyDeviceToHost);
    cudaMemcpy(rhou.data(), d.rhou, rhou.size()*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(rhov.data(), d.rhov, rhov.size()*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(E.data(),    d.E,    E.size()*sizeof(double),    cudaMemcpyDeviceToHost);
}

template<typename TreeT, typename PatchLayoutT>
auto pack_tree_state_2d(const TreeT& tree)
{
    struct Packed
    {
        std::size_t num_patches;
        std::size_t P;

        std::vector<double> rho;
        std::vector<double> rhou;
        std::vector<double> rhov;
        std::vector<double> E;
    };

    Packed out;

    constexpr std::size_t P = PatchLayoutT::flat_size();

    out.num_patches = tree.size();
    out.P = P;

    out.rho.resize(out.num_patches * P);
    out.rhou.resize(out.num_patches * P);
    out.rhov.resize(out.num_patches * P);
    out.E.resize(out.num_patches * P);

    for(std::size_t p = 0; p < out.num_patches; ++p)
    {
        auto rho_patch  = tree.template get_patch<amr::cell::Rho>(p);
        auto rhou_patch = tree.template get_patch<amr::cell::Rhou>(p);
        auto rhov_patch = tree.template get_patch<amr::cell::Rhov>(p);
        auto e_patch = tree.template get_patch<amr::cell::E2D>(p);

        std::size_t base = p * P;

        for(std::size_t i = 0; i < P; ++i)
        {
            out.rho [base+i] = rho_patch[i];
            out.rhou[base+i] = rhou_patch[i];
            out.rhov[base+i] = rhov_patch[i];
            out.E   [base+i] = e_patch[i];
        }
    }

    return out;
}

template<typename TreeT, typename PatchLayoutT>
void unpack_tree_state_2d(
    TreeT& tree,
    const std::vector<double>& rho,
    const std::vector<double>& rhou,
    const std::vector<double>& rhov,
    const std::vector<double>& E)
{
    constexpr std::size_t P = PatchLayoutT::flat_size();

    std::size_t num_patches = tree.size();

    for(std::size_t p = 0; p < num_patches; ++p)
    {
        auto rho_patch  = tree.template get_patch<amr::cell::Rho>(p);
        auto rhou_patch = tree.template get_patch<amr::cell::Rhou>(p);
        auto rhov_patch = tree.template get_patch<amr::cell::Rhov>(p);
        auto e_patch = tree.template get_patch<amr::cell::E2D>(p);

        std::size_t base = p * P;

        for(std::size_t i = 0; i < P; ++i)
        {
            rho_patch[i]  = rho [base+i];
            rhou_patch[i] = rhou[base+i];
            rhov_patch[i] = rhov[base+i];
            e_patch[i]    = E   [base+i];
        }
    }
}