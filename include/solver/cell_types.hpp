#ifndef CELL_TYPES_HPP
#define CELL_TYPES_HPP

#include <cstddef>
#include <iostream>
#include <string_view>
#include <tuple>

namespace amr::cell
{

struct Rho
{
    using type = double;

    static constexpr auto index() noexcept -> std::size_t
    {
        return 0;
    }

    static constexpr auto name() noexcept -> std::string_view
    {
        return "Rho";
    }

    type value;
};

struct Rhou
{
    using type = double;

    static constexpr auto index() noexcept -> std::size_t
    {
        return 1;
    }

    static constexpr auto name() noexcept -> std::string_view
    {
        return "Rhou";
    }

    type value;
};

struct Rhov
{
    using type = double;

    static constexpr auto index() noexcept -> std::size_t
    {
        return 2;
    }

    static constexpr auto name() noexcept -> std::string_view
    {
        return "Rhov";
    }

    type value;
};

struct Rhow
{
    using type = double;

    static constexpr auto index() noexcept -> std::size_t
    {
        return 3;
    }

    static constexpr auto name() noexcept -> std::string_view
    {
        return "Rhow";
    }

    type value;
};

struct E2D
{
    using type = double;

    static constexpr auto index() noexcept -> std::size_t
    {
        return 3;
    }

    static constexpr auto name() noexcept -> std::string_view
    {
        return "E2D";
    }

    type value;
};

struct E3D
{
    using type = double;

    static constexpr auto index() noexcept -> std::size_t
    {
        return 4;
    }

    static constexpr auto name() noexcept -> std::string_view
    {
        return "E3D";
    }

    type value;
};

/**
 * @brief 2D Euler cell containing [rho, rho*u, rho*v, E2D]
 *
 * Conservative variables for 2D compressible Euler equations:
 * - Rho:  Density
 * - Rhou: X-momentum (rho * u)
 * - Rhov: Y-momentum (rho * v)
 * - E2D:  Total energy
 */
struct EulerCell2D
{
    using deconstructed_types_map_t = std::tuple<Rho, Rhou, Rhov, E2D>;

    EulerCell2D(double rho, double rhou, double rhov, double e2d)
    {
        std::get<Rho>(m_data).value  = rho;
        std::get<Rhou>(m_data).value = rhou;
        std::get<Rhov>(m_data).value = rhov;
        std::get<E2D>(m_data).value  = e2d;
    }

    auto data_tuple() -> auto&
    {
        return m_data;
    }

    auto data_tuple() const -> auto const&
    {
        return m_data;
    }

    deconstructed_types_map_t m_data;
};

auto operator<<(std::ostream& os, EulerCell2D const& c) -> std::ostream&
{
    return os << "Rho: " << std::get<Rho>(c.data_tuple()).value
              << ", Rhou: " << std::get<Rhou>(c.data_tuple()).value
              << ", Rhov: " << std::get<Rhov>(c.data_tuple()).value
              << ", E2D: " << std::get<E2D>(c.data_tuple()).value;
}

/**
 * @brief 3D Euler cell containing [rho, rho*u, rho*v, rho*w, E3D]
 *
 * Conservative variables for 3D compressible Euler equations:
 * - Rho:  Density
 * - Rhou: X-momentum (rho * u)
 * - Rhov: Y-momentum (rho * v)
 * - Rhow: Z-momentum (rho * w)
 * - E3D:  Total energy
 */
struct EulerCell3D
{
    using deconstructed_types_map_t = std::tuple<Rho, Rhou, Rhov, Rhow, E3D>;

    EulerCell3D(double rho, double rhou, double rhov, double rhow, double e3d)
    {
        std::get<Rho>(m_data).value  = rho;
        std::get<Rhou>(m_data).value = rhou;
        std::get<Rhov>(m_data).value = rhov;
        std::get<Rhow>(m_data).value = rhow;
        std::get<E3D>(m_data).value  = e3d;
    }

    auto data_tuple() -> auto&
    {
        return m_data;
    }

    auto data_tuple() const -> auto const&
    {
        return m_data;
    }

    deconstructed_types_map_t m_data;
};

auto operator<<(std::ostream& os, EulerCell3D const& c) -> std::ostream&
{
    return os << "Rho: " << std::get<Rho>(c.data_tuple()).value
              << ", Rhou: " << std::get<Rhou>(c.data_tuple()).value
              << ", Rhov: " << std::get<Rhov>(c.data_tuple()).value
              << ", Rhow: " << std::get<Rhow>(c.data_tuple()).value
              << ", E3D: " << std::get<E3D>(c.data_tuple()).value;
}

struct AdvectionCell {
        static constexpr std::size_t count = 1;
        enum Vars { Scalar = 0 };
    };

} // namespace amr::cell

#endif // CELL_TYPES_HPP
