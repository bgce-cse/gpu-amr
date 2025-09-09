#ifndef CELL_TYPES_HPP
#define CELL_TYPES_HPP

#include <cstddef>
#include <tuple>
#include <iostream>

namespace amr::cell
{

struct Rho {
    using type = double;
    static constexpr auto index() noexcept -> std::size_t { return 0; }
    type value;
};

struct Rhou {
    using type = double;
    static constexpr auto index() noexcept -> std::size_t { return 1; }
    type value;
};

struct Rhov {
    using type = double;
    static constexpr auto index() noexcept -> std::size_t { return 2; }
    type value;
};

struct E {
    using type = double;
    static constexpr auto index() noexcept -> std::size_t { return 3; }
    type value;
};

struct EulerCell
{
    using deconstructed_types_map_t = std::tuple<Rho, Rhou, Rhov, E>;

    EulerCell(double rho, double rhou, double rhov, double e) {
        std::get<Rho>(m_data).value = rho;
        std::get<Rhou>(m_data).value = rhou;
        std::get<Rhov>(m_data).value = rhov;
        std::get<E>(m_data).value = e;
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

auto operator<<(std::ostream& os, EulerCell const& c) -> std::ostream&
{
    return os << "Rho: " << std::get<Rho>(c.data_tuple()).value
              << ", Rhou: " << std::get<Rhou>(c.data_tuple()).value
              << ", Rhov: " << std::get<Rhov>(c.data_tuple()).value
              << ", E: " << std::get<E>(c.data_tuple()).value;
}

} // namespace amr::cell

#endif // CELL_TYPES_HPP