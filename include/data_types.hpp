#pragma once

#include <cstddef>
#include <ostream>
#include <tuple>

template <typename T>
using value_t = typename T::type;

template <class... Ts>
struct overloads : Ts...
{
    using Ts::operator()...;
};

template <typename T, std::size_t Idx>
struct map_type_t
{
    using type = T;
    static constexpr auto index() noexcept -> std::size_t
    {
        return Idx;
    }
    T value;
};

using u_t = map_type_t<double, 0>;
using v_t = map_type_t<double, 1>;
using f_t = map_type_t<double, 2>;
using g_t = map_type_t<double, 3>;
using p_t = map_type_t<double, 4>;
using t_t = map_type_t<double, 5>;
using rhs_t = map_type_t<double, 6>;

struct cell
{
    using deconstructed_types_map_t =
        std::tuple<u_t, v_t, f_t, g_t, p_t, t_t, rhs_t>;

    cell(
        value_t<u_t> u, value_t<v_t> v, value_t<f_t> f, value_t<g_t> g,
        value_t<p_t> p, value_t<t_t> t, value_t<rhs_t> rhs
    )
    {
        std::get<u_t>(m_data).value = u;
        std::get<v_t>(m_data).value = v;
        std::get<f_t>(m_data).value = f;
        std::get<g_t>(m_data).value = g;
        std::get<p_t>(m_data).value = p;
        std::get<t_t>(m_data).value = t;
        std::get<rhs_t>(m_data).value = rhs;
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

inline auto operator<<(std::ostream& os, cell const& c) -> std::ostream&
{
    return os << "u: " << std::get<u_t>(c.data_tuple()).value
              << ", v: " << std::get<v_t>(c.data_tuple()).value
              << ", f: " << std::get<f_t>(c.data_tuple()).value
              << ", g: " << std::get<g_t>(c.data_tuple()).value
              << ", p: " << std::get<p_t>(c.data_tuple()).value
              << ", t: " << std::get<t_t>(c.data_tuple()).value;
}

struct PhysicalBC
{
    std::string type;
    double value;
};

enum struct cell_temperature_type
{
    ADIABATIC,
    DIABATIC,
};

enum struct cell_type
{
    FLUID,
    OBSTACLE,
    DEFAULT,
};
