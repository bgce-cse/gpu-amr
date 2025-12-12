#ifndef DG_HELPERS_EQUATIONS_HPP
#define DG_HELPERS_EQUATIONS_HPP

#include "advection.hpp"
#include "equation_impl.hpp"
#include "euler.hpp"
#include "generated_config.hpp"

namespace amr::equations
{
/**
 * @brief String template for equation selection (C++20)
 *
 * Allows passing a string literal as a template parameter.
 * Usage: SelectEquation<NumDOF, Order, Dim, "advection">
 */
template <
    amr::config::EquationType EQ,
    std::size_t               NumDOF,
    std::size_t               Order,
    std::size_t               Dim>
struct EquationTraits;

template <std::size_t NumDOF, std::size_t Order, std::size_t Dim>
struct EquationTraits<amr::config::EquationType::Advection, NumDOF, Order, Dim>
{
    using type = amr::equations::Advection<NumDOF, Order, Dim>;
};

template <std::size_t NumDOF, std::size_t Order, std::size_t Dim>
struct EquationTraits<amr::config::EquationType::Euler, NumDOF, Order, Dim>
{
    using type = amr::equations::Euler<Order, Dim>;
};
} // namespace amr::equations

#endif // DG_HELPERS_EQUATIONS_HPP
