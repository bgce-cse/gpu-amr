#ifndef INCLUDED_RANDOM_DISTRIBUTIONS
#define INCLUDED_RANDOM_DISTRIBUTIONS

#include "error_handling.hpp"
#include "utility_concepts.hpp"
#include <cmath>
#include <concepts>
#include <random>
#include <type_traits>

namespace utility::random_distributions
{

enum struct DistributionCategory
{
    Uniform,
    Normal,
    Exponential,
    Gamma,
    Geometric,
    Bernoulli,
    Binomial,
    StudentsT,
    Discrete,
    PicewiseLinear,
};

enum struct DiscretizationPolicy
{
    _none,
    Floor,
    Ceil,
    Trunc,
    Round,
};

namespace impl
{

template <utility::concepts::Arithmetic T, DistributionCategory Distribution>
struct distribution;

template <utility::concepts::Arithmetic T>
struct distribution<T, DistributionCategory::Uniform>
{
    using distribution_t = std::conditional_t<
        std::is_floating_point_v<T>,
        std::uniform_real_distribution<T>,
        std::uniform_int_distribution<T>>;
    using result_type = typename distribution_t::result_type;
    using param_type  = typename distribution_t::param_type;
};

template <utility::concepts::Arithmetic T>
struct distribution<T, DistributionCategory::Normal>
{
    using distribution_t = std::normal_distribution<std::common_type_t<T, float>>;
    using result_type    = typename distribution_t::result_type;
    using param_type     = typename distribution_t::param_type;
};

template <utility::concepts::Arithmetic T>
struct distribution<T, DistributionCategory::Exponential>
{
    using distribution_t = std::exponential_distribution<std::common_type_t<T, float>>;
    using result_type    = typename distribution_t::result_type;
    using param_type     = typename distribution_t::param_type;
};

template <utility::concepts::Arithmetic T>
struct distribution<T, DistributionCategory::Gamma>
{
    using distribution_t = std::gamma_distribution<std::common_type_t<T, float>>;
    using result_type    = typename distribution_t::result_type;
    using param_type     = typename distribution_t::param_type;
};

template <std::integral T>
struct distribution<T, DistributionCategory::Geometric>
{
    using distribution_t = std::geometric_distribution<std::common_type_t<T, int>>;
    using result_type    = typename distribution_t::result_type;
    using param_type     = typename distribution_t::param_type;
};

template <>
struct distribution<bool, DistributionCategory::Bernoulli>
{
    using distribution_t = std::bernoulli_distribution;
    using result_type    = typename distribution_t::result_type;
    using param_type     = typename distribution_t::param_type;
};

template <std::integral T>
struct distribution<T, DistributionCategory::Binomial>
{
    using distribution_t = std::binomial_distribution<std::common_type_t<T, int>>;
    using result_type    = typename distribution_t::result_type;
    using param_type     = typename distribution_t::param_type;
};

template <utility::concepts::Arithmetic T>
struct distribution<T, DistributionCategory::StudentsT>
{
    using distribution_t = std::student_t_distribution<std::common_type_t<T, float>>;
    using result_type    = typename distribution_t::result_type;
    using param_type     = typename distribution_t::param_type;
};

template <std::integral T>
struct distribution<T, DistributionCategory::Discrete>
{
    using distribution_t = std::discrete_distribution<std::common_type_t<T, int>>;
    using result_type    = typename distribution_t::result_type;
    using param_type     = typename distribution_t::param_type;
};

template <utility::concepts::Arithmetic T>
struct distribution<T, DistributionCategory::PicewiseLinear>
{
    using distribution_t =
        std::piecewise_linear_distribution<std::common_type_t<T, float>>;
    using result_type = typename distribution_t::result_type;
    using param_type  = typename distribution_t::param_type;
};


} // namespace impl

template <
    utility::concepts::Arithmetic T,
    DistributionCategory          Distribution,
    DiscretizationPolicy          Discretization_Policy = DiscretizationPolicy::_none>
    requires(
        (std::is_floating_point_v<T> and
         Discretization_Policy == DiscretizationPolicy::_none) or
        (std::is_integral_v<T> and
         (std::is_integral_v<typename impl::distribution<T, Distribution>::result_type> or
          Discretization_Policy != DiscretizationPolicy::_none))
    )
class random_distribution
{
public:
    using random_engine_t                       = std::mt19937_64;
    static constexpr auto distribution_category = Distribution;
    static constexpr auto discretizaton_policy  = Discretization_Policy;
    using distribution_impl_type = impl::distribution<T, distribution_category>;
    using distribution_t         = typename distribution_impl_type::distribution_t;
    using impl_result_type       = typename distribution_t::result_type;
    using result_type            = T;
    using param_type             = typename distribution_impl_type::param_type;

    static_assert(std::is_convertible_v<impl_result_type, result_type>);

public:
    random_distribution(
        param_type   params,
        unsigned int seed = std::random_device{}()
    ) noexcept :
        random_engine_(seed),
        distribution_(params)
    {
    }

    [[nodiscard]]
    inline auto operator()() noexcept -> result_type
    {
        const auto value = distribution_(random_engine_);
        if constexpr (std::is_same_v<result_type, impl_result_type>)
        {
            return value;
        }
        else if constexpr (std::is_integral_v<result_type> and
                           std::is_floating_point_v<impl_result_type>)
        {
            if constexpr (discretizaton_policy == DiscretizationPolicy::Floor)
            {
                return static_cast<result_type>(std::floor(value));
            }
            else if constexpr (discretizaton_policy == DiscretizationPolicy::Ceil)
            {
                return static_cast<result_type>(std::ceil(value));
            }
            else if constexpr (discretizaton_policy == DiscretizationPolicy::Trunc)
            {
                return static_cast<result_type>(std::trunc(value));
            }
            else if constexpr (discretizaton_policy == DiscretizationPolicy::Round)
            {
                return static_cast<result_type>(std::round(value));
            }
            else
            {
                utility::error_handling::assert_unreachable();
            }
        }
        else if constexpr (std::is_floating_point_v<result_type> and
                           std::is_integral_v<impl_result_type>)
        {
            return static_cast<result_type>(value);
        }
        else if constexpr (std::is_convertible_v<impl_result_type, result_type>)
        {
            return static_cast<result_type>(value);
        }
        else
        {
            utility::error_handling::assert_unreachable();
        }
    }

private:
    random_engine_t random_engine_;
    distribution_t  distribution_;
};

} // namespace utility::random_distributions

#endif // INCLUDED_RANDOM_DISTRIBUTIONS
