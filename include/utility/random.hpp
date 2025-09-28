#ifndef INCLUDED_RANDOM_NUMBER_GENERATOR
#define INCLUDED_RANDOM_NUMBER_GENERATOR

#include "utility_concepts.hpp"
#include <concepts>
#include <type_traits>
#ifndef NOMINMAX
#    define NOMINMAX
#endif

#include <cassert>
#include <random>

#ifdef max
#    undef max
#endif

namespace utility::random
{

template <utility::concepts::Arithmetic T, typename Enables = void>
class random;

template <utility::concepts::Arithmetic T>
class random<T, std::enable_if_t<std::is_integral_v<T>>>
{
public:
    using value_type = T;

private:
    using float_value_type = std::common_type_t<float, value_type>;

public:
    random(unsigned int seed = std::random_device{}()) noexcept
    {
        seed_engine(seed);
    }

    /// @brief Generates a number in the range [min, max] for integral types and [min,
    /// max]
    /// @tparam T Arithmetic type
    /// @param min Inclusive lower bound
    /// @param max Inclusive upper bound
    /// @return Number of type T uniformly distributed in the range
    /// [min, max]
    [[nodiscard]]
    inline auto randrange(value_type min, value_type max) noexcept -> value_type
    {
        using distribution_t = std::uniform_int_distribution<value_type>;
        assert(min <= max);
        distribution_t uniform_dist(min, max);
        return uniform_dist(random_engine_);
    }

    inline auto seed_engine(unsigned int seed) noexcept -> void
    {
        random_engine_.seed(seed);
    }

private:
    std::mt19937_64 random_engine_;
};

template <utility::concepts::Arithmetic T>
class random<T, std::enable_if_t<std::is_floating_point_v<T>>>
{
public:
    using value_type = T;

private:
    using float_value_type = std::common_type_t<float, value_type>;

public:
    random(unsigned int seed = std::random_device{}()) noexcept
    {
        seed_engine(seed);
    }

    [[nodiscard]]
    inline auto randfloat() noexcept -> value_type
        requires std::is_floating_point_v<value_type>
    {
        return uniform_real_(random_engine_);
    }

    /// @brief Generates a number in the range [min, max] for integral types and [min,
    /// max) for floating-point types
    /// @tparam T Arithmetic type
    /// @param min Inclusive lower bound
    /// @param max Inclusive upper bound
    /// @return Number of type T uniformly distributed in the range
    /// [min, max] for integral types and [min, max) for floating-point types
    [[nodiscard]]
    inline auto randrange(value_type min, value_type max) noexcept -> value_type
    {
        using distribution_t = std::uniform_real_distribution<value_type>;
        assert(min <= max);
        distribution_t uniform_dist(min, max);
        return uniform_dist(random_engine_);
    }

    [[nodiscard]]
    inline auto randnormal(value_type avg, value_type stddev) noexcept -> value_type
        requires std::is_floating_point_v<value_type>
    {
        std::normal_distribution<value_type> n(avg, stddev);
        return n(random_engine_);
    }

    [[nodiscard]]
    inline auto randnormal() noexcept -> value_type
        requires std::is_floating_point_v<value_type>
    {
        return default_normal_(random_engine_);
    }

    inline auto seed_engine(unsigned int seed) noexcept -> void
    {
        random_engine_.seed(seed);
    }

private:
    std::mt19937_64                                  random_engine_;
    std::uniform_real_distribution<float_value_type> uniform_real_ =
        std::uniform_real_distribution<float_value_type>(
            float_value_type{ 0 },
            float_value_type{ 1 }
        );
    std::normal_distribution<float_value_type> default_normal_ =
        std::normal_distribution<float_value_type>(
            float_value_type{ 0 },
            float_value_type{ 1 }
        );
};

struct srandom
{
private:
    template <utility::concepts::Arithmetic T>
    [[nodiscard]]
    inline static auto static_instance() noexcept -> random<T>&
    {
        static random<T> instance;
        return instance;
    }

public:
    template <std::floating_point F>
    inline static auto seed(unsigned int seed_ = std::random_device{}()) noexcept -> void
    {
        static_instance<F>().seed_engine(seed_);
    }

    template <std::floating_point F>
    [[nodiscard]]
    inline static auto randfloat() noexcept -> F
    {
        return static_instance<F>().randfloat();
    }

    /// @brief Generates a number in the range [min, max] for integral types and [min,
    /// max) for floating-point types
    /// @tparam T Arithmetic type
    /// @param min Inclusive lower bound
    /// @param max Inclusive/Exclusive upper bound
    /// @return Number of type T uniformly distributed in the range
    /// [min, max] for integral types and [min, max) for floating-point types
    template <utility::concepts::Arithmetic T>
    [[nodiscard]]
    inline static auto randrange(T min, T max) noexcept -> T
    {
        return static_instance<T>().randrange(min, max);
    }

    template <std::floating_point F>
    [[nodiscard]]
    inline static auto randnormal(F avg, F stddev) noexcept -> F
    {
        return static_instance<F>().randnormal(avg, stddev);
    }

    template <std::floating_point F>
    [[nodiscard]]
    inline static auto randnormal() noexcept -> F
    {
        return static_instance<F>().randnormal();
    }
};

} // namespace utility::random

#endif // INCLUDED_RANDOM_NUMBER_GENERATOR
