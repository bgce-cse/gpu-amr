#ifndef AMR_INCLUDED_STATIC_SHAPE
#define AMR_INCLUDED_STATIC_SHAPE

#include "utility/compile_time_utility.hpp"
#include "utility/utility_concepts.hpp"
#include <array>
#include <cassert>
#include <numeric>

#ifndef NDEBUG
#    define AMR_CONTAINERS_CHECKBOUNDS
#endif

namespace amr::containers
{

template <std::integral auto N, std::integral auto... Ns>
    requires utility::concepts::are_same<decltype(N), decltype(Ns)...> && (N > 0) &&
             ((Ns > 0) && ...)
class static_shape
{
public:
    // TODO: This can be dangerous, maybe hardcode a type once we know what we
    // need
    using size_type = std::common_type_t<decltype(N), decltype(Ns)...>;
    using rank_t    = size_type;

private:
    inline static constexpr size_type                     s_elements = (N * ... * Ns);
    inline static constexpr rank_t                        s_rank     = sizeof...(Ns) + 1;
    inline static constexpr std::array<size_type, s_rank> s_sizes    = { N, Ns... };

public:
    [[nodiscard]]
    static constexpr auto rank() noexcept -> rank_t
    {
        return s_rank;
    }

    [[nodiscard]]
    static constexpr auto elements() noexcept -> size_type
    {
        return s_elements;
    }

    [[nodiscard]]
    static constexpr auto sizes() noexcept -> auto const&
    {
        return s_sizes;
    }
};

} // namespace amr::containers

#endif // AMR_INCLUDED_STATIC_SHAPE
