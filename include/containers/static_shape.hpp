#ifndef AMR_INCLUDED_STATIC_SHAPE
#define AMR_INCLUDED_STATIC_SHAPE

#include "container_concepts.hpp"
#include "utility/utility_concepts.hpp"
#include <algorithm>
#include <array>
#include <cassert>

#ifndef NDEBUG
#    define AMR_CONTAINERS_CHECKBOUNDS
#endif

namespace amr::containers
{

template <concepts::Container auto Sizes>
    requires std::integral<typename decltype(Sizes)::value_type>
class static_shape
{
public:
    // TODO: This can be dangerous, maybe hardcode a type once we know what we
    // need
    using size_type = typename decltype(Sizes)::value_type;
    using rank_t    = size_type;
    using index_t   = size_type;

private:
    static_assert(
        std::ranges::all_of(Sizes, [](auto const& e) { return e > size_type{}; })
    );

private:
    inline static constexpr auto const& s_sizes = Sizes;
    inline static constexpr rank_t      s_rank  = s_sizes.size();
    inline static constexpr size_type   s_elements =
        std::ranges::fold_left(s_sizes, size_type{ 1 }, std::multiplies{});

    static_assert(s_rank > 0);

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

    [[nodiscard]]
    constexpr static auto size(index_t const i) noexcept -> size_type
    {
        if (!std::is_constant_evaluated())
        {
            assert(i < rank());
        }
        using container_index_t =
            typename std::remove_cvref_t<decltype(s_sizes)>::size_type;
        return s_sizes[static_cast<container_index_t>(i)];
    }
};

} // namespace amr::containers

#endif // AMR_INCLUDED_STATIC_SHAPE
