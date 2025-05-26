#ifndef AMR_INCLUDED_AMR_BUFFER_ALLOCATOR
#define AMR_INCLUDED_AMR_BUFFER_ALLOCATOR

#include "allocator_types.hpp"
#include "allocator_utils.hpp"
#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <vector>

namespace amr::allocator
{

template <
    std::integral auto          Block_Size,
    std::unsigned_integral auto Alignment =
        std::make_unsigned_t<decltype(Block_Size)>{ Block_Size }>
class buffer_allocator
{
    // TODO: look into std::align_val_t
    // TODO: Maybe worry about fundamental and extended alignments
    static_assert(Block_Size >= decltype(Block_Size){ 1 });
    static_assert(utils::is_valid_alignment(Alignment));
    static_assert(Alignment >= Block_Size);

public:
    using size_type       = decltype(Block_Size);
    using value_type      = std::byte;
    using pointer         = value_type*;
    using const_pointer   = value_type const*;
    using reference       = value_type&;
    using const_reference = value_type const&;
    using difference_type = std::ptrdiff_t;
    // TODO: Vector has an allocator also. Would that be supported in the gpu?
    using free_list_t    = std::vector<pointer>;
    using release_list_t = std::vector<std::pair<pointer, size_type>>;

private:
    static constexpr auto s_block_size           = Block_Size;
    static constexpr auto s_alignment            = Alignment;
    static constexpr auto s_block_alloc_size     = s_alignment;
    static constexpr auto s_initial_release_size = size_type{ 1 };

    // TODO: Add growth factor / grow chunk size configuration

public:
    explicit buffer_allocator(
        size_type const n,
        size_type const min_underlying_buffer_size
    ) noexcept
        : m_free_list{}
        , m_release_list{}
        , m_min_underlying_buffer_size{ min_underlying_buffer_size }
    {
        assert(n >= size_type{ 1 });
        assert(min_underlying_buffer_size >= n);
        assert(min_underlying_buffer_size % s_block_alloc_size == 0);
        m_free_list.reserve(n);
        m_release_list.reserve(s_initial_release_size);
        allocate_buffer(n);
    }

    explicit buffer_allocator(size_type const n) noexcept
        : buffer_allocator(n, n)
    {
    }

    buffer_allocator(buffer_allocator const&) noexcept                    = delete;
    buffer_allocator(buffer_allocator&&) noexcept                         = delete;
    auto operator=(buffer_allocator const&) noexcept -> buffer_allocator& = delete;
    auto operator=(buffer_allocator&&) noexcept -> buffer_allocator&      = delete;

    ~buffer_allocator() noexcept
    {
        for (auto&& [buffer, size] : m_release_list)
        {
#if ALLOCATOR_DEBUG_INITIALIZE
            utils::fill_buffer(buffer, size, ALLOCATOR_DEBUG_RELEASE_VALUE);
#endif
            std::free(buffer, size);
        }
    }

    // TODO: Implement hint
    // TODO: Backup allocator?
    auto allocate_buffer(size_type const n, [[maybe_unused]] pointer const hint = nullptr)
        -> pointer
    {
        assert(n > 0);
        static_assert(s_block_alloc_size % s_alignment == 0);
        const auto underlying_buffer_size = n * s_block_alloc_size;
        const auto buffer =
            static_cast<pointer>(std::aligned_alloc(s_alignment, underlying_buffer_size));
        m_release_list.emplace_back({ buffer, underlying_buffer_size });
#if ALLOCATOR_DEBUG_INITIALIZE
        utils::fill_buffer(buffer, underlying_buffer_size, ALLOCATOR_DEBUG_INIT_VALUE);
#endif
        [[assume(n > 0)]];
        for (auto p = buffer; p != p + underlying_buffer_size; p += s_block_alloc_size)
        {
            assert(std::less_equal(p, &buffer[underlying_buffer_size]));
            assert(
                std::less_equal(p + s_block_alloc_size, &buffer[underlying_buffer_size])
            );
            m_free_list.push_back(p);
        }
        return buffer;
    }

    [[nodiscard]]
    auto allocate_one() -> pointer
    {
        if (available() == size_type{}) [[unlikely]]
        {
            allocate_buffer(m_min_underlying_buffer_size);
        }
        const auto p = *std::end(m_free_list);
#if ALLOCATOR_DEBUG_INITIALIZE
        utils::fill_buffer(p, s_block_alloc_size, ALLOCATOR_DEBUG_ALLOC_VALUE);
#endif
        m_free_list.pop_back();
        return p;
    }

    [[nodiscard]]
    auto allocate_n(size_type const n) -> pointer
    {
        if (available() < n)
        {
            // TODO: This block is to be returned
            // Maybe unless it happens to be contiguous to some empty space?
            // Not worth it probably
            allocate_buffer(m_min_underlying_buffer_size);
        }
        // TODO: Find consecutive block of memory
        // TODO: Remove these entries from the free list
        // TODO: Debug initialize memory
    }

    inline auto deallocate_one(pointer p) -> void
    {
        assert(!std::ranges::contains(m_free_list, p));
        m_free_list.push_back(p);
#if ALLOCATOR_DEBUG_INITIALIZE
        utils::fill_buffer(p, s_block_alloc_size, ALLOCATOR_DEBUG_RELEASE_VALUE);
#endif
    }

    inline auto deallocate_n(pointer p, size_type const n) noexcept -> void
    {
        for (auto i = size_type{}; i != n; ++i)
        {
            deallocate_one(p);
            p += s_block_alloc_size;
        }
    }

    [[nodiscard]]
    inline auto available() const noexcept -> size_type
    {
        return std::size(m_free_list);
    }

private:
    [[nodiscard]]
    static constexpr auto block_alloc_size() noexcept -> size_type
    {
        return s_block_alloc_size;
    }

private:
    free_list_t    m_free_list{};
    release_list_t m_release_list{};
    // const because I dont think we need assignemnt for this
    const size_type m_min_underlying_buffer_size{};
};

} // namespace amr::allocator

#endif // AMR_INCLUDED_AMR_BUFFER_ALLOCATOR
