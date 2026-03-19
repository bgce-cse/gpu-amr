#include "containers/static_shape.hpp"
#include <concepts>
#include <gtest/gtest.h>

namespace ContianerBuildingBlocks
{

template <std::integral auto... Ns>
struct StaticShapeWrapper
{
    static constexpr auto sizes = std::array{ Ns... };
    using test_type_t           = amr::containers::static_shape<Ns...>;
    using size_type             = typename test_type_t::size_type;
    static_assert(test_type_t::rank() == sizeof...(Ns));
    static_assert(test_type_t::sizes() == sizes);
};

using types =
    ::testing::Types<StaticShapeWrapper<1, 2, 3>, StaticShapeWrapper<4, 5>, StaticShapeWrapper<1, 9, 5, 2>>;

template <typename T>
class StaticShapeTest : public ::testing::Test
{
};

TYPED_TEST_SUITE(StaticShapeTest, types);

TYPED_TEST(StaticShapeTest, RankIsCorrect)
{
    EXPECT_EQ(TypeParam::test_type_t::rank(), TypeParam::sizes.size());
}

TYPED_TEST(StaticShapeTest, SizesAreCorrect)
{
    EXPECT_EQ(TypeParam::test_type_t::sizes(), TypeParam::sizes);
    for (auto i = typename TypeParam::size_type{}; i != TypeParam::test_type_t::rank();
         ++i)
    {
        EXPECT_EQ(TypeParam::test_type_t::size(i), TypeParam::sizes[i])
            << "Mismatch at index " << i;
    }
}

} // namespace
