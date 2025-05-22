#ifndef INCLUDED_LIFETIME_UTILITY
#define INCLUDED_LIFETIME_UTILITY

#include <iostream>

#ifndef DEBUG_OSTREAM
#define DEBUG_OSTREAM std::cout
#endif

namespace utility::lifetimes
{

struct debug_struct
{
    debug_struct() noexcept
    {
        DEBUG_OSTREAM << "debug_struct()\n";
    }

    debug_struct(debug_struct const&) noexcept
    {
        DEBUG_OSTREAM << "debug_struct(const&)\n";
    }

    debug_struct(debug_struct&&) noexcept
    {
        DEBUG_OSTREAM << "debug_struct(&&)\n";
    }

    debug_struct& operator=(debug_struct const&) noexcept
    {
        DEBUG_OSTREAM << "operator= debug_struct(const&)\n";
        return *this;
    }

    debug_struct& operator=(debug_struct&&) noexcept
    {
        DEBUG_OSTREAM << "debug_struct(&&)\n";
        return *this;
    }

    ~debug_struct() noexcept
    {
        DEBUG_OSTREAM << "~debug_struct()\n";
    }
};

} // namespace utility::lifetimes

#endif // INCLUDED_LIFETIME_UTILITY
