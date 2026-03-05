#ifndef AMR_INCLUDED_CONFIG_DEFINITION
#define AMR_INCLUDED_CONFIG_DEFINITION

#include "utility/macro_definitions.hpp"

// -----------------------
// Loggiing
// -----------------------
#ifdef AMR_LOG_LEVEL
#    define UTILITY_LOG_LEVEL AMR_LOG_LEVEL
#endif

// -----------------------
// EXECUTION_POLICY
// -----------------------
#ifndef AMR_EXECUTION_SEQ
#    define AMR_EXECUTION_SEQ std::execution::seq
#endif
#ifndef AMR_EXECUTION_PAR
#    define AMR_EXECUTION_PAR std::execution::par_unseq
#endif

#ifdef AMR_EXECUTION
#    define AMR_EXECUTION_POLICY UTILITY_CONCATENATE_MACRO(AMR_EXECUTION_, AMR_EXECUTION)
#else
#    define AMR_EXECUTION_POLICY AMR_EXECUTION_SEQ
#endif

#endif // AMR_INCLUDED_CONFIG_DEFINITION
