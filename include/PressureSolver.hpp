#pragma once

#include "domain.hpp"
#include <utility>

class SOR
{
  public:
    SOR() = default;

    /**
     * @brief Constructor of SOR solver
     *
     * @param[in] relaxation factor
     */
    SOR(double omega);

    double solve(sim_domain& tree) const;

  private:
    double _omega;
};
