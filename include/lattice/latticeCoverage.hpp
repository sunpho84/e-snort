#ifndef _LATTICECOVERAGE_HPP
#define _LATTICECOVERAGE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file lattice/latticeCoverage.hpp

#include <metaprogramming/hasMember.hpp>

namespace grill
{
  /// Specifies whether we cover the full lattice, the even or odd part
  enum class LatticeCoverage{NONE,EVEN,ODD,EVEN_ODD};
  
  inline constexpr LatticeCoverage oppositeLatticeCoverage(const LatticeCoverage& LC)
  {
    switch (LC)
      {
      case LatticeCoverage::NONE:
	return  LatticeCoverage::NONE;
	break;
      case LatticeCoverage::EVEN:
	return  LatticeCoverage::ODD;
	break;
      case LatticeCoverage::ODD:
	return  LatticeCoverage::EVEN;
	break;
      case LatticeCoverage::EVEN_ODD:
	return  LatticeCoverage::EVEN_ODD;
	break;
      }
  }
  
  PROVIDE_HAS_MEMBER(latticeCoverage);
}

#endif
