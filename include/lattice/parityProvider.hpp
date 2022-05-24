#ifndef _PARITY_PROVIDER_HPP
#define _PARITY_PROVIDER_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file grill/parityProvider.hpp

#include <lattice/latticeCoverage.hpp>

namespace grill
{
  /// Provides the parity of a given site
  template <typename L,
	    LatticeCoverage LC>
  struct ParityProvider
  {
    static_assert(LC!=LatticeCoverage::EVEN_ODD,"Cannot define the parity of an even/odd coverage");
    
    static_assert(LC!=LatticeCoverage::NONE,"Cannot define the parity when no coverage present");
    
    static constexpr typename L::Parity parity=
      (LC==LatticeCoverage::EVEN)?0:1;
  };
  
  /// Default case in which no parity is defined
  template <typename L>
  struct ParityProvider<L,LatticeCoverage::NONE>
  {
  };
  
  /// Default case in which both parities are defined
  template <typename L>
  struct ParityProvider<L,LatticeCoverage::EVEN_ODD>
  {
  };
}

#endif
