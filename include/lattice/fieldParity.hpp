#ifndef _FIELDPARITY_HPP
#define _FIELDPARITY_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file grill/fieldParity.hpp

#include <lattice/fieldDeclaration.hpp>

namespace grill
{
  /// Provides the parity of a given site
  template <typename L,
	    LatticeCoverage LC>
  struct FieldParity
  {
    static_assert(LC!=LatticeCoverage::EVEN_ODD,"Cannot define the parity of an even/odd field");
    
    static constexpr typename L::Parity parity=
      (LC==LatticeCoverage::EVEN)?0:1;
  };
  
  /// Default case in which no parity is defined
  template <typename L>
  struct FieldParity<L,LatticeCoverage::EVEN_ODD>
  {
  };
}

#endif
