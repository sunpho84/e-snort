#ifndef _FIELDDECLARATION_HPP
#define _FIELDDECLARATION_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file lattice/fieldDeclaration.hpp

#include <expr/assign/executionSpace.hpp>

namespace grill
{
  /// Specifies whether we cover the full lattice, the even or odd part
  enum class LatticeCoverage{EVEN,ODD,EVEN_ODD};
  
  /// Layout to be used for organizing internal data
  enum class FieldLayout{SERIAL,SIMD,GPU};
  
  /// Field, forward declaration
  template <typename TP,
	    typename Fund,
	    typename L,
	    LatticeCoverage LC,
	    FieldLayout FL,
	    ExecSpace ES>
  struct Field;
}

#endif
