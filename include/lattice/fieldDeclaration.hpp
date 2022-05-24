#ifndef _FIELDDECLARATION_HPP
#define _FIELDDECLARATION_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file lattice/fieldDeclaration.hpp

#include <expr/assign/executionSpace.hpp>
#include <lattice/latticeCoverage.hpp>
#include <tuples/tupleHasType.hpp>

namespace grill
{
  /// Layout to be used for organizing internal data
  enum class FieldLayout{SERIAL,SIMDIFIABLE,SIMDIFIED,GPU};
  
  /// Use or not the halo
  enum class HaloPresence{WITHOUT_HALO,WITH_HALO};
  
  /// Default presence of halo
  inline constexpr HaloPresence defaultHaloPresence=
    HaloPresence::WITHOUT_HALO;
  
  /// Default field layout
  inline constexpr FieldLayout defaultFieldLayout=
    FieldLayout::
#if ENABLE_DEVICE_CODE
    GPU
#else
    SIMDIFIABLE
#endif
    ;
  
  PROVIDE_HAS_MEMBER(fieldLayout);
  
  /// Field, forward declaration
  template <typename InnerComps,
	    typename Fund,
	    typename L,
	    LatticeCoverage LC=LatticeCoverage::EVEN_ODD,
	    FieldLayout FL=defaultFieldLayout,
	    ExecSpace ES=defaultExecSpace,
	    bool IsRef=false>
  struct Field;
}

#endif
