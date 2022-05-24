#ifndef _LATTICE_COVERAGE_GETTER_HPP
#define _LATTICE_COVERAGE_GETTER_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file lattice/latticeCoverageGetter.hpp

#include <expr/nodes/subNodes.hpp>
#include <lattice/fieldDeclaration.hpp>
#include <tuples/tupleExecForAllTypes.hpp>

namespace grill
{
  template <typename Arg>
  constexpr auto getLatticeCoverage();
  
  namespace internal
  {
    template <typename...T>
    constexpr auto _tupleGetLatticeCoverages(const std::tuple<T...>*)
    {
      LatticeCoverage res=LatticeCoverage::NONE;
      bool found=true;
      
      for(LatticeCoverage lc : {getLatticeCoverage<std::decay_t<T>>()...})
	{
	  if(res==LatticeCoverage::NONE)
	    res=lc;
	  else
	    if(res!=lc)
	      found=false;
	}
      
      if(found)
     	return res;
      else
     	return LatticeCoverage::NONE;
    }
  }
  
  /// Returns the lattice inside the argument list
  template <typename Arg>
  constexpr auto getLatticeCoverage()
  {
    if constexpr(hasMember_latticeCoverage<Arg>)
      return Arg::latticeCoverage;
    else
      if constexpr(hasMember_subNodes<Arg>)
	return internal::_tupleGetLatticeCoverages((typename Arg::SubNodes::type*){});
      else
	return LatticeCoverage::NONE;
  }
}

#endif
