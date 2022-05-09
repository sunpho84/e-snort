#ifndef _THREADEDASSIGN_HPP
#define _THREADEDASSIGN_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/assign/threadAssign.hpp
///
/// \brief Assign two expressions using threads

#include <expr/assign/assignerFactory.hpp>
#include <expr/comps/compLoops.hpp>
#include <ios/logger.hpp>
#include <metaprogramming/inline.hpp>

namespace grill
{
  /// Assign two expressions using device
  template <typename Lhs,
	    typename Rhs>
  INLINE_FUNCTION
  void threadAssign(Lhs& lhs,
		    const Rhs& rhs)
  {
    static_assert(Lhs::nDynamicComps<=1,"Need exactly one dynamic comps to run threads");
    
    /// We use threads only if there is at least one dynamic component
    LOGGER<<"Using thread kernel";
    
    using DC=std::tuple_element_t<0,typename Lhs::DynamicComps>;
    
    using Index=typename DC::Index;
    
    const auto dynamicSize=
      ~std::get<0>(lhs.getDynamicSizes());
    
    const auto assign=getAssigner(lhs,rhs);
    
#pragma omp parallel for
    for(Index dc=0;dc<dynamicSize;dc++)
      loopOnAllComps<typename Lhs::StaticComps>(lhs.getDynamicSizes(),assign,DC(dc));
  }
}

#endif
