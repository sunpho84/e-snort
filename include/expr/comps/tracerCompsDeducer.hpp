#ifndef _TRACERCOMPSDEDUCER_HPP
#define _TRACERCOMPSDEDUCER_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/nodes/tracerCompsDeducer.hpp

#include <expr/comps/comps.hpp>

namespace grill
{
  /// Classifies the components, determining which one are visible or traced
  ///
  /// Internal implementation, forward declaration
  template <typename TC>
  struct TracerCompsDeducer;
  
  /// Classifies the components, determining which one are visible or traced
  template <typename...Tc>
  struct TracerCompsDeducer<CompsList<Tc...>>
  {
    template <typename T>
    struct Classify
    {
      static constexpr bool isTraced=
	T::isTransposable and
	(std::is_same_v<T,Tc> or...) and
	(std::is_same_v<typename T::Transp,Tc> or...);
      
      ///
      using VisiblePart=
	std::conditional_t<isTraced,std::tuple<>,std::tuple<T>>;
      
      using TracedPart=
	std::conditional_t<isTraced and isRow<T>,std::tuple<T>,std::tuple<>>;
    };
    
    using VisibleComps=
      TupleCat<typename Classify<Tc>::VisiblePart...>;
    
    using TracedComps=
      TupleCat<typename Classify<Tc>::TracedPart...>;
  };
}

#endif
