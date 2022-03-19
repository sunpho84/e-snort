#ifndef _COMPLOOPS_HPP
#define _COMPLOOPS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file compLoops.hpp

#include <expr/comps.hpp>
#include <metaprogramming/inline.hpp>

namespace esnort
{
  namespace internal
  {
    template <typename Comps>
    struct _CompsLooper;
    
    template <typename FirstComp,
	      typename...RemainingComps>
    struct _CompsLooper<CompsList<FirstComp,
			       RemainingComps...>>
    {
      /// Loop to be executed
      template <typename DynamicComps,
		typename Function,
		typename...ProcessedComps>
      constexpr INLINE_FUNCTION
      static void loop(const DynamicComps& dynamicSizes,
                       Function function,
		       const ProcessedComps&...processedComps)
      {
	/// Loop task
	auto task=
	  [&dynamicSizes,&function,&processedComps...] (const FirstComp& comp) INLINE_ATTRIBUTE
	{
	  if constexpr(sizeof...(RemainingComps))
	    _CompsLooper<CompsList<RemainingComps...>>::loop(dynamicSizes,function,processedComps...,comp);
	  else
	    function(processedComps...,comp);
	};
	
	/// Size at compile time
	constexpr int sizeAtCompileTime=
		    FirstComp::sizeAtCompileTime;
	
	if constexpr(sizeAtCompileTime)
	  for(FirstComp comp=0;comp<sizeAtCompileTime;comp++)
	    task(comp);
	else
	  for(FirstComp comp=0;comp<std::get<FirstComp>(dynamicSizes);comp++)
	    task(comp);
      }
      
      /////////////////////////////////////////////////////////////////
      
      /// Loop to be executed
      template <typename DynamicComps,
		typename Function,
		typename...ProcessedComps>
      constexpr INLINE_FUNCTION DEVICE_ATTRIB
      static void deviceLoop(const DynamicComps& dynamicSizes,
			     Function function,
			     const ProcessedComps&...processedComps)
      {
	/// Loop task
	auto task=
	  [&dynamicSizes,&function,&processedComps...] DEVICE_ATTRIB (const FirstComp& comp) INLINE_ATTRIBUTE
	{
	  if constexpr(sizeof...(RemainingComps))
	    _CompsLooper<CompsList<RemainingComps...>>::deviceLoop(dynamicSizes,function,processedComps...,comp);
	  else
	    function(processedComps...,comp);
	};
	
	/// Size at compile time
	constexpr int sizeAtCompileTime=
		    FirstComp::sizeAtCompileTime;
	
	if constexpr(sizeAtCompileTime)
	  for(FirstComp comp=0;comp<sizeAtCompileTime;comp++)
	    task(comp);
	else
	  for(FirstComp comp=0;comp<std::get<FirstComp>(dynamicSizes);comp++)
	    task(comp);
      }
    };
  }
  
  /// Execute the given loop on all components
  template <typename Comps,
	    typename...DynamicComps,
	    typename Function,
	    typename...ProcessedComps>
  constexpr INLINE_FUNCTION
  static void loopOnAllComps(const CompsList<DynamicComps...>& dynamicSizes,
			     Function function,
			     const ProcessedComps&...processedComps)
  {
    internal::_CompsLooper<Comps>::loop(dynamicSizes,function,processedComps...);
  }
  
  /// Execute the given loop on all components
  template <typename Comps,
	    typename...DynamicComps,
	    typename Function,
	    typename...ProcessedComps>
  constexpr INLINE_FUNCTION DEVICE_ATTRIB
  static void deviceLoopOnAllComps(const CompsList<DynamicComps...>& dynamicSizes,
				   Function function,
				   const ProcessedComps&...processedComps)
  {
    internal::_CompsLooper<Comps>::loop(dynamicSizes,function,processedComps...);
  }
}

#endif
