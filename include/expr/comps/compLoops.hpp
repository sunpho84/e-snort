#ifndef _COMPLOOPS_HPP
#define _COMPLOOPS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/comps/compLoops.hpp

#include <expr/comps/comps.hpp>
#include <metaprogramming/inline.hpp>
#include <metaprogramming/unrolledFor.hpp>

namespace esnort
{
  /// Loop over a component
  template <typename T,
	    typename F,
	    ENABLE_THIS_TEMPLATE_IF(T::sizeAtCompileTime>0)>
  INLINE_FUNCTION constexpr HOST_DEVICE_ATTRIB
  void compLoop(F f)
  {
    constexpr int n=T::sizeAtCompileTime;
    
    if constexpr(n<10)
      unrolledFor<0,n>(f);
    else
      for(typename T::Index i=0;i<n;i++)
	f(i);
  }
  
  /// Loop over a component
  template <typename T,
	    typename F,
	    ENABLE_THIS_TEMPLATE_IF(not T::sizeAtCompileTime)>
  INLINE_FUNCTION constexpr HOST_DEVICE_ATTRIB
  void compLoop(F f,typename T::Index n)
  {
    for(typename T::Index i=0;i<n;i++)
      f(i);
  }
  
  namespace internal
  {
    /// Loops on all components
    ///
    /// Forward declaration
    template <typename Comps>
    struct _CompsLooper;
    
    /// Loops on all components
    ///
    /// Empty list of components
    template <>
    struct _CompsLooper<CompsList<>>
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
	// Avoid warning on unused dynamicSizes
	(void)dynamicSizes;
	
	function(processedComps...);
      };
    };
    
    /// Loops on all components
    ///
    /// Default case
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
	    {
	      // Avoid warning on unused dynamicSizes
	      (void)dynamicSizes;
	      
	      function(processedComps...,comp);
	    }
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
	  [&dynamicSizes,&function,&processedComps...] DEVICE_ATTRIB (const FirstComp& comp) CONSTEXPR_INLINE_ATTRIBUTE
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
	  compLoop<Comp>(task);
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
