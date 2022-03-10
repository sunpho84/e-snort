#ifndef _STACKTENS_HPP
#define _STACKTENS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/stackTens.hpp

#include <expr/baseTens.hpp>
#include <expr/comps.hpp>
#include <expr/dynamicCompsProvider.hpp>
#include <expr/executionSpace.hpp>
#include <expr/expr.hpp>
#include <expr/indexComputer.hpp>

namespace esnort
{
  /// Tensor
  ///
  /// Forward declaration
  template <typename C,
	    typename F>
  struct StackTens;
  
#define THIS					\
    StackTens<CompsList<C...>,_Fund>
  
  /// Tensor
  template <typename...C,
	    typename _Fund>
  struct THIS :
    BaseTens<THIS,_Fund,ExecutionSpace::HOST>
  {
    using This=THIS;
    
#undef THIS
    
    using BaseTens<This,_Fund,ExecutionSpace::HOST>::operator=;
    
    static_assert((C::sizeIsKnownAtCompileTime &...),"Trying to instantiate a stack tensor with dynamic comps");
    
    /// Components
    using Comps=CompsList<C...>;
    
    /// Fundamental type
    using Fund=_Fund;
    
    /// Executes where allocated
    static constexpr auto execSpace=
      ExecutionSpace::HOST;
    
    /// Cost of changing the execution space
    static constexpr auto execSpaceChangeCost=
      ExecutionSpaceChangeCost::ALOT;
    
    /// Size of stored data
    static constexpr auto storageSize=
      indexMaxValue<C...>();
    
    /// Data
    Fund storage[storageSize];
    
#define PROVIDE_EVAL(ATTRIB)					\
    template <typename...U>					\
    constexpr INLINE_FUNCTION					\
    ATTRIB Fund& eval(const U&...cs) ATTRIB			\
    {								\
      return storage[orderedIndex<C...>(std::tuple<>{},cs...)];	\
    }
    
    PROVIDE_EVAL(const);
    
    PROVIDE_EVAL(/* non const */);
    
#undef PROVIDE_EVAL
  };
}

#endif
