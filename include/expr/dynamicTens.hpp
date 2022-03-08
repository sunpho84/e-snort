#ifndef _DYNAMICTENS_HPP
#define _DYNAMICTENS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/dynamicTens.hpp

#include <expr/comp.hpp>
#include <expr/comps.hpp>
#include <expr/dynamicCompsProvider.hpp>
#include <expr/executionSpace.hpp>
#include <expr/expr.hpp>
#include <resources/memory.hpp>
#include <tuples/tupleDiscriminate.hpp>

namespace esnort
{
  /// Tensor
  
  /// Forward declaration
  template <typename C,
	    typename F,
	    ExecutionSpace ES>
  struct DynamicTens;
  
  /// Tensor
  template <typename...C,
	    typename _Fund,
	    ExecutionSpace ES>
  struct DynamicTens<CompsList<C...>,_Fund,ES> :
    Expr<DynamicTens<CompsList<C...>,_Fund,ES>>,
    DynamicCompsProvider<C...>
  {
    /// Fundamental type
    using Fund=_Fund;
    
    /// Executes where allocated
    static constexpr ExecutionSpace execSpace()
    {
      return ES;
    }
    
    static constexpr auto execSpaceChangeCost()
    {
      return ExecutionSpaceChangeCost::ALOT;
    }
    
    /// Components
    using Comps=CompsList<C...>;
    
    /// Pointer to storage
    Fund* storage;
    
    /// Storage size
    int64_t storageSize;
    
    /// Determine if allocated
    bool allocated{false};
    
    /// Gets the maximal value for the given comp
    template <typename T>
    constexpr auto _getMaxCompValue() const
    {
      if constexpr(T::sizeIsKnownAtCompileTime)
	return T::sizeAtCompileTime;
      else
	return std::get<T>(this->dynamicSizes)();
    }
    
    /// Allocate the storage
    template <typename...TD>
    void allocate(const CompsList<TD...>& td)
    {
      if(allocated)
	CRASH<<"Already allocated";
      
      tupleFillWithSubset(this->dynamicSizes,td);
      
      storageSize=(_getMaxCompValue<C>()*...*1);
      
      storage=memory::manager<ES>.template provide<Fund>(storageSize);
      
      allocated=true;
    }
    
    /// Initialize the tensor with the knowledge of the dynamic sizes
    template <typename...TD>
    explicit DynamicTens(const CompsList<TD...>& td)
    {
      allocate(td);
    }
    
    /// Initialize the tensor without allocating
    constexpr
    DynamicTens()
    {
      if constexpr(DynamicCompsProvider<C...>::nDynamicComps==0)
	allocate({});
      else
	allocated=false;
    }
    
    /// Destructor
    ~DynamicTens()
    {
      if(allocated)
	memory::manager<ES>.release(storage);
      allocated=false;
    }
  };

}

#endif
