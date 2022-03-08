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
#include <expr/indexComputer.hpp>
#include <resources/memory.hpp>
#include <tuples/tupleDiscriminate.hpp>

namespace esnort
{
  /// Tensor
  ///
  /// Forward declaration
  template <typename C,
	    typename F,
	    ExecutionSpace ES>
  struct DynamicTens;
  
#define THIS					\
  DynamicTens<CompsList<C...>,_Fund,ES>
  
  /// Tensor
  template <typename...C,
	    typename _Fund,
	    ExecutionSpace ES>
  struct THIS :
    Expr<THIS>,
    DynamicCompsProvider<C...>
  {
    using This=THIS;
    
#undef THIS
    
    using Expr<This>::operator=;
    
    /// Components
    using Comps=CompsList<C...>;
    
    /// Fundamental type
    using Fund=_Fund;
    
    /// Executes where allocated
    static constexpr ExecutionSpace execSpace=ES;
    
    /// Cost of changing the execution space
    static constexpr auto execSpaceChangeCost=
      ExecutionSpaceChangeCost::ALOT;
    
    /// Pointer to storage
    Fund* storage;
    
    /// Storage size
    int64_t storageSize;
    
    /// Determine if allocated
    bool allocated{false};
    
    /// Allocate the storage
    template <typename...TD>
    void allocate(const CompsList<TD...>& td)
    {
      if(allocated)
	CRASH<<"Already allocated";
      
      tupleFillWithSubset(this->dynamicSizes,td);
      
      storageSize=indexMaxValue<C...>(this->dynamicSizes);
      
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
    
#define PROVIDE_EVAL(ATTRIB)						\
    template <typename...U>						\
    HOST_DEVICE_ATTRIB constexpr INLINE_FUNCTION			\
    ATTRIB Fund& eval(const U&...cs) ATTRIB				\
    {									\
      return storage[orderedIndex<C...>(this->dynamicSizes,cs...)];	\
    }
    
    PROVIDE_EVAL(const);
    
    PROVIDE_EVAL(/* non const */);
    
#undef PROVIDE_EVAL
  };
}

#endif
