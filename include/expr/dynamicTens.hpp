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
#include <metaprogramming/constnessChanger.hpp>
#include <resources/memory.hpp>
#include <tuples/tupleDiscriminate.hpp>

namespace esnort
{
  /// Tensor
  ///
  /// Forward declaration
  template <typename C,
	    typename F,
	    ExecutionSpace ES,
	    bool _IsRef=false>
  struct DynamicTens;
  
#define THIS					\
  DynamicTens<CompsList<C...>,_Fund,ES,IsRef>
  
  /// Tensor
  template <typename...C,
	    typename _Fund,
	    ExecutionSpace ES,
	    bool IsRef>
  struct THIS :
    Expr<THIS>,
    DynamicCompsProvider<C...>
  {
    using This=THIS;
    
#undef THIS
    
    using Expr<This>::operator=;
    
    /// List of dynamic comps
    using DynamicComps=
      typename DynamicCompsProvider<C...>::DynamicComps;
    
    /// Components
    using Comps=CompsList<C...>;
    
    /// Fundamental type
    using Fund=_Fund;

    /// Determine if it is reference
    static constexpr bool isRef=IsRef;
    
    /// Executes where allocated
    static constexpr ExecutionSpace execSpace=ES;
    
    /// Cost of changing the execution space
    static constexpr auto execSpaceChangeCost=
      ExecutionSpaceChangeCost::ALOT;
    
    /// Pointer to storage
    ConstIf<isRef,Fund*> storage;
    
    /// Storage size
    ConstIf<isRef,int64_t> storageSize;
    
    /// Determine if allocated
    ConstIf<isRef,bool> allocated{false};
    
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
      if constexpr(not isRef)
	allocate(td);
      else
	CRASH<<"Trying to allocate a reference";
    }
    
    /// Initialize the tensor without allocating
    constexpr
    DynamicTens()
    {
      if constexpr(not isRef)
	{
	  if constexpr(DynamicCompsProvider<C...>::nDynamicComps==0)
	    allocate({});
	  else
	    allocated=false;
	}
      else
	CRASH<<"Trying to create a reference to nothing";
    }
    
    /// Initialize the tensor as a reference
    constexpr
    DynamicTens(Fund* storage,
		const int64_t& storageSize,
		const DynamicComps& dynamicSizes) :
      DynamicCompsProvider<C...>{dynamicSizes},
      storage(storage),
      storageSize(storageSize)
    {
      if constexpr(not isRef)
	CRASH<<"Trying to create as a reference a non-reference";
    }
    
    /// Assign from another dynamic tensor of the very same type
    template <ExecutionSpace OtherES,
	      bool OtherIsRef>
    DynamicTens& operator=(const DynamicTens<Comps,Fund,OtherES,OtherIsRef>& oth)
    {
      if(storageSize!=oth.storageSize)
	CRASH<<"Storage size not agreeing";
      memory::memcpy<ES,OtherES>(storage,oth.storage,oth.storageSize);
      
      return *this;
    }
    
    /// Destructor
    ~DynamicTens()
    {
      if constexpr(not isRef)
	{
	  if(allocated)
	    memory::manager<ES>.release(storage);
	  allocated=false;
	  storageSize=0;
	}
    }
    
    /////////////////////////////////////////////////////////////////
    
#define PROVIDE_GET_REF(ATTRIB)						\
    auto getRef() ATTRIB						\
    {									\
      return DynamicTens<Comps,ATTRIB Fund,ES,true>(storage,storageSize,this->dynamicSizes); \
    }
    
    PROVIDE_GET_REF(const);
    
    PROVIDE_GET_REF(/* non const */);
    
#undef PROVIDE_GET_REF
    
    /////////////////////////////////////////////////////////////////
    
#define PROVIDE_EVAL(ATTRIB)						\
    template <typename...U>						\
    HOST_DEVICE_ATTRIB constexpr INLINE_FUNCTION			\
    ATTRIB Fund& eval(const U&...cs) ATTRIB				\
    {									\
      assertCorrectEvaluationStorage<ES>();				\
      return storage[orderedIndex<C...>(this->dynamicSizes,cs...)];	\
    }
    
    PROVIDE_EVAL(const);
    
    PROVIDE_EVAL(/* non const */);
    
#undef PROVIDE_EVAL
  };
}

#endif
