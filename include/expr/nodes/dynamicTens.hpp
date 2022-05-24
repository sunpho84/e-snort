#ifndef _DYNAMICTENS_HPP
#define _DYNAMICTENS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/nodes/dynamicTens.hpp

#include <expr/comps/comp.hpp>
#include <expr/comps/comps.hpp>
#include <expr/nodes/baseTens.hpp>
#include <expr/comps/dynamicCompsProvider.hpp>
#include <expr/nodes/dynamicTensDeclaration.hpp>
#include <expr/assign/executionSpace.hpp>
#include <expr/nodes/node.hpp>
#include <expr/comps/indexComputer.hpp>
#include <expr/nodes/tensRef.hpp>
#include <metaprogramming/constnessChanger.hpp>
#include <resources/memory.hpp>
#include <tuples/tupleDiscriminate.hpp>

namespace grill
{
#define THIS					\
  DynamicTens<CompsList<C...>,_Fund,ES>
  
#define BASE					\
  BaseTens<THIS,CompsList<C...>,_Fund,ES>
  
  /// Dynamic Tensor
  template <typename...C,
	    typename _Fund,
	    ExecSpace ES>
  struct THIS :
    BASE,
    DetectableAsDynamicTens
  {
    using This=THIS;
    using Base=BASE;
    
#undef BASE
#undef THIS
    
    /// Importing assignment operator from BaseTens
    using Base::operator=;
    
    /// Copy assign
    INLINE_FUNCTION
    DynamicTens& operator=(const DynamicTens& oth)
    {
      Base::operator=(oth);
      
      return *this;
    }
    
    /// Move assign
    INLINE_FUNCTION
    DynamicTens& operator=(DynamicTens&& oth)
    {
      if(dynamicSizes!=oth.dynamicSizes)
	CRASH<<"trying to assign different dynamic sized tensor";
      
      if(not canAssign())
	CRASH<<"trying to assign to unsassignable tensor";
      
      std::swap(this->storage,oth.storage);
      std::swap(this->allocated,oth.allocated);
      
      return *this;
    }
    
    /// List of dynamic comps
    using DynamicComps=
      typename Base::DynamicComps;
    
    /// Components
    using Comps=CompsList<C...>;
    
    /// Fundamental type
    using Fund=_Fund;
    
    /// Executes where allocated
    static constexpr ExecSpace execSpace=ES;
    
    /// Sizes of the dynamic components
    DynamicComps dynamicSizes;
    
    /// Returns the dynamic sizes
    const DynamicComps& getDynamicSizes() const
    {
      return dynamicSizes;
    }
    
    /// Pointer to storage
    Fund* storage;
    
    /// Number of elements
    int64_t nElements;
    
    /// Determine if allocated
    bool allocated{false};
    
    /// Returns whether can assign
    bool canAssign()
    {
      return allocated;
    }
    
    /// We keep referring to the original object
    static constexpr bool storeByRef=true;
    
    /// Allocate the storage
    template <typename...T,
	      ENABLE_THIS_TEMPLATE_IF(tupleHaveTypes<std::tuple<T...>,DynamicComps>)>
    void allocate(const std::tuple<T...>& _dynamicSizes)
    {
      if(allocated)
	CRASH<<"Already allocated";
      
      tupleFillWithSubset(dynamicSizes,_dynamicSizes);
      
      nElements=indexMaxValue<C...>(this->dynamicSizes);
      
      storage=memory::manager<ES>.template provide<Fund>(nElements);
      
      allocated=true;
    }
    
    /// Allocate the storage
    template <typename...T,
	      typename...I>
    void allocate(const BaseComp<T,I>&...td)
    {
      allocate(Base::filterDynamicComps(td...));
    }
    
    /// Initialize the tensor with the knowledge of the dynamic sizes as a list
    template <typename...T,
	      typename...I>
    explicit DynamicTens(const BaseComp<T,I>&...td) :
      DynamicTens(Base::filterDynamicComps(td...))
    {
    }
    
    /// Initialize the tensor with the knowledge of the dynamic sizes
    explicit DynamicTens(const DynamicComps& td)
    {
      allocate(td);
    }
    
    /// Initialize the tensor without allocating
    constexpr
    DynamicTens()
    {
      // if constexpr(DynamicCompsProvider<Comps>::nDynamicComps==0)
      // 	allocate();
      // else
	allocated=false;
    }
    
    /// Create from another node
    template <typename TOth>
    constexpr INLINE_FUNCTION
    explicit DynamicTens(const Node<TOth>& oth) :
      DynamicTens(DE_CRTPFY(const TOth,&oth).getDynamicSizes())
    {
      (*this)=DE_CRTPFY(const TOth,&oth);
    }
    
    /// Copy constructor
    DynamicTens(const DynamicTens& oth) :
      DynamicTens(oth.getDynamicSizes())
    {
      LOGGER_LV3_NOTIFY("Using copy constructor of DynamicTens");
      (*this)=oth;
    }
    
    // /Move constructor
    DynamicTens(DynamicTens&& oth) :
      dynamicSizes(oth.dynamicSizes),storage(oth.storage),nElements(oth.nElements)
    {
      LOGGER_LV3_NOTIFY("Using move constructor of DynamicTens");
      
      oth.allocated=false;
    }
    
    /// Destructor
    HOST_DEVICE_ATTRIB
    ~DynamicTens()
    {
#ifndef __CUDA_ARCH__
      if(allocated)
	memory::manager<ES>.release(storage);
      allocated=false;
      nElements=0;
#endif
    }
    
    /////////////////////////////////////////////////////////////////
    
#define PROVIDE_EVAL(ATTRIB)						\
    template <typename...U>						\
    HOST_DEVICE_ATTRIB constexpr INLINE_FUNCTION			\
    ATTRIB Fund& eval(const U&...cs) ATTRIB				\
    {									\
      assertCorrectEvaluationStorage<ES>();				\
      									\
      return storage[orderedIndex<C...>(this->dynamicSizes,cs...)];	\
    }
    
    PROVIDE_EVAL(const);
    
    PROVIDE_EVAL(/* non const */);
    
#undef PROVIDE_EVAL
  };
  
  /// Fill a dynamic tensor and returns it
  template <typename T>
  INLINE_FUNCTION
  auto Node<T>::fillDynamicTens() const
  {
    DynamicTens<typename T::Comps,typename T::Fund,T::execSpace> res(DE_CRTPFY(const T,this).getDynamicSizes());
    
    res=*this;
    
    return res;
  }
  
  /////////////////////////////////////////////////////////////////
#define PROVIDE_FUND_CAST(ATTRIB)					\
  template <typename T,							\
	    typename...C,						\
	    typename F,							\
	    ExecSpace ES>						\
  template <typename FC>						\
  auto BaseTens<T,CompsList<C...>,F,ES>::fundCast() ATTRIB		\
  {									\
    decltype(auto) t=DE_CRTPFY(ATTRIB T,this);				\
									\
    if(not t.allocated)							\
      CRASH<<"Cannot take the reference of a non allocated tensor";	\
									\
    return TensRef<CompsList<C...>,ATTRIB FC,ES>((ATTRIB FC*)t.storage,t.nElements,t.getDynamicSizes()); \
  }
  
  PROVIDE_FUND_CAST(const);
  
  PROVIDE_FUND_CAST(/* non const */);
  
#undef PROVIDE_FUND_CAST
  
/////////////////////////////////////////////////////////////////
  
#define PROVIDE_BASE_FUND_CAST_OPERATOR(ATTRIB)				\
  template <typename T,							\
	    typename...C,						\
	    typename F,							\
	    ExecSpace ES>						\
  auto BaseTens<T,CompsList<C...>,F,ES>::operator~() ATTRIB		\
  {									\
    static_assert(isComp<F>,"For now only defined for comps");		\
    									\
    return this->fundCast<typename F::Index>();				\
  }
  
  PROVIDE_BASE_FUND_CAST_OPERATOR(const);
  
  PROVIDE_BASE_FUND_CAST_OPERATOR(/* non const */);
  
#undef PROVIDE_BASE_FUND_CAST_OPERATOR
  
  /////////////////////////////////////////////////////////////////
  
#define PROVIDE_GET_REF(ATTRIB)						\
  template <typename T,							\
	    typename...C,						\
	    typename F,							\
	    ExecSpace ES>						\
  auto BaseTens<T,CompsList<C...>,F,ES>::getRef() ATTRIB		\
  {									\
    return this->fundCast<F>();						\
  }
  
  PROVIDE_GET_REF(const);
  
  PROVIDE_GET_REF(/* non const */);
  
#undef PROVIDE_GET_REF
  
  /////////////////////////////////////////////////////////////////
  
#define PROVIDE_SIMDIFY(ATTRIB)						\
  template <typename T,							\
	    typename...C,						\
	    typename F,							\
	    ExecSpace ES>						\
  INLINE_FUNCTION							\
  auto BaseTens<T,CompsList<C...>,F,ES>::simdify() ATTRIB		\
									\
  {									\
    decltype(auto) t=DE_CRTPFY(ATTRIB T,this);				\
									\
    /*LOGGER<<"Building simdified view "<<execSpaceName<ES><<" tensor-like, pointer: "<<t.storage;*/ \
    									\
    									\
    using Traits=CompsListSimdifiableTraits<CompsList<C...>,F>;		\
									\
    using SimdFund=typename Traits::SimdFund;				\
									\
    return TensRef<typename Traits::Comps,ATTRIB SimdFund,ES>		\
      ((ATTRIB SimdFund*)t.storage,					\
       t.nElements/Traits::nNonSimdifiedElements,			\
       t.getDynamicSizes());						\
  }
  
  PROVIDE_SIMDIFY(const);
  
  PROVIDE_SIMDIFY(/* non const */);
  
#undef PROVIDE_SIMDIFY
  
  /////////////////////////////////////////////////////////////////

  template <typename T,
	    typename...C,
	    typename F,
	    ExecSpace ES>
  template <ExecSpace OES>
  DynamicTens<CompsList<C...>,F,OES> BaseTens<T,CompsList<C...>,F,ES>::getCopyOnExecSpace() const
  {
    if constexpr(ES==OES)
      return *this;
    else
      {
	/// Derived class of this
	const T& t=DE_CRTPFY(const T,this);
	
	/// Result
	DynamicTens<CompsList<C...>,F,OES> res(t.getDynamicSizes());
	
	/// \todo if no component is dynamic and we are on host, we could return a stackTens
	
	device::memcpy<OES,ES>(res.storage,t.storage,t.nElements*sizeof(F));
	
	return res;
      }
  }
  
  /// Calss the fundamental caster
  template <typename F,
	    typename T>
  auto fundCast(T&& t)
  {
    return t.template fundCast<F>();
  }
}

#endif
