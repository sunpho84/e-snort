#ifndef _TENSREF_HPP
#define _TENSREF_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/tensRef.hpp

#include <expr/baseTens.hpp>
#include <expr/comps.hpp>
#include <expr/executionSpace.hpp>
#include <expr/expr.hpp>
#include <expr/indexComputer.hpp>

namespace esnort
{
  PROVIDE_DETECTABLE_AS(TensRef);
  
  /// Tensor reference
  ///
  /// Forward declaration
  template <typename C,
	    typename F,
	    ExecSpace ES>
  struct TensRef;

#define THIS					\
  TensRef<CompsList<C...>,_Fund,ES>

#define BASE					\
  BaseTens<THIS,CompsList<C...>,_Fund,ES>
  
  /// Tensor reference
  template <typename...C,
	    typename _Fund,
	    ExecSpace ES>
  struct THIS :
    BASE,
    DetectableAsTensRef
  {
    using This=THIS;
    using Base=BASE;
    
#undef BASE
#undef THIS
    
    /// Importing assignment operator from Expr
    using Base::operator=;
    
    /// Copy assign: we still assign the content
    INLINE_FUNCTION
    TensRef& operator=(const TensRef& oth)
    {
      Base::operator=(oth);
      
      return *this;
    }
    
    /// Move assign
    INLINE_FUNCTION
    TensRef& operator=(TensRef&& oth)
    {
      std::swap(this->storage,oth.storage);
      std::swap(this->dynamicSizes,oth.dynamicSizes);
      
      return *this;
    }
    
    /// List of dynamic comps
    using DynamicComps=
      typename Base::DynamicComps;
    
    /// Components
    using Comps=CompsList<C...>;
    
    /// Fundamental type
    using Fund=_Fund;
    
    /// Executes where originally allocated
    static constexpr ExecSpace execSpace=ES;
    
    /// Sizes of the dynamic components
    DynamicComps dynamicSizes;
    
    /// Returns the dynamic sizes
    const DynamicComps& getDynamicSizes() const
    {
      return dynamicSizes;
    }
    
    /// Pointer to storage
    Fund* const storage;
    
    /// Number of elements
    const int64_t nElements;
    
    /// Returns whether can assign: we must assume that the reference is pointing to a valid pointer
    bool canAssign()
    {
      return true;
    }
    
    /// We can safely copy since it is a light object
    static constexpr bool storeByRef=false;
    
    /// Initialize the reference
    INLINE_FUNCTION HOST_DEVICE_ATTRIB constexpr
    TensRef(Fund* storage,
	    const int64_t& nElements,
	    const DynamicComps& dynamicSizes) :
      dynamicSizes(dynamicSizes),
      storage(storage),
      nElements(nElements)
    {
    }
    
    /// Copy constructor
    INLINE_FUNCTION HOST_DEVICE_ATTRIB
    TensRef(const TensRef& oth) :
      TensRef(oth.storage,oth.nElements,oth.dynamicSizes)
    {
    }
    
    // /Move constructor
    INLINE_FUNCTION HOST_DEVICE_ATTRIB
    TensRef(TensRef&& oth) :
      TensRef(oth.storage,nElements,dynamicSizes)
    {
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
}

#endif
