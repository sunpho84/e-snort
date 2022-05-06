#ifndef _STACKTENS_HPP
#define _STACKTENS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/nodes/stackTens.hpp

#include <expr/nodes/baseTens.hpp>
#include <expr/comps/comps.hpp>
#include <expr/comps/dynamicCompsProvider.hpp>
#include <expr/assign/executionSpace.hpp>
#include <expr/nodes/node.hpp>
#include <expr/comps/indexComputer.hpp>

namespace grill
{
  PROVIDE_DETECTABLE_AS(StackTens);
  
  /// Tensor
  ///
  /// Forward declaration
  template <typename C,
	    typename F>
  struct StackTens;
  
#define THIS					\
    StackTens<CompsList<C...>,_Fund>
  
#define BASE					\
  BaseTens<THIS,CompsList<C...>,_Fund,ExecSpace::HOST>
  
  /// Tensor
  template <typename...C,
	    typename _Fund>
  struct THIS :
    BASE,
    DetectableAsStackTens
  {
    using This=THIS;
    
    using Base=BASE;
    
#undef BASE
#undef THIS
    
    /// Import base class assigners
    using Base::operator=;
    
    /// Copy-assign
    INLINE_FUNCTION
    StackTens& operator=(const StackTens& oth)
    {
      Base::operator=(oth);
      
      return *this;
    }
    
    /// Move-assign
    INLINE_FUNCTION
    StackTens& operator=(StackTens&& oth)
    {
      Base::operator=(std::move(oth));
      
      return *this;
    }
    
    static_assert((C::sizeIsKnownAtCompileTime and ... and true),"Trying to instantiate a stack tensor with dynamic comps");
    
    /// Components
    using Comps=CompsList<C...>;
    
    /// Fundamental type
    using Fund=_Fund;
    
    /// Executes where allocated
    static constexpr auto execSpace=
      ExecSpace::HOST;
    
    /// Returns empty dynamic sizes
    constexpr const CompsList<> getDynamicSizes() const
    {
      return {};
    }
    
    /// Size of stored data
    static constexpr auto nElements=
      indexMaxValue<C...>();
    
    /// Data
    Fund storage[nElements];
    
    /// We store the reference to the tensor
    static constexpr bool storeByRef=true;
    
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
    
    /// Default constructor
    constexpr INLINE_FUNCTION
    StackTens(const CompsList<> ={})
    {
      if constexpr(std::is_class_v<_Fund>)
	{
	  for(std::decay_t<decltype(nElements)> iEl=0;iEl<nElements;iEl++)
	    new(&storage[iEl]) _Fund;
	}
    }
    
    /// Copy constructor
    constexpr INLINE_FUNCTION
    StackTens(const StackTens& oth)
    {
      if constexpr(std::is_class_v<_Fund>)
	for(std::decay_t<decltype(nElements)> iEl=0;iEl<nElements;iEl++)
	  new(&storage[iEl]) _Fund(oth.storage[iEl]);
      else
	std::copy(oth.storage,oth.storage+nElements,storage);
    }
    
    /// Construct from another node
    template <typename TOth>
    constexpr INLINE_FUNCTION
    StackTens(const Node<TOth>& _oth)
    {
      const auto& oth=DE_CRTPFY(const TOth,&_oth);
      
      // if constexpr(std::is_class_v<_Fund>)
	loopOnAllComps<Comps>({},
			      [this,&oth](const auto&...c) CONSTEXPR_INLINE_ATTRIBUTE
			      {
				const auto cs=tupleGetSubset<typename TOth::Comps>(std::make_tuple(c...));
				
				new(&(*this)(c...)) _Fund(std::apply(oth,cs));
			      });
      // else
      // 	(*this)=DE_CRTPFY(const TOth,&oth);
    }
    
    /// Construct from fundamental
    constexpr INLINE_FUNCTION
    StackTens(const Fund& oth)
    {
      loopOnAllComps<Comps>({},
			    [this,&oth](const auto&...c) CONSTEXPR_INLINE_ATTRIBUTE
			    {
			      new(&(*this)(c...)) _Fund(oth);
			    });
    }
    
    /// Construct from an invocable
    template <typename F,
	      ENABLE_THIS_TEMPLATE_IF(not isNode<F> and std::is_invocable_v<F,C...>)>
    constexpr INLINE_FUNCTION
    explicit StackTens(// InitializerFunction,
		       F f) : storage{}
    {
      loopOnAllComps<Comps>({},[this,f](const auto&...c) CONSTEXPR_INLINE_ATTRIBUTE
      {
	new(&((*this)(c...))) _Fund(f(c...));
      });
    }
    
    /// Initialize from list
    template <typename...Tail>
    constexpr INLINE_FUNCTION
    StackTens(const Fund& first,const Tail&...tail) :
      storage{first,(Fund)tail...}
    {
    }
  };
}

#endif
