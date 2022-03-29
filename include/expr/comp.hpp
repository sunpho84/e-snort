#ifndef _COMP_HPP
#define _COMP_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file comp.hpp

#include <expr/baseComp.hpp>
#include <expr/transposableComp.hpp>
#include <metaprogramming/detectableAs.hpp>

namespace esnort
{
  /// Compnent
  template <compFeat::IsTransposable IsTransposable,
	    typename Index,
	    typename Derived>
  struct Comp  :
    compFeat::Transposable<IsTransposable,Derived>,
    BaseComp<Derived,Index>
  {
    using Base=BaseComp<Derived,Index>;
    
    using Base::Base;
  };
  
  /// Predicate if a certain component has known size at compile time
  template <typename T>
  struct SizeIsKnownAtCompileTime
  {
    static constexpr bool value=T::sizeIsKnownAtCompileTime;
  };
  
  /// Promotes the argument i to a component of type TYPE
#define DECLARE_COMPONENT_FACTORY(FACTORY,TYPE)			\
  template <typename T>						\
  INLINE_FUNCTION constexpr HOST_DEVICE_ATTRIB			\
  TYPE FACTORY(T&& i)						\
  {								\
    return i;							\
  }
  
  PROVIDE_DETECTABLE_AS(TransposableComp);
  
#define DEFINE_TRANSPOSABLE_COMP(NAME,TYPE,SIZE,FACTORY)	\
  template <RwCl _RC=RwCl::ROW,					\
	    int _Which=0>					\
  struct NAME :							\
    DetectableAsTransposableComp,				\
    Comp<compFeat::IsTransposable::TRUE,			\
	 TYPE,							\
	 NAME<_RC,_Which>>					\
  {								\
    using Base=							\
      Comp<compFeat::IsTransposable::TRUE,			\
      TYPE,							\
      NAME<_RC,_Which>>;					\
    								\
    using Base::Base;						\
    								\
    static constexpr int sizeAtCompileTime=SIZE;		\
  };								\
  								\
  using NAME ## Row=NAME<RwCl::ROW,0>;				\
  								\
  using NAME ## Cln=NAME<RwCl::CLN,0>;				\
  								\
  /*! Transposed of a transposable component */			\
  template <RwCl RC,						\
	    int Which>						\
  INLINE_FUNCTION constexpr HOST_DEVICE_ATTRIB			\
  NAME<transpRwCl<RC>,Which>					\
  transp(const NAME<RC,Which>& c)				\
  {								\
    return c();							\
  }								\
  								\
  DECLARE_COMPONENT_FACTORY(FACTORY ## Row,NAME ## Row);	\
								\
  DECLARE_COMPONENT_FACTORY(FACTORY ## Cln,NAME ## Cln)

#define DEFINE_UNTRANSPOSABLE_COMP(NAME,TYPE,SIZE,FACTORY)	\
  struct NAME :							\
    Comp<compFeat::IsTransposable::FALSE,			\
	 TYPE,							\
	 NAME>							\
  {								\
    using Base=							\
      Comp<compFeat::IsTransposable::FALSE,			\
      TYPE,							\
      NAME>;							\
								\
    using Base::Base;						\
								\
    static constexpr int sizeAtCompileTime=SIZE;		\
  };								\
								\
  /*! Transposed of a non-transposable component */		\
  INLINE_FUNCTION constexpr HOST_DEVICE_ATTRIB			\
  const NAME& transp(const NAME& c)				\
  {								\
    return c;							\
  }								\
								\
  DECLARE_COMPONENT_FACTORY(FACTORY,NAME)
  
  /////////////////////////////////////////////////////////////////
  
  namespace internal
  {
    template <typename T,
	      ENABLE_THIS_TEMPLATE_IF(isTransposableComp<T>)>
    typename T::Transp* _transp();
    
    template <typename T,
	      ENABLE_THIS_TEMPLATE_IF(not isTransposableComp<T>)>
    T _transp();
  }
  
  template <typename T>
  using Transp=decltype(internal::_transp<T>());
}

#endif
