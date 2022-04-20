#ifndef _TUPLEHASTYPE_HPP
#define _TUPLEHASTYPE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file tupleHasType.hpp

#include <tuple>

namespace grill
{
  /// Predicate returning whether the type is present in the list
  ///
  /// Forward definition
  template <int N,
	    typename Tp>
  struct TypeIsInList;
  
  /// Predicate returning whether the type is present in the list
  template <int N,
	    typename...Tp>
  struct TypeIsInList<N,std::tuple<Tp...>>
  {
    /// Internal implementation
    template <typename T>
    struct t
    {
      /// Predicate result
      static constexpr bool value=((std::is_same<T,Tp>::value+...+0)==N);
    };
  };
  
  template <typename Tp,
	    typename T,
	    int N=1>
  constexpr bool tupleHasType=
    TypeIsInList<N,Tp>::template t<T>::value;
  
  template <typename Tp,
	    typename T,
	    int N=1>
  constexpr bool tupleHaveTypes=false;
  
  /// Returns whether the tuple has all the given types
  template <typename Tp,
	    typename...T,
	    int N>
  inline constexpr bool tupleHaveTypes<Tp,std::tuple<T...>,N> =
    (tupleHasType<Tp,T> and ... and true);
  
  namespace internal
  {
    /// Returns whether the two tuples T1 and T2 contains the same types
    ///
    /// Internal procedure forward declaration
    template <typename T1,
	      typename T2>
    struct _TuplesContainsSameTypes;
    
    /// Returns whether the two tuples T1 and T2 contains the same types
    ///
    /// Internal procedure
    template <typename...T1,
	      typename...T2>
    struct _TuplesContainsSameTypes<std::tuple<T1...>,std::tuple<T2...>>
    {
      /// Compares a single component
      template <typename T,
		typename...U>
      static constexpr int howManyTimeInList=
	(std::is_same_v<T,U>+...+0);
      
      /// Result
      static constexpr bool value=
	sizeof...(T1)==sizeof...(T2) and
	((howManyTimeInList<T1,T2...> ==1)&...);
    };
  }
  
  /// Returns whether the two tuples T1 and T2 contains the same types, no matter the order
  template <typename T1,
	    typename T2>
  constexpr bool tuplesContainsSameTypes=
    internal::_TuplesContainsSameTypes<T1,T2>::value;
}

#endif
