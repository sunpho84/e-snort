#ifndef _TUPLEFIRSTOCCURRENCE_HPP
#define _TUPLEFIRSTOCCURRENCE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file tupleFirstOccurrence.hpp

#include <cstdlib>
#include <tuple>

namespace grill
{
  /// Inspects a tuple
  ///
  /// Forward declaration
  template <typename TP>
  struct TupleInspect;
    
  /// Inspect a tuple
  template <typename...TPs>
  struct TupleInspect<std::tuple<TPs...>>
  {
    /// Returns the first occurrence of the type
    ///
    /// Internal implementation
    template <typename T>
    static constexpr size_t _firstOccurrenceOfType()
    {
      /// Compare the passed type
      constexpr bool is[]=
	{std::is_same_v<TPs,T>...};
      
      /// Returned position
      size_t pos=0;
      
      while(pos<sizeof...(TPs) and
	      not is[pos])
	pos++;
      
      return
	pos;
    }
    
    /// Returns the first occurrence of the type
    template <typename T>
    static constexpr size_t firstOccurrenceOfType=
      _firstOccurrenceOfType<T>();
    
    /// Returns the first occurrence of the types
    template <typename...T>
    using FirstOccurrenceOfTypes=
      std::index_sequence<firstOccurrenceOfType<T>...>;
    
    /// Returns the first occurrence of the types incapsulated in the tuple
    ///
    /// Internal implementation
    template <typename...OTPs>
    static constexpr auto _firstOccurrenceOfTupleTypes(std::tuple<OTPs...>*)
    {
      return
	FirstOccurrenceOfTypes<OTPs...>{};
    }
    
    /// Returns the first occurrence of the types incapsulated in the tuple
    template <typename OTP>
    using FirstOccurrenceOfTupleTypes=
      decltype(_firstOccurrenceOfTupleTypes((OTP*)nullptr));
  };
  
  /// Returns the first occurrence of the first type in the argument tuple
  template <typename T,
	    typename Tp>
  constexpr size_t firstOccurrenceOfTypeInTuple=
    TupleInspect<Tp>::template firstOccurrenceOfType<T>;
  
  /// Returns the first occurrence of the first type in the list
  template <typename T,
	    typename...Tp>
  constexpr size_t firstOccurrenceOfTypeInList=
    TupleInspect<std::tuple<Tp...>>::template firstOccurrenceOfType<T>;
  
  /// Returns the first occurrence of the first list of types in the argument tuple
  template <typename TupleTypesToSearch,
	    typename TupleToInspect>
  using FirstOccurrenceOfTypes=
    typename TupleInspect<TupleToInspect>::template FirstOccurrenceOfTupleTypes<TupleTypesToSearch>;
}

#endif
