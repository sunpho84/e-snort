#ifndef _TUPLEEXECFORALLTYPES_HPP
#define _TUPLEEXECFORALLTYPES_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file tupleExecForAllTypes.hpp

#include <metaprogramming/call.hpp>
#include <resources/device.hpp>

namespace grill
{
  template <typename TP,
	    typename F,
	    size_t...Is>
  HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
  void _execForAllTupleTypes(F&& f,
			     std::index_sequence<Is...>)
  {
    [[maybe_unused]]
    auto l=
      {grill::internal::call(f,((std::tuple_element_t<Is,TP>*)nullptr))...,0};
  }
  
#define EXEC_FOR_ALL_TUPLE_TYPES(T,TP,CORE...)			\
  _execForAllTupleTypes<TP>([&](auto* t) INLINE_ATTRIBUTE	\
  {								\
    using T=							\
      std::decay_t<decltype(*t)>;				\
    								\
    CORE;							\
  },std::make_index_sequence<std::tuple_size_v<TP>>())
  
  /////////////////////////////////////////////////////////////////
  
  template <typename TP,
	    typename F,
	    size_t...Is>
  HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
  void _execForAllTupleIds(F&& f,
			   std::index_sequence<Is...>)
  {
    [[maybe_unused]]
    auto l=
      {grill::internal::call(f,std::integral_constant<int,Is>())...,0};
  }
  
#define EXEC_FOR_ALL_TUPLE_IDS(I,TP,CORE...)			\
  _execForAllTupleIds<TP>([&](auto t) INLINE_ATTRIBUTE		\
  {								\
    static constexpr size_t I=					\
      std::decay_t<decltype(t)>::value;				\
    								\
    CORE;							\
  },std::make_index_sequence<std::tuple_size_v<TP>>())
}

#endif
