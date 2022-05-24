#ifndef _LATTICEGETTER_HPP
#define _LATTICEGETTER_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file lattice/latticeGetter.hpp

#include <lattice/field.hpp>

namespace grill
{
  /// Returns the lattice inside the argument list
  ///
  /// Forward declaration
  template <typename...Args>
  auto getLattice(Args&&...args);
  
  namespace internal
  {
    template <typename Arg>
    auto _getLattice(Arg&& arg)
    {
      if constexpr(isField<Arg>)
	return std::make_tuple(arg.lattice);
      else
	if constexpr(hasMember_subNodes<Arg>)
	  return std::apply([](const auto&...o)
	  {
	    return std::tuple_cat(_getLattice(o)...);
	  },arg.subNodes);
	else
	  return std::make_tuple();
    }
  }
  
  /// Returns the lattice inside the argument list
  template <typename...Args>
  auto getLattice(Args&&...args)
  {
    auto tmp=std::tuple_cat(internal::_getLattice(args)...);
    
    constexpr int nLattice=std::tuple_size_v<decltype(tmp)>;
    
    if constexpr(nLattice>1)
      {
	auto lattice=std::get<0>(tmp);
	
	forEachInTuple(tmp,[&lattice](const auto& o)
	{
	  static_assert(std::is_same_v<std::decay_t<decltype(o)>,decltype(lattice)>,
			"Found different lattice types");
	  
	  if(lattice!=o)
	    CRASH<<"Found different lattices";
	});
	
	return lattice;
      }
  }
}

#endif
