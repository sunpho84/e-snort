#ifndef _NODECLOSER_HPP
#define _NODECLOSER_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/nodes/nodeCloser.hpp
///
/// \brief Close the expression into an appropriate tensor or field

#include <expr/nodes/node.hpp>
#include <lattice/fieldLayoutGetter.hpp>
#include <lattice/latticeCoverageGetter.hpp>
#include <metaprogramming/inline.hpp>

namespace grill
{
  /// Close the expression into an appropriate tensor or field
  template <typename T>
  INLINE_FUNCTION
  auto Node<T>::close() const
  {
    static_assert(not std::is_void_v<decltype(getLattice(DE_CRTPFY(const T,this)))>,"Not implemented when closing to something other than field");

    //if constexpr(not std::is_void_v<decltype(getLattice(DE_CRTPFY(const T,this)))>)
      {
	auto lattice=*getLattice(DE_CRTPFY(const T,this));
	
	using L=std::decay_t<decltype(lattice)>;
	
	constexpr LatticeCoverage LC=getLatticeCoverage<T>();
	
	constexpr FieldLayout FL=getFieldLayout<T>();
	
	using Comps=TupleFilterAllTypes<typename T::Comps,typename L::SpaceTimeComps>;
	
	using Res=Field<Comps,typename T::Fund,L,LC,FL,T::execSpace>;
	
	return closeAs<Res>(lattice);
      }
    // DynamicTens<typename T::Comps,typename T::Fund,T::execSpace> res(DE_CRTPFY(const T,this).getDynamicSizes());
    
    // res=*this;
    
    // return res;
  }
}

#endif
