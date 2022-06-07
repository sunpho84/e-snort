#ifndef _FIELD_LAYOUT_GETTER_HPP
#define _FIELD_LAYOUT_GETTER_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file lattice/fieldLayoutGetter.hpp

#include <expr/nodes/subNodes.hpp>
#include <lattice/fieldDeclaration.hpp>
#include <tuples/tupleExecForAllTypes.hpp>

namespace grill
{
  template <typename Arg>
  constexpr FieldLayout getFieldLayout();
  
  namespace internal
  {
    template <typename...T>
    constexpr auto _tupleGetFieldLayout(const std::tuple<T...>*)
    {
      FieldLayout res=FieldLayout::SERIAL;
      bool found=true;
      bool inited=false;
      
      for(FieldLayout lc : {getFieldLayout<std::decay_t<T>>()...})
	{
	  if(inited==false)
	    res=lc;
	  else
	    if(res!=lc)
	      found=false;
	  
	  inited=true;
	}
      
      return std::make_pair(found,res);
    }
  }
  
  /// Returns the field layout inside the argument list
  template <typename Arg>
  constexpr FieldLayout getFieldLayout()
  {
    if constexpr(hasMember_fieldLayout<Arg>)
      return Arg::fieldLayout;
    else
      {
	static_assert(hasMember_subNodes<Arg>,"Not field layout found so far, no subnodes present");
	constexpr std::pair<bool,FieldLayout> res=internal::_tupleGetFieldLayout((typename Arg::SubNodes::type*)nullptr);
	static_assert(res.first,"No field layout found");
	
	return res.second;
      }
  }
}

#endif
