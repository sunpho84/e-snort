#ifndef _SUBNODES_HPP
#define _SUBNODES_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <tuple>

#include <expr/exprRefOrVal.hpp>
#include <metaprogramming/hasMember.hpp>

/// \file expr/subNodes.hpp
///
/// \brief Type to hold subnodes

namespace grill
{
  PROVIDE_HAS_MEMBER(subNodes);
  
  /// Holds the subnodes
  template <typename..._E>
  struct SubNodes
  {
    using type=std::tuple<NodeRefOrVal<_E>...>;
    
    /// Subnodes
    type subNodes;
    
#define PROVIDE_SUBNODE(ATTRIB)				\
    /*! Proxy for the I-subexpression */		\
    template <int I>					\
    INLINE_FUNCTION constexpr HOST_DEVICE_ATTRIB	\
    decltype(auto) subNode() ATTRIB			\
    {							\
      return std::get<I>(subNodes);			\
    }
    
    PROVIDE_SUBNODE(const);
    
    PROVIDE_SUBNODE(/* non const */);
    
#undef PROVIDE_SUBNODE
    
    /// Type of the I-th subnode
    template <int I>
    using SubNode=
      std::decay_t<std::tuple_element_t<I,std::tuple<_E...>>>;
    
    /// Constructor
    template <typename...T>
    INLINE_FUNCTION constexpr HOST_DEVICE_ATTRIB
    SubNodes(T&&...t) :
      subNodes(std::forward<T>(t)...)
    {
    }
  };
  
  /// Access subnodes
#define SUBNODE(I)				\
  this->template subNode<I>()

#define IMPORT_SUBNODE_TYPES				\
  /*! Import subnodes type */				\
  template <int I>					\
  using SubNode=					\
    typename SubNodes<_E...>::template SubNode<I>
}

#endif
