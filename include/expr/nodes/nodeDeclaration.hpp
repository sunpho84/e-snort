#ifndef _NODEDECLARATION_HPP
#define _NODEDECLARATION_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/nodes/nodeDeclaration.hpp
///
/// \brief Declares a node in the syntactic tree

#include <metaprogramming/detectableAs.hpp>

namespace esnort
{
  /// Base type representing a node
  template <typename T>
  struct Node;
  
  PROVIDE_DETECTABLE_AS(Node);
}

#endif
