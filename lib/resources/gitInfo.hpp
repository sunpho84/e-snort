#ifndef _GITINFO_HPP
#define _GITINFO_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file gitInfo.hpp

namespace esnort::git
{
  extern const char* hash;
  
  extern const char* time;
  
  extern const char* committer;
  
  extern const char* log;
}

#endif
