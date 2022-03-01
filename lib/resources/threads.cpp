#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file device.cpp

#if ENABLE_THREADS
# include <omp.h>
#endif

#define DEFINE_HIDDEN_VARIABLES_ACCESSORS
# include <resources/threadsHiddenVariablesDeclarations.hpp>
#undef DEFINE_HIDDEN_VARIABLES_ACCESSORS

#include <ios/logger.hpp>

namespace esnort::threads
{
  void initialize()
  {
#if ENABLE_THREADS
    _nThreads=omp_get_num_threads();
#endif
    
    logger()<<"NThreads: "<<nThreads;
  }
}
