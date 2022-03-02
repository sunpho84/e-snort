#ifndef _THREADS_HPP
#define _THREADS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file threads.hpp

#if ENABLE_THREADS
# include <omp.h>
#endif

#include <resources/threadsHiddenVariablesDeclarations.hpp>
#include <metaprogramming/inline.hpp>

namespace esnort::threads
{
  void initialize();
  
  INLINE_FUNCTION bool isInsideParallelSection()
  {
#if ENABLE_THREADS
    return omp_in_parallel();
#else
    return false;
#endif
  }
  
  INLINE_FUNCTION int threadId()
  {
#if ENABLE_THREADS
    return omp_get_thread_num();
#else
    return 0;
#endif
  }
  
  INLINE_FUNCTION bool isMaster()
  {
#if ENABLE_THREADS
    return threadId()==0;
#else
    return true;
#endif
  }
  
  /// Wraps openmp lock
  struct Lock
  {
#if ENABLE_THREADS
    omp_lock_t lock;
#endif
    
    Lock()
    {
#if ENABLE_THREADS
      omp_init_lock(&lock);
#endif
    }
    
    void set()
    {
#if ENABLE_THREADS
      omp_set_lock(&lock);
#endif
    }
    
    void unset()
    {
#if ENABLE_THREADS
      omp_unset_lock(&lock);
#endif
    }
    
    ~Lock()
    {
#if ENABLE_THREADS
      omp_destroy_lock(&lock);
#endif
    }
  };
}


#endif
