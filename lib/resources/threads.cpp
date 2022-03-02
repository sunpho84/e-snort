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
  const char* translateProcBindingToString(const omp_proc_bind_t& binding)
  {
    switch(binding)
      {
      case omp_proc_bind_close:
	return "close";
	break;
      case omp_proc_bind_master:
	return "master";
	break;
      case omp_proc_bind_false:
	return "false";
	break;
      case omp_proc_bind_true:
	return "true";
	break;
      case omp_proc_bind_spread:
	return "spread";
	break;
      }
    
    return "";
  }
  
  void initialize()
  {
#if ENABLE_THREADS
# pragma omp parallel
#  pragma omp master
    _nThreads=omp_get_num_threads();
#endif
    
    logger()<<"NThreads: "<<nThreads;
    
#if ENABLE_THREADS
    {
      SCOPE_INDENT();
      logger()<<"Threads <-> Core binding: "<<translateProcBindingToString(omp_get_proc_bind());
      logger()<<"Threads dynamicity: "<<(omp_get_dynamic()?"true":"false");
    }
#endif
  }
}
