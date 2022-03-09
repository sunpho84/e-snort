#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file signalTrap.cpp

#include <cstdio>
#include <signal.h>

#include <debug/backtracing.hpp>
#include <debug/minimalCrash.hpp>

namespace esnort
{
  void signalHandler(int sig)
  {
    //master_printf("maximal memory used: %ld\n",max_required_memory);
    //verbosity_lv=3;
    char name[100];
    switch(sig)
      {
      case SIGSEGV: sprintf(name,"segmentation violation");break;
      case SIGFPE: sprintf(name,"floating-point exception");break;
      case SIGXCPU: sprintf(name,"cpu time limit exceeded");break;
      case SIGBUS: sprintf(name,"bus error");break;
      case SIGINT: sprintf(name," program interrupted");break;
      case SIGABRT: sprintf(name,"abort signal");break;
      default: sprintf(name,"unassociated");break;
      }
    
    printBacktraceList();
    
    //print_all_vect_content();
    MINIMAL_CRASH("signal %d (%s) detected, exiting",sig,name);
  }
  
  void setSignalTraps()
  {
    signal(SIGBUS,signalHandler);
    signal(SIGSEGV,signalHandler);
    signal(SIGFPE,signalHandler);
    signal(SIGXCPU,signalHandler);
    signal(SIGABRT,signalHandler);
    signal(SIGINT,signalHandler);
  }
  
}
