#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file gdbAttach.hpp

#include <unistd.h>

#define EXTERN_GDB_ATTACH
# include <debug/gdbAttach.hpp>

#include <system/Mpi.hpp>
#include <ios/logger.hpp>

namespace esnort
{
  void waitToAttachDebugger()
  {
    /// Flag used to trap
    volatile int flag=0;
    
    printf("Entering debug loop on rank %d, flag has address %p please type:\n"
	   "$ gdb -p %d\n"
	   "$ set flag=1\n"
	   "$ continue\n",
	   (int)(Mpi::rank()),
	   &flag,
	   getpid());
    
    if(thisRank()==0)
      while(flag==0);
    
    ranksBarrier();
  }
  
  void possiblyWaitToAttachDebugger()
  {
    /// String used to detect debugging directive
    const char DEBUG_LOOP_STRING[]="WAIT_TO_ATTACH_DEBUGGER";
    
    if(getenv(DEBUG_LOOP_STRING)!=NULL)
      waitToAttachDebugger();
    else
      LOGGER<<"To wait attaching the debugger please export: "<<DEBUG_LOOP_STRING<<"=1"<<endl;
  }
}
