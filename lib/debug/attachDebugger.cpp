#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file gdbAttach.hpp

#include <unistd.h>

#include <debug/attachDebuggerGlobalVariablesDeclarations.hpp>
#include <ios/logger.hpp>
#include <resources/Mpi.hpp>

namespace esnort
{
  void possiblyWaitToAttachDebugger()
  {
    if(esnort::waitToAttachDebugger)
      {
	/// Flag used to trap
	volatile int flag=0;
	
	SCOPE_ALL_RANKS_CAN_PRINT();
	
	Mpi::onAllRanksSequentiallyDo([&flag](const int& iRank)
	{
	  LOGGER<<"Entering debug loop on rank "<<(int)(Mpi::rank)<<", flag has address "<<&flag<<" please type:\n"
	    "$ gdb -p "<<getppid()<<"\n"
	    "$ set flag=1\n"
	    "$ continue\n";
	});
	  
	if(Mpi::rank==0)
	  while(flag==0);
	
	Mpi::barrier();
      }
  }
}
