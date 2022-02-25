#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file esnort.cpp
///
/// \brief Implements the parts of code which require dedicated compilation units

#include <cstdarg>
#include <cstdio>

#include <ios/logger.hpp>
#include <resources/device.hpp>
#include <system/aliver.hpp>

// #include <Threads.hpp>
// #include <debug/Crash.hpp>
// #include <ios/Logger.hpp>
// #include <ios/MinimalLogger.hpp>
// #include <ios/TextFormat.hpp>
// #include <random/TrueRandomGenerator.hpp>
// #include <system/Memory.hpp>
// #include <system/Mpi.hpp>
// #include <system/Timer.hpp>
// #include <utility/Aliver.hpp>
// #include <utility/SingleInstance.hpp>

#ifndef CONFIG_TIME
 /// Null time
 #define CONFIG_TIME				\
//  ""
#endif

#ifndef CONFIG_FLAGS
 /// Null flags
 #define CONFIG_FLAGS				\
//   ""
#endif

namespace esnort
{
  
// #ifdef USE_THREADS
//   void ThreadPool::fill(const pthread_attr_t* attr)
//   {
//     {
//       ALLOWS_ALL_THREADS_TO_PRINT_FOR_THIS_SCOPE(runLog);
      
//       runLog()<<"Filling the thread pool with "<<nThreads<<" threads";
      
//       // Checks that the pool is not filled
//       if(isFilled)
// 	MINIMAL_CRASH("Cannot fill again the pool!");
      
//       // Resize the pool to contain all threads
//       pool.resize(nThreads,0);
      
//       // Marks the pool as filled, even if we are still filling it, this will keep the threads swimming
//       isFilled=
// 	true;
      
//       for(int threadId=1;threadId<nThreads;threadId++)
// 	{
// 	  //runLog()<<"thread of id "<<threadId<<" spwawned\n";
	  
// 	  // Allocates the parameters of the thread
// 	  ThreadPars* pars=
// 	    new ThreadPars{this,threadId};
	  
// 	  if(pthread_create(&pool[threadId],attr,threadPoolSwim,pars)!=0)
// 	    MINIMAL_CRASH_STDLIBERR("creating the thread");
// 	}
      
//       waitPoolToBeFilled(masterThreadId);
//     }
    
//     // Marks the pool is waiting for job to be done
//     isWaitingForWork=
//       true;
//   }
// #endif // USE_THREADS
  
//   int aliverHelper()
//   {
//     return
//       0;
//   }
  
  /// Global timings
  Timer timings("Total time",Timer::NO_FATHER,Timer::UNSTOPPABLE);
  
  Logger Logger::fakeLogger("/dev/null");
  
  /// Global logger
  Logger runLog("/dev/stdout");
  
  /// Global MPI
  Mpi mpi;
  
//   /// Global thrads
//   ThreadPool threads;
  
//   /// Global true random generator
//   TrueRandomGenerator trueRandomGenerator;
  
//   /// Memory manager
//   Memory memory;
  
  Device device;
  
  Aliver aliver;
}

