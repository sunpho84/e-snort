#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file Mpi.cpp

#define DEFINE_HIDDEN_VARIABLES_ACCESSORS
# include <resources/MpiHiddenVariables.hpp>
#undef DEFINE_HIDDEN_VARIABLES_ACCESSORS

#include <resources/Mpi.hpp>

namespace esnort::Mpi
{
  int crashOnError(const int line,
		   const char *file,
		   const char *function,
		   const int rc,
		   const char* mess)
  {
#ifdef USE_MPI
    
    if(rc!=MPI_SUCCESS and rank==0)
      {
	/// Length of the error message
	int len;
	
	/// Error message
	char err[MPI_MAX_ERROR_STRING];
	MPI_Error_string(rc,err,&len);
	
	minimalCrash(file,line,__PRETTY_FUNCTION__,"%s, raised error %d, err: %s",mess,rc,err);
      }
    
#endif
    
    return rc;
  }
  
  /// Initialize MPI
  void initialize()
  {
#ifdef USE_MPI
    
    /// Takes the time
    Duration initDur;
    
    MPI_CRASH_ON_ERROR(durationOf(initDur,MPI_Init,nullptr,nullptr),"Initializing");
    
    MPI_CRASH_ON_ERROR(MPI_Comm_size(MPI_COMM_WORLD,&_nRanks),"Getting the number of ranks");
    
    MPI_CRASH_ON_ERROR(MPI_Comm_rank(MPI_COMM_WORLD,&_rank),"Getting the rank");
    
    _isMaster=(rank==MASTER_RANK);
    
    minimalLogger("MPI initialized in %lg s, nranks: %d",durationInSec(initDur),nRanks);
    
    _inited=true;
    
#endif
  }
  
  /// Finalize MPI
  void finalize()
  {
#ifdef USE_MPI
    
    MPI_CRASH_ON_ERROR(MPI_Finalize(),"Finalizing");
    
    _inited=false;
    
#endif
  }
}
