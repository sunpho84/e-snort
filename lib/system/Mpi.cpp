#ifdef HAVE_CONFIG_H
# define DEFINE_HIDDEN_VARIABLES_ACCESSORS
# include "config.hpp"
#endif

/// \file Mpi.cpp

#include <system/Mpi.hpp>

namespace esnort
{
  namespace Mpi
  {
    /// Decrypt the returned value of an MPI call
    ///
    /// Returns the value of \c rc
    int crashOnError(const int line,        ///< Line of file where the error needs to be checked
		     const char *file,      ///< File where the error must be checked
		     const char *function,  ///< Function where the error was possibly raised
		     const int rc)          ///< Exit condition of the called routine
    {
#ifdef USE_MPI
      
      if(rc!=MPI_SUCCESS and rank==0)
	{
	  /// Length of the error message
	  int len;
	  
	  /// Error message
	  char err[MPI_MAX_ERROR_STRING];
	  MPI_Error_string(rc,err,&len);
	  
	  minimalCrash(file,line,__PRETTY_FUNCTION__,"(args ignored!), raised error %d, err: %s",rc,err);
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
      
      MPI_CRASH_ON_ERROR(durationOf(initDur,MPI_Init,nullptr,nullptr));
      
      MPI_CRASH_ON_ERROR(MPI_Comm_size(MPI_COMM_WORLD,&_nRanks));
      
      MPI_CRASH_ON_ERROR(MPI_Comm_rank(MPI_COMM_WORLD,&_rank));
      
      _isMasterRank=(rank==MASTER_RANK);
      
      minimalLogger("MPI initialized in %lg s, nranks: %d",durationInSec(initDur),nRanks);
      
      _inited=true;
      
#endif
    }
    
    /// Finalize MPI
    void finalize()
    {
#ifdef USE_MPI
      
      MPI_CRASH_ON_ERROR(MPI_Finalize());
      
      _inited=false;
      
#endif
    }
  }
}
