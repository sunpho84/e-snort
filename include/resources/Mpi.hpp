#ifndef _MPI_HPP
#define _MPI_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file Mpi.hpp
///
/// \brief Incapsulate all functionalities of MPI into a more
/// convenient form

#ifdef ENABLE_MPI
# include <mpi.h>
#endif

#include <ios/minimalLogger.hpp>
#include <metaprogramming/singleInstance.hpp>
#include <metaprogramming/templateEnabler.hpp>
#include <serialize/binarize.hpp>
#include <resources/MpiHiddenVariables.hpp>
#include <resources/timerGlobalVariablesDeclarations.hpp>

  /// Makes all thread print for current scope
#define ALLOWS_ALL_RANKS_TO_PRINT_FOR_THIS_SCOPE(LOGGER)		\
  SET_FOR_CURRENT_SCOPE(LOGGER_ALL_RANKS_PRINT,LOGGER.onlyMasterRankPrint,false)
  
  /// Crash on MPI error, providing a meaningful error
# define MPI_CRASH_ON_ERROR(ARGS...)					\
    Mpi::crashOnError(__LINE__,__FILE__,__PRETTY_FUNCTION__,ARGS)
  
namespace esnort::Mpi
{
  void initialize();
  
  void finalize();
  
  /// Decrypt the returned value of an MPI call
  ///
  /// Returns the value of \c rc
  int crashOnError(const int line,        ///< Line of file where the error needs to be checked
		   const char *file,      ///< File where the error must be checked
		   const char *function,  ///< Function where the error was possibly raised
		   const int rc,          ///< Exit condition of the called routine
		   const char* format,     ///< Additional error message
		   ...);
  
#ifdef ENABLE_MPI
  /// Provides the \c MPI_Datatype of an any unknown type
  template <typename T>
  inline MPI_Datatype type()
  {
    return
      nullptr;
  }
  
  /// Provides the \c MPI_Datatype of a given type
#define PROVIDE_MPI_DATATYPE(MPI_TYPE,TYPE)		\
  template <>						\
  inline MPI_Datatype type<TYPE>()			\
  {							\
    return						\
      MPI_TYPE;						\
  }
  
  PROVIDE_MPI_DATATYPE(MPI_CHAR,char);
  
  PROVIDE_MPI_DATATYPE(MPI_INT,int);
  
  PROVIDE_MPI_DATATYPE(MPI_DOUBLE,double);
#endif
  
  /// Id of master rank
  constexpr int MASTER_RANK=0;
  
  /// Placeholder for all ranks
  [[ maybe_unused ]]
  constexpr int ALL_RANKS=-1;
  
  /// Reduces among all MPI process
  template <typename T>
  T allReduce(const T& in)
  {
    
#ifdef ENABLE_MPI
    
    /// Result
    T out;
    
    minimalLogger("%p %d",&out,rank);
    
    MPI_CRASH_ON_ERROR(MPI_Allreduce(&in,&out,1,type<T>(),MPI_SUM,MPI_COMM_WORLD),"Reducing among all processes");
    
    return
      out;
    
#else
    
    return
      in;
    
#endif
    
  }
  
  /// Barrier among all MPI process
  inline void barrier()
  {
#ifdef ENABLE_MPI
    MPI_CRASH_ON_ERROR(MPI_Barrier(MPI_COMM_WORLD),"Waiting");
#endif
  }
  
  /// Broadcast among all MPI process
  ///
  /// This is a simple wrapper around the MPI_Bcast function
  template <typename T,
	    ENABLE_THIS_TEMPLATE_IF(std::is_trivially_copyable_v<T>)>
  void broadcast(T* x,                   ///< Quantity to broadcast
		 const size_t& size,     ///< Size of the quantity to broadcast
		 int root=MASTER_RANK)   ///< Rank from which to broadcast
  {
#ifdef ENABLE_MPI
    minimalLogger("%p %d",x,rank);
    MPI_CRASH_ON_ERROR(MPI_Bcast(x,size,MPI_CHAR,root,MPI_COMM_WORLD),"Broadcasting");
#endif
  }
  
  /// Broadcast among all MPI process
  ///
  /// Accepts trivially copyable structures
  template <typename T,
	    ENABLE_THIS_TEMPLATE_IF(std::is_trivially_copyable_v<T>)>
  void broadcast(T& x,                   ///< Quantity to broadcast
		 int root=MASTER_RANK)   ///< Rank from which to broadcast
  {
    broadcast(&x,sizeof(T),root);
  }
  
  /// Broadcast among all MPI process
  ///
  /// Accepts all binarizable classes
  template <typename T,
	    ENABLE_THIS_TEMPLATE_IF(isBinarizable<T>)>
  void broadcast(T&& val,                ///< Quantity to broadcast
		 int root=MASTER_RANK)   ///< Rank from which to broadcast
  {
    
#ifdef ENABLE_MPI
    Binarizer bin=
      val.binarize();
    
    broadcast(&*bin.begin(),bin.size(),root);
    
    val.deBinarize(bin);
#endif
  }
  
  /// Exec on all ranks sequentially
  template <typename F>
  void onAllRanksSequentiallyDo(const F& f)
  {
    for(int iRank=0;iRank<Mpi::nRanks;iRank++)
      {
	Mpi::barrier();
	
	if(iRank==Mpi::rank)
	  f(iRank);
      }
  }
}

#endif
