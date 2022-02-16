#ifndef _ASSIGN_HPP
#define _ASSIGN_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <cstdio>
#include <utility>

#ifdef ENABLE_CUDA_CODE
# include <cuda/cuda.hpp>
#endif
#include <expr/executionSpace.hpp>

namespace esnort
{
  template <ExecutionSpace LhsSpace,
	    ExecutionSpace RhsSpace,
	    WhichSideToChange WhichSide>
  struct Assign;
  
  template <WhichSideToChange W>
  struct Assign<EXEC_HOST,EXEC_HOST,W>
  {
    template <typename Lhs,
	      typename Rhs>
    static void exec(Lhs&& lhs,
		     Rhs&& rhs) CUDA_HOST
    {
      lhs()=rhs();
    }
  };
  
  template <WhichSideToChange W>
  struct Assign<EXEC_DEVICE,EXEC_DEVICE,W>
  {
    template <typename Lhs,
	      typename Rhs>
    static void exec(Lhs&& lhs,
		     Rhs&& rhs) CUDA_HOST
    {
#ifndef ENABLE_CUDA_CODE
      Assign<EXEC_HOST,EXEC_HOST,CHANGE_EXEC_SPACE_LHS_SIDE>::exec(std::forward<Lhs>(lhs),std::forward<Rhs>(rhs));
#else
      // #ifndef __CUDA_ARCH__
      //     fprintf(stderr,"");
      //     exit(1);
      // #else
      const dim3 block_dimension(1);
      const dim3 grid_dimension(1);
      cuda_generic_kernel<<<grid_dimension,block_dimension>>>([lhs,rhs] CUDA_DEVICE () mutable
      {
	lhs()=rhs();
      });
      
      cudaDeviceSynchronize();
      // #endif
#endif
    }
  };
  
  template <>
  struct Assign<EXEC_HOST,EXEC_DEVICE,CHANGE_EXEC_SPACE_RHS_SIDE>
  {
    template <typename Lhs,
	      typename Rhs>
    static void exec(Lhs&& lhs,
		     Rhs&& rhs) CUDA_HOST
    {
      lhs()=rhs.template changeExecSpaceTo<EXEC_HOST>()();
    }
  };
  
  template <>
  struct Assign<EXEC_DEVICE,EXEC_HOST,CHANGE_EXEC_SPACE_RHS_SIDE>
  {
    template <typename Lhs,
	      typename Rhs>
    static void exec(Lhs&& lhs,
		     Rhs&& rhs)
    {
      auto deviceRhs=rhs.template changeExecSpaceTo<EXEC_DEVICE>();
      
      printf("Copying to device\n");
      
      //Assign<EXEC_DEVICE,EXEC_DEVICE,CHANGE_EXEC_SPACE_RHS_SIDE>::exec(std::forward<Lhs>(lhs),deviceRhs);
    }
  };
}

#endif
