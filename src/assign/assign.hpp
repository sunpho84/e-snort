#ifndef _ASSIGN_HPP
#define _ASSIGN_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <cstdio>
#include <utility>

#if ENABLE_CUDA_CODE
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
#if !ENABLE_CUDA_CODE
      Assign<EXEC_HOST,EXEC_HOST,CHANGE_EXEC_SPACE_LHS_SIDE>::exec(std::forward<Lhs>(lhs),std::forward<Rhs>(rhs));
#else
      printf("Launching the kernel\n");
      
      const dim3 block_dimension(1);
      const dim3 grid_dimension(1);
      auto f=[lhs=lhs.getRef(),rhs=rhs.getRef(),*this] CUDA_DEVICE ()
      {
	//lhs()=rhs();
      };
      
      static_assert(__nv_is_extended_device_lambda_closure_type(decltype(f)),"");
      
      cuda_generic_kernel<<<grid_dimension,block_dimension>>>(0,1,f);
      
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
      
      Assign<EXEC_DEVICE,EXEC_DEVICE,CHANGE_EXEC_SPACE_RHS_SIDE>::exec(std::forward<Lhs>(lhs),deviceRhs);
    }
  };
}

#endif
