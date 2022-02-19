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
  struct Assign<ExecutionSpace::HOST,ExecutionSpace::HOST,W>
  {
    template <typename Lhs,
	      typename Rhs>
    static void exec(Lhs&& lhs,
		     Rhs&& rhs)
    {
      lhs()=rhs();
    }
  };
  
  template <WhichSideToChange W>
  struct Assign<ExecutionSpace::DEVICE,ExecutionSpace::DEVICE,W>
  {
    template <typename Lhs,
	      typename Rhs>
    static void exec(Lhs&& lhs,
		     Rhs&& rhs)
    {
#if !ENABLE_CUDA_CODE
      Assign<ExecutionSpace::HOST,ExecutionSpace::HOST,WhichSideToChange::LHS>::exec(std::forward<Lhs>(lhs),std::forward<Rhs>(rhs));
#else
      printf("Launching the kernel D to D\n");
      
      const dim3 block_dimension(1);
      const dim3 grid_dimension(1);
      
      auto devLhs=lhs.getRef();
      const auto devRhs=rhs.getRef();
      
      auto f=[=] CUDA_DEVICE (const int& i) mutable
      {
	return devLhs()=devRhs();
      };

#ifdef __NVCC__
      static_assert(__nv_is_extended_device_lambda_closure_type(decltype(f)),"");
#endif
      
      cuda_generic_kernel<<<grid_dimension,block_dimension>>>(0,1,f);
      
      cudaDeviceSynchronize();
      // #endif
#endif
    }
  };
  
  template <>
  struct Assign<ExecutionSpace::HOST,ExecutionSpace::DEVICE,WhichSideToChange::RHS>
  {
    template <typename Lhs,
	      typename Rhs>
    static void exec(Lhs&& lhs,
		     Rhs&& rhs)
    {
      printf("Copying to host the rhs\n");
      
      auto hostRhs=
	rhs.template changeExecSpaceTo<ExecutionSpace::HOST>();
      
      lhs()=hostRhs();
    }
  };
  
  template <>
  struct Assign<ExecutionSpace::DEVICE,ExecutionSpace::HOST,WhichSideToChange::RHS>
  {
    template <typename Lhs,
	      typename Rhs>
    static void exec(Lhs&& lhs,
		     Rhs&& rhs)
    {
      printf("Copying to device the rhs\n");
      
      const auto deviceRhs=
	rhs.template changeExecSpaceTo<ExecutionSpace::DEVICE>();
      
      lhs=deviceRhs;
      
      //       #if 0
//       Assign<ExecutionSpace::DEVICE,ExecutionSpace::DEVICE,CHANGE_EXEC_SPACE_RHS_SIDE>::exec(std::forward<Lhs>(lhs),deviceRhs);
//       #endif

//             printf("Launching the kernel\n");
      
//       const dim3 block_dimension(1);
//       const dim3 grid_dimension(1);
      
//       auto devLhs=lhs.getRef();
//       const auto devRhs=deviceRhs.getRef();
      
//       auto f=[=] CUDA_DEVICE (const int& i) mutable
//       {
// 	return devLhs()=devRhs();
//       };

// #ifdef __NVCC__
//       static_assert(__nv_is_extended_device_lambda_closure_type(decltype(f)),"");
// #endif
      
//       cuda_generic_kernel<<<grid_dimension,block_dimension>>>(0,1,f);
      
//       cudaDeviceSynchronize();

    }
  };
}

#endif
