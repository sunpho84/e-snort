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
  // template <ExecutionSpace LhsSpace,
  // 	    ExecutionSpace RhsSpace,
  // 	    WhichSideToChange WhichSide>
  // struct Assign;
  
  // template <WhichSideToChange W>
  // struct Assign<EXEC_HOST,EXEC_HOST,W>
  // {
  using ExecHost=std::integral_constant<ExecutionSpace,EXEC_HOST>;
  using ExecDevice=std::integral_constant<ExecutionSpace,EXEC_DEVICE>;
  using ChangeLhsSide=std::integral_constant<WhichSideToChange,CHANGE_EXEC_SPACE_LHS_SIDE>;
  using ChangeRhsSide=std::integral_constant<WhichSideToChange,CHANGE_EXEC_SPACE_RHS_SIDE>;
  
  template <typename Lhs,
	    typename Rhs,
	    typename...W>
  static void exec(Lhs&& lhs,
		   Rhs&& rhs,
		   ExecHost,
		   ExecHost,
		   W...)
  {
      lhs()=rhs();
    }
  // };
  
  // template <WhichSideToChange W>
  // struct Assign<EXEC_DEVICE,EXEC_DEVICE,W>
  // {
    template <typename Lhs,
	      typename Rhs,
	      typename...W>
    static void exec(Lhs&& lhs,
		     Rhs&& rhs,
		     ExecDevice,
		     ExecDevice,
		     W...)
    {
#if !ENABLE_CUDA_CODE
      // Assign<EXEC_HOST,EXEC_HOST,CHANGE_EXEC_SPACE_LHS_SIDE>::
      exec(std::forward<Lhs>(lhs),std::forward<Rhs>(rhs),ExecHost{},ExecHost{},ChangeLhsSide{});
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
  // };
  
  // template <>
  // struct Assign<EXEC_HOST,EXEC_DEVICE,CHANGE_EXEC_SPACE_RHS_SIDE>
  // {
    template <typename Lhs,
	      typename Rhs>
    static void exec(Lhs&& lhs,
		     Rhs&& rhs,
		     ExecHost,
		     ExecDevice,
		     ChangeRhsSide)
    {
      printf("Copying to host the rhs\n");
      
      auto hostRhs=
	rhs.template changeExecSpaceTo<EXEC_HOST>();
      
      lhs()=hostRhs();
    }
  // };
  
  // template <>
  // struct Assign<EXEC_DEVICE,EXEC_HOST,CHANGE_EXEC_SPACE_RHS_SIDE>
  // {
    template <typename Lhs,
	      typename Rhs>
    static void exec(Lhs&& lhs,
		     Rhs&& rhs,
		     ExecDevice,
		     ExecHost,
		     ChangeRhsSide)
    {
      printf("Copying to device the rhs\n");
      
      const auto deviceRhs=
	rhs.template changeExecSpaceTo<EXEC_DEVICE>();
      
      lhs=deviceRhs;
      
      //       #if 0
//       Assign<EXEC_DEVICE,EXEC_DEVICE,CHANGE_EXEC_SPACE_RHS_SIDE>::exec(std::forward<Lhs>(lhs),deviceRhs);
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
  // };
}

#endif
