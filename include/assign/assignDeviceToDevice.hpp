#ifndef _ASSIGNDEVICETODEVICE_HPP
#define _ASSIGNDEVICETODEVICE_HPP

#include <assign/assignBase.hpp>
#include <assign/assignHostToHost.hpp>

namespace esnort
{
  template <WhichSideToChange W>
  struct Assign<ExecutionSpace::DEVICE,ExecutionSpace::DEVICE,W>
  {
    template <typename Lhs,
	      typename Rhs>
    static void exec(Lhs&& lhs,
		     Rhs&& rhs)
    {
#if not ENABLE_DEVICE_CODE
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
}

#endif
