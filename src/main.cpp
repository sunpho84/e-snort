#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <cstdio>
#include <type_traits>

template <typename IMin,
	  typename IMax,
	  typename F>
CUDA_GLOBAL
void cuda_generic_kernel(const IMin min,
			 const IMax max,
			 F f)
{
#ifdef ENABLE_CUDA_CODE
  const auto i=
    min+blockIdx.x*blockDim.x+threadIdx.x;
  
  if(i<max)
    f(i);
#endif
}

inline void thread_barrier_internal()
{
#ifdef ENABLE_CUDA_CODE
  cudaDeviceSynchronize();
#endif
}

template <typename IMin,
	  typename IMax,
	  typename F>
void cuda_parallel_for(const int line,
		       const char *file,
		       const IMin min,
		       const IMax max,
		       F f)
{
#ifdef ENABLE_CUDA_CODE
  const auto length=(max-min);
  const dim3 block_dimension(128);
  int i=(length+block_dimension.x-1)/block_dimension.x;
  const dim3 grid_dimension(i);
  
  extern int verbosity_lv;
  printf("at line %d of file %s launching kernel on loop [%ld,%ld) using blocks of size %d and grid of size %d\n",
	 line,file,(int64_t)min,(int64_t)max,block_dimension.x,grid_dimension.x);

  cuda_generic_kernel<<<grid_dimension,block_dimension>>>(min,max,std::forward<F>(f));
  thread_barrier_internal();
  
  printf(" finished\n");
#endif
}

#ifdef __NVCC__
  #ifndef __CUDA_ARCH__
CUDA_HOST constexpr std::integral_constant<bool,false> isOnDevice() { return {}; }
  #else
CUDA_DEVICE constexpr std::integral_constant<bool,true> isOnDevice() { return {}; }
  #endif
#else
# ifdef ENABLE_CUDA_CODE
CUDA_HOST constexpr std::integral_constant<bool,false> isOnDevice() { return {}; }
CUDA_DEVICE constexpr std::integral_constant<bool,true> isOnDevice() { return {}; }
# else
  static constexpr std::integral_constant<bool,false> isOnDevice() { return {}; }
# endif
#endif

struct A
{
  double data[3]={};

#ifdef __NVCC__
  #ifndef __CUDA_ARCH__
CUDA_HOST static constexpr std::integral_constant<bool,false> isOnDevice() { return {}; }
  #else
CUDA_DEVICE static constexpr std::integral_constant<bool,true> isOnDevice() { return {}; }
  #endif
#else
# ifdef ENABLE_CUDA_CODE
  CUDA_HOST static constexpr std::integral_constant<bool,false> isOnDevice() { return {}; }
  CUDA_DEVICE static constexpr std::integral_constant<bool,true> isOnDevice() { return {}; }
# else
  static constexpr std::integral_constant<bool,false> isOnDevice() { return {}; }
# endif
#endif
  //   static constexpr auto isOnDevice() CUDA_HOST CUDA_DEVICE
  // {
  //   return ::isOnDevice();
  // }
};

//_CUDA_MANAGED
template <typename I>
static CUDA_MANAGED int iss;

template <typename T>
struct Ref
{
  const T& t;
  
  Ref(const T& t) : t(t)
  {
  }
};

int main()
{
  //iss<int> =1;
  
  A j;
  
  Ref ref(j);
   
 #if not COMPILING_FOR_DEVICE
  static_assert(not A::isOnDevice(),"We are issuing A on the host");
#endif
  printf("%d\n",A::isOnDevice()());
  
//   auto e=std::integral_constant<bool,A::isOnDevice()>{};
// #ifndef __CUDA_ARCH__
//   int* r=A::isOnDevice();
//  #endif
    
//  #ifndef __CUDA_ARCH__
//     int* r=(decltype(j)::isOnDevice());
// #endif
    cuda_parallel_for(__LINE__,__FILE__,0,12,[ref]  CUDA_DEVICE (const int i)
  {
    //int* r=(std::decay_t<decltype(ref.t)>::isOnDevice());
    // auto e=std::integral_constant<bool,A::isOnDevice()>{};
    //int* r=isOnDevice();
    // i+j.data[0];
  });
  
  return 0;
}
