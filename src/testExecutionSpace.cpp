#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <cstdio>
#include <unistd.h>

template <typename F>
__global__
void cuda_generic_kernel(F f)
{
  f();
}

enum ExecutionSpace{EXEC_HOST,EXEC_DEVICE,EXEC_UNDEFINED};
enum ExecutionSpaceChangeCost{EXEC_SPACE_CHANGE_COSTS_NOTHING,EXEC_SPACE_CHANGE_COSTS_LITTLE,EXEC_SPACE_CHANGE_COSTS_ALOT};

constexpr ExecutionSpace currentExecSpace()
{
 return
#ifdef __CUDA_ARCH__
   EXEC_DEVICE
#else
   EXEC_HOST
#endif
   ;
}

/// Implements the CRTP pattern
template <typename T>
struct Crtp
{
#define PROVIDE_CRTP(ATTRIB)				\
  /*! Crtp access the type */				\
  __host__ __device__ inline constexpr			\
  ATTRIB T& crtp() ATTRIB				\
  {							\
    return						\
      *static_cast<ATTRIB T*>(this);			\
  }
  
  PROVIDE_CRTP(const);
  
  PROVIDE_CRTP(/* not const*/ );
  
#undef PROVIDE_CRTP
};

enum ExecutionSpaceAssignmentType{ASSIGN_MATCHING_SPACE,ASSIGN_MISMATCHING_SPACE};

enum WhichSideToChange{CHANGE_EXEC_SPACE_LHS_SIDE,CHANGE_EXEC_SPACE_RHS_SIDE};

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
		   Rhs&& rhs) __host__
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
		   Rhs&& rhs) __host__
  {
// #ifndef __CUDA_ARCH__
//     fprintf(stderr,"");
//     exit(1);
// #else
    const dim3 block_dimension(1);
    const dim3 grid_dimension(1);
    cuda_generic_kernel<<<grid_dimension,block_dimension>>>([lhs,rhs] __device__ () mutable
    {
#ifdef __CUDA_ARCH__
      lhs()=rhs();
#endif
    });
    cudaDeviceSynchronize();
// #endif
  }
};

template <>
struct Assign<EXEC_HOST,EXEC_DEVICE,CHANGE_EXEC_SPACE_RHS_SIDE>
{
  template <typename Lhs,
	    typename Rhs>
  static void exec(Lhs&& lhs,
		   Rhs&& rhs) __host__
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
    
    Assign<EXEC_DEVICE,EXEC_DEVICE,CHANGE_EXEC_SPACE_RHS_SIDE>::exec(std::forward<Lhs>(lhs),deviceRhs);
  }
};

template <typename T>
struct Expr :
  Crtp<T>
{
  template <typename U>
  T& operator=(const Expr<U>& u)
  {
    decltype(auto) lhs=this->crtp();
    decltype(auto) rhs=u.crtp();
    
    using Lhs=std::decay_t<decltype(lhs)>;
    using Rhs=std::decay_t<decltype(rhs)>;
    
    constexpr ExecutionSpace lhsExecSpace=Lhs::execSpace();
    constexpr ExecutionSpace rhsExecSpace=Rhs::execSpace();
    
    constexpr WhichSideToChange whichSideToChange=
		(Rhs::execSpaceChangeCost()>Lhs::execSpaceChangeCost())?
		CHANGE_EXEC_SPACE_LHS_SIDE:
                CHANGE_EXEC_SPACE_RHS_SIDE;
    
    Assign<lhsExecSpace,rhsExecSpace,whichSideToChange>::exec(lhs,rhs);
    
    this->crtp()()=u.crtp()();
    
    return this->crtp();
  }
};

template <typename T,
	  ExecutionSpace ExecSpace>
struct DynamicVariable :
  Expr<DynamicVariable<T,ExecSpace>>
{
  using Expr<DynamicVariable<T,ExecSpace>>::operator=;
  
  static constexpr ExecutionSpace execSpace()
  {
    return ExecSpace;
  }
  
  static constexpr ExecutionSpaceChangeCost execSpaceChangeCost()
  {
    return EXEC_SPACE_CHANGE_COSTS_LITTLE;
  }
  
  T* ptr;
  
  const T& operator()() const __host__ __device__
  {
    return *ptr;
  }
  
  T& operator()() __host__ __device__
  {
    return *ptr;
  }
  
  DynamicVariable()
  {
    if(execSpace()==EXEC_DEVICE)
      cudaMalloc(&ptr,sizeof(T));
    else
      ptr=new T;
  }
  
  ~DynamicVariable()
  {
    if(execSpace()==EXEC_DEVICE)
      cudaFree(ptr);
    else
      delete ptr;
  }
  
  template <ExecutionSpace OthExecSpace>
  decltype(auto) changeExecSpaceTo() const
  {
    if constexpr(OthExecSpace==execSpace())
      return *this;
    else
      {
	DynamicVariable<T,OthExecSpace> res;
	cudaMemcpy(res.ptr,
		   ptr,
		   sizeof(T),
		   OthExecSpace==EXEC_DEVICE?
		   cudaMemcpyHostToDevice:
		   cudaMemcpyDeviceToHost);
	return res;
      }
  }
};

template <typename T>
struct StackedVariable :
    Expr<StackedVariable<T>>
{
  using Expr<StackedVariable<T>>::operator=;
  
  static constexpr ExecutionSpace execSpace()
  {
    return currentExecSpace();
  }
  
  static constexpr ExecutionSpaceChangeCost execSpaceChangeCost()
  {
    return EXEC_SPACE_CHANGE_COSTS_LITTLE;
  }
  
  template <ExecutionSpace OthExecSpace>
  decltype(auto) changeExecSpaceTo() const
  {
    if constexpr(OthExecSpace==execSpace())
      return *this;
    else
      {
	DynamicVariable<T,OthExecSpace> res;
	cudaMemcpy(res.ptr,
		   &value,
		   sizeof(T),
		   OthExecSpace==EXEC_DEVICE?
		   cudaMemcpyHostToDevice:
		   cudaMemcpyDeviceToHost);
	return res;
      }
  }
  
  T value;
  
  const T& operator()() const __host__ __device__
  {
    return value;
  }
  
  T& operator()() __host__ __device__
  {
    return value;
  }
};

int main()
{
  StackedVariable<int> a;
  a()=1;
  
  DynamicVariable<int,EXEC_DEVICE> c;
  c=a;
  StackedVariable<int> b;
  b=c;
  auto d=c.changeExecSpaceTo<EXEC_HOST>();
  c.changeExecSpaceTo<EXEC_HOST>();
  
  return 0;
}
