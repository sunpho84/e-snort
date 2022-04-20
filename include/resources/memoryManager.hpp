#ifndef _MEMORY_MANAGER_HPP
#define _MEMORY_MANAGER_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file memoryManager.hpp

#include <map>
#include <vector>
#include <cstdint>

#include <expr/assign/executionSpace.hpp>
#include <ios/logger.hpp>
#include <metaprogramming/crtp.hpp>
#include <resources/device.hpp>
#include <resources/memoryGlobalVariablesDeclarations.hpp>
#include <resources/valueWithExtreme.hpp>

namespace grill
{
#define MEMORY_LOGGER				\
  VERBOSE_LOGGER(3)
  
  /// Type used for size
  using Size=int64_t;
  
  /// Minimal alignment
#define DEFAULT_ALIGNMENT 16
  
  /// Memory manager
  template <ExecSpace ES>
  class MemoryManager
  {
  protected:
    
    /// Number of allocation performed
    Size nAlloc{0};
    
  private:
    
    /// Store initialization
    bool inited;
    
    /// Name of the memory manager
    const char* const name;
    
    /// List of dynamical allocated memory
    std::map<void*,Size> used;
    
    /// List of cached allocated memory
    std::map<Size,std::vector<void*>> cached;
    
    /// Size of used memory
    ValWithMax<Size> usedSize;
    
    /// Size of cached memory
    ValWithMax<Size> cachedSize;
    
    /// Number of cached memory reused
    Size nCachedReused{0};
    
    /// Add to the list of used memory
    void pushToUsed(void* ptr,
		    const Size& size)
    {
      used[ptr]=size;
      
      usedSize+=size;
      
      //verbosity_lv3_master_printf("Pushing to used %p %zu, used: %zu\n",ptr,size,used.size());
    }
    
    /// Removes a pointer from the used list, without actually freeing associated memory
    ///
    /// Returns the size of the memory pointed
    Size popFromUsed(void* ptr) ///< Pointer to the memory to move to cache
    {
      //verbosity_lv3_master_printf("Popping from used %p\n",ptr);
      
      /// Iterator to search result
      auto el=used.find(ptr);
      
      if(el==used.end())
	CRASH<<"Unable to find dinamically allocated memory "<<ptr;
      
      /// Size of memory
      const Size size=el->second;
      
      usedSize-=size;
      
      used.erase(el);
      
      return size;
    }
    
    /// Adds a memory to cache
    void pushToCache(void* ptr,          ///< Memory to cache
		     const Size& size)    ///< Memory size
    {
      cached[size].push_back(ptr);
      
      cachedSize+=size;
      
      MEMORY_LOGGER<<"Pushing to cache "<<size<<" "<<ptr<<", cache size: "<<cached.size();
    }
    
    /// Check if a pointer is suitably aligned
    static bool isAligned(const void* ptr,
			  const Size& alignment)
    {
      return reinterpret_cast<uintptr_t>(ptr)%alignment==0;
    }
    
    /// Pop from the cache, returning to use
    void* popFromCache(const Size& size,
		       const Size& alignment)
    {
      MEMORY_LOGGER<<"Trying to find "<<size<<" bytes with alignment "<<alignment<<" in the cache ";
      
      /// List of memory with searched size
      auto cachedIt=cached.find(size);
      
      if(cachedIt==cached.end())
	{
	  MEMORY_LOGGER<<"Unable to find in the cache";
	  return nullptr;
	}
      else
	{
	  MEMORY_LOGGER<<"Found in the cache, popping it";
	  
	  /// Vector of pointers
	  auto& list=cachedIt->second;
	  
	  /// Get latest cached memory with appropriate alignment
	  auto it=list.end()-1;
	  
	  while(it!=list.begin()-1 and not isAligned(*it,alignment))
	    it--;
	  
	  if(it==list.begin()-1)
	    return nullptr;
	  else
	    {
	      /// Returned pointer, copied here before erasing
	      void* ptr=*it;
	      
	      list.erase(it);
	      
	      cachedSize-=size;
	      
	      if(list.size()==0)
		cached.erase(cachedIt);
	      
	      return ptr;
	    }
	}
    }
    
    /// Move the allocated memory to cache
    void moveToCache(void* ptr) ///< Pointer to the memory to move to cache
    {
      MEMORY_LOGGER<<"Moving to cache "<<ptr;
      
      /// Size of pointed memory
      const Size size=popFromUsed(ptr);
      
      pushToCache(ptr,size);
    }
    
  public:
    
    /// Enable cache usage
    void enableCache()
    {
      memory::useCache=true;
    }
    
    /// Disable cache usage
    void disableCache()
    {
      memory::useCache=false;
      
      clearCache();
    }
    
    /// Allocate or get from cache after computing the proper size
    template <class T>
    T* provide(const Size& nel,
	       const Size& alignment=DEFAULT_ALIGNMENT)
    {
      /// Total size to allocate
      const Size size=sizeof(T)*nel;
      
      /// Allocated memory
      void* ptr;
      
      // Search in the cache
      ptr=popFromCache(size,alignment);
      
      // If not found in the cache, allocate new memory
      if(ptr==nullptr)
	ptr=allocateRaw(size,alignment);
      else
	nCachedReused++;
      
      pushToUsed(ptr,size);
      
      return static_cast<T*>(ptr);
    }
    
    /// Declare unused the memory and possibly free it
    template <typename T>
    void release(T* &ptr) ///< Pointer getting freed
    {
      if(memory::useCache)
	moveToCache(static_cast<void*>(ptr));
      else
	{
	  popFromUsed(ptr);
	  this->deAllocateRaw(ptr);
	}
    }
    
    /// Release all used memory
    void releaseAllUsedMemory()
    {
      /// Iterator on elements to release
      auto el=
	used.begin();
      
      while(el!=used.end())
	{
	  MEMORY_LOGGER<<"Releasing "<<(el->first)<<" size "<<(el->second);
	  
	  /// Pointer to memory to release
	  void* ptr=el->first;
	  
	  // Increment iterator before releasing
	  el++;
	  
	  release(ptr);
	}
    }
    
    /// Release all memory from cache
    void clearCache()
    {
      MEMORY_LOGGER<<"";
      MEMORY_LOGGER<<"Clearing cache";
      
      /// Iterator to elements of the cached memory list
      auto el=cached.begin();
      
      while(el!=cached.end())
	{
	  /// Number of elements to free
	  const Size& n=el->second.size();
	  
	  /// Size to be removed
	  const Size& size=el->first;
	  
	  // Increment before erasing
	  el++;
	  
	  for(Size i=0;i<n;i++)
	    {
	      /// Memory to free
	      void* ptr=popFromCache(size,DEFAULT_ALIGNMENT);
	      	
	      deAllocateRaw(ptr);
	    }
	}
    }
    
    /// Print to a stream
    void printStatistics()
    {
      LOGGER<<"";
      LOGGER<<
	"Maximal memory used for "<<name<<": "<<usedSize.extreme()<<" bytes, "
	"number of allocation: "<<nAlloc<<", "
	"current memory used: "<<(Size)cachedSize<<" bytes, "
	"number of reused: "<<nCachedReused;
    }
    
    /// Create the memory manager
    MemoryManager(const char* name) :
      name(name)
    {
    }
    
    void initialize()
    {
      if(not inited)
	{
	  inited=true;
	  usedSize=0;
	  cachedSize=0;
	  
	  LOGGER<<"";
	  LOGGER<<"Starting the "<<name<<" memory manager";
	}
      else
	CRASH<<"Trying to initialize the already initialized "<<name<<" memory manager";
    }
    
    void finalize()
    {
      if(inited)
	{
	  inited=false;
	  
	  LOGGER<<"";
	  LOGGER<<"Stopping the "<<name<<" memory manager";
	  
	  printStatistics();
	  
	  releaseAllUsedMemory();
	  
	  clearCache();
	}
      else
	CRASH<<"Trying to finalize the not-initialized "<<name<<" memory manager";
    }
    
    /// Destruct the memory manager
    ~MemoryManager()
    {
      if(inited)
	finalize();
    }
    
    /// Get memory
    ///
    /// Call the system routine which allocate memory
    void* allocateRaw(const Size& size,        ///< Amount of memory to allocate
		      const Size& alignment)   ///< Required alignment
    {
      /// Result
      void* ptr=nullptr;
      
      /// Returned condition
      MEMORY_LOGGER<<"Allocating size "<<size<<" on "<<name;
      
#if ENABLE_DEVICE_CODE
      if constexpr(ES==ExecSpace::DEVICE)
	device::malloc(ptr,size);
      else
#endif
	{
	  const int rc=posix_memalign(&ptr,alignment,size);
	  if(rc)
	    CRASH<<"Failed to allocate "<<size<<" "<<name<<" memory with alignement "<<alignment;
	}
      
      MEMORY_LOGGER<<"ptr: "<<ptr;
      
      nAlloc++;
      
      return ptr;
    }
    
    /// Properly free
    template <typename T>
    void deAllocateRaw(T* &ptr)
    {
      MEMORY_LOGGER<<"Freeing from "<<name<<" memory address: "<<ptr;
      
#if ENABLE_DEVICE_CODE
      if constexpr(ES==ExecSpace::DEVICE)
	device::free(ptr);
      else
#else
	free(ptr);
#endif
      
      ptr=nullptr;
    }
  };
}

#endif
