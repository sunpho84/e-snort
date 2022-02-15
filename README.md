# e-snort

We need to fresh up our mind.

MemoryManager
---
We put two memory manager, the correct one is automatically chosen

Stacked tensor
------------
Is automatically captured? yes

Dynamic sized tensor
--------------------
Is located on host or device, default storage is device but we can manually take host.
Can we add a method or a wrapper:

```
onHost()
```

```
onDevice()
```

Mirrored data
-------------
We use like lookup table, with proxy to avoid consolidate.
Mirrored data cannot be written directly but an accessor can be asked.

```
getWriteableAcces([](auto& instance)
{

})

getReadableAccess(const auto& instance)
```

Metaprogramming
---
Avoid SFINAE, use tag dispatch

Parallelization
---
Parallelization is issued when assigning or reducing

Execution space
---
Each expression must have an execution space associated.
Question: A stacked tensor understand the execution space?

Reduction
---
We fill a buffer and iterate reduction. 
Accumulated precision must be adjustable.
The reduction must output a stacked tensor.

Nodes
---
If we take a reference inside a node on the host, then we send it to a
device function, is the reference copied, or do we refer to the
original quantity?
