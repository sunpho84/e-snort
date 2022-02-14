# e-snort

We need to fresh up our mind.

MemoryManager
---
We put two memory manager

Stacked tensor
------------
Is automatically captured? yes

Dynamic sized tensor
--------------------
Is located on host or device, default storage is device but we can manually take host.
Can we add a method or a warapper onHost() onDevice?

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


Reduction
---
We fill a buffer and iterate reduction. 
Accumulated precision must be adjustable.
The reduction must output a stacked tensor.
