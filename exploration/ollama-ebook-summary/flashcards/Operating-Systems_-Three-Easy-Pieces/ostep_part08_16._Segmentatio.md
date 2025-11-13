# Flashcards: Operating-Systems_-Three-Easy-Pieces_processed (Part 8)

**Starting Chapter:** 16. Segmentation

---

#### Problem with Base and Bounds Registers
Background context explaining the inefficiency of using a single base and bounds register pair for managing address spaces, especially when large address spaces have significant unused segments between stack and heap. This approach leads to wastage of physical memory and lack of flexibility.

:p How does the simple approach of using a base and bounds register pair per process lead to inefficiencies?
??x
The simple approach uses a single base and bounds register pair for the entire address space, leading to unnecessary allocation of physical memory even when parts of the virtual address space are unused. For instance, in a 32-bit address space (4GB), most programs only use megabytes but still demand that their entire address space be resident in memory.

This results in significant wastage of physical memory and makes it challenging to run large address spaces efficiently.
x??

---

#### Segmentation: Generalized Base/Bounds
Background context on segmentation, an older idea from the early 1960s, which divides the virtual address space into segments. Each segment has its own base and bounds register pair.

:p How does segmentation solve the inefficiency problem in managing large address spaces?
??x
Segmentation allows each logical segment of the address space to be placed independently in physical memory, avoiding the wastage of physical memory by unused portions of the virtual address space. This is achieved by having a separate base and bounds register pair for each segment.

For example, consider a 32-bit address space divided into code, stack, and heap segments. Each segment can be placed at different physical addresses, only occupying the necessary amount of physical memory.
x??

---

#### Physical Memory Layout with Segmentation
Background on how segmentation allows placing different segments (code, stack, heap) in various parts of physical memory without wasting space.

:p How is the address space divided and placed into physical memory using segmentation?
??x
The address space is divided into logical segments such as code, stack, and heap. Each segment has its own base and bounds register pair. The operating system can then place each segment independently in different parts of physical memory to optimize usage.

For instance, the example given places:
- Code at 32KB with a size of 2KB
- Heap at 34KB with a size of 2KB
- Stack elsewhere

This layout ensures only used space is allocated, thus optimizing the use of physical memory.
x??

---

#### Hardware Support for Segmentation
Explanation on how hardware supports segmentation through base and bounds registers.

:p What additional hardware support is required to implement segmentation?
??x
To support segmentation, the MMU (Memory Management Unit) needs a set of base and bounds register pairs, one pair per segment. Each pair holds the starting address (base) and the size (bounds) of a segment.

For example:
- Code: Base = 32K, Size = 2K
- Heap: Base = 34K, Size = 2K
- Stack: Base = 28K, Size = 2K

This setup allows each segment to be placed independently in physical memory.
x??

---

#### Example Physical Memory Layout
Illustration of how the segments are placed in physical memory.

:p How is a 64KB physical memory divided into segments with segmentation?
??x
In a 64KB physical memory, the segments can be placed as follows:
- Code segment: Base = 32K, Size = 2K (from 32K to 33K)
- Heap segment: Base = 34K, Size = 2K (from 34K to 35K)
- Stack segment: Typically placed elsewhere

This layout ensures that only the used memory is allocated in physical memory, optimizing space usage.
x??

---

#### Virtual Address Translation

Background context explaining how virtual addresses are translated to physical addresses using segmentation. Segmentation divides the address space into segments, each with a base and bounds.

:p How does the hardware translate a virtual address to a physical address when using segmentation?
??x
The hardware translates the virtual address by first determining which segment it belongs to (based on the top bits of the 14-bit virtual address) and then adding the offset within that segment to the corresponding base register value. This process ensures the correct physical address is accessed.

```c
// Pseudocode for translation
int Segment = (VirtualAddress & SEG_MASK) >> SEG_SHIFT;
int Offset = VirtualAddress & OFFSET_MASK;

if (Offset >= Bounds[Segment]) {
    RaiseException(PROTECTION_FAULT);
} else {
    PhysAddr = Base[Segment] + Offset;
    Register = AccessMemory(PhysAddr);
}
```
x??

---

#### Segmentation Fault

Background context on what a segmentation fault is and its origin. It arises from an illegal memory access in a segmented system.

:p What is a segmentation violation or fault, and why does it occur?
??x
A segmentation violation (or segmentation fault) occurs when the hardware attempts to access a virtual address that falls outside the valid bounds of any defined segment. This can happen if the offset plus base register value exceeds the segment's bound, indicating an illegal memory access.

```c
// Pseudocode for detecting out-of-bounds addresses
if (VirtualAddress >= MAX_VA) {
    RaiseException(SEGMENTATION_FAULT);
}
```
x??

---

#### Segment Base and Bounds

Background context on how segments are defined in the address space. Each segment has a base register value and bounds.

:p How does the hardware use segment registers to translate virtual addresses?
??x
The hardware uses the top bits of the 14-bit virtual address to determine which segment it belongs to (using segment registers). It then extracts the offset by masking off these top bits and adding them to the corresponding base register value. This process ensures only valid memory locations are accessed.

```c
// Pseudocode for determining the segment
Segment = (VirtualAddress & SEG_MASK) >> SEG_SHIFT;

// Determine the physical address
Offset = VirtualAddress & OFFSET_MASK;
PhysAddr = Base[Segment] + Offset;
```
x??

---

#### Heap Address Translation

Background context on how heap addresses are translated to physical addresses. The virtual address needs an offset adjustment before adding it to the base of the heap.

:p How does the hardware translate a heap virtual address to a physical address?
??x
For heap addresses, the hardware first calculates the offset within the segment by subtracting the base address (4KB) from the virtual address. It then adds this offset to the heap's base register value to get the correct physical address.

```c
// Pseudocode for translating heap virtual address
Offset = VirtualAddress - HeapBase;
PhysAddr = HeapBase + Offset;
```
x??

---

#### Example: Address 4200 in Heap

Background context on a specific example of a heap address and its translation. The offset is calculated by subtracting the base from the virtual address.

:p What is the physical address for virtual address 4200 in the heap?
??x
The physical address for virtual address 4200 in the heap can be found as follows:
1. Calculate the offset: `Offset = 4200 - 4096 = 104`
2. Add this offset to the heap base: `PhysAddr = HeapBase + 104 = 34KB + 104 = 34920`

```c
// Pseudocode for calculating physical address of heap virtual address
Offset = VirtualAddress - HeapBase;
PhysAddr = HeapBase + Offset;
```
x??

---

#### Illegal Address Handling

Background context on how the hardware handles illegal addresses. It checks if the offset is within bounds and raises a fault if it's not.

:p How does the hardware handle an out-of-bounds address?
??x
The hardware checks if the calculated offset is less than the segment's bounds before adding it to the base register value. If the offset exceeds the bounds, it triggers a protection fault by raising an exception.

```c
// Pseudocode for handling illegal addresses
if (Offset >= Bounds[Segment]) {
    RaiseException(PROTECTION_FAULT);
} else {
    PhysAddr = Base[Segment] + Offset;
}
```
x??

#### Segmentation and Address Space Layout

Address spaces are divided into segments, each with a base address, size, and whether it grows positively or negatively. The SEGMASK is set to 0x3000, SEGSHIFT to 12, and OFFSETMASK to 0xFFF for a three-segment system.

:p What does the setup of `SEGMASK` (0x3000), `SEGSHIFT` (12), and `OFFSETMASK` (0xFFF) imply in the context of segmentation?

??x
The SEGMASK, SEGSHIFT, and OFFSETMASK values are part of a scheme to manage address space for segments. Specifically:
- `SEGMASK` (0x3000): This value is used to extract segment information from the virtual address.
- `SEGSHIFT` (12): Indicates that the top 12 bits of the virtual address will be used as part of the segment selector.
- `OFFSETMASK` (0xFFF): Specifies the offset within the segment, which can handle up to 4KB.

The setup implies a system with segments where each virtual address is split into a high part (segment) and a low part (offset). This configuration allows for efficient addressing but requires careful handling of segments that grow in different directions.
x??

---
#### Stack and Address Space Layout

The stack grows backwards, meaning it starts at a higher physical address and shrinks towards lower addresses. For example, if the stack is placed starting from 28KB, it will extend to 26KB.

:p How does the stack's backward growth affect virtual to physical address translation?

??x
When the stack grows backwards, its virtual addresses map to physical addresses that decrease as more data is pushed onto the stack. For instance, a stack starting at 28KB and growing downwards means:
- Virtual address 16KB corresponds to physical address 28KB.
- Virtual address 14KB corresponds to physical address 27KB.

This requires special handling in address translation because standard translation methods assume forward growth. Hardware must support indicators (e.g., a bit) to determine if the segment grows positively or negatively.

```java
// Pseudocode for stack address translation
public int translateStackAddress(int virtualAddr, int base, int size) {
    // Extract the offset from the virtual address using OFFSETMASK
    int offset = virtualAddr & 0xFFF;
    
    // Calculate the negative offset if the segment grows negatively
    int negOffset = (offset - size);
    
    // The physical address is the base minus the negative offset
    int physAddr = base + negOffset;
    
    return physAddr;
}
```
x??

---
#### Segment Registers and Growth Direction

Segment registers store not just base addresses and sizes but also whether segments grow positively or negatively. This additional information allows for handling segments that grow in both directions.

:p What does the hardware need to know about a segment when dealing with negative growth?

??x
When dealing with segments that can grow in either direction, the hardware needs to know:
- The base address of the segment.
- The size of the segment.
- Whether the segment grows positively or negatively (indicated by a bit).

For example, if we have a stack segment starting at 28KB and growing downwards, the hardware must subtract the offset from the maximum possible offset for that segment to get the correct physical address.

```java
// Pseudocode for handling negative growth segments
public int translateNegativeGrowthSegment(int virtualAddr, int base, int size) {
    // Extract the offset from the virtual address using OFFSETMASK
    int offset = virtualAddr & 0xFFF;
    
    // Calculate the negative offset if the segment grows negatively
    int negOffset = (size - offset);
    
    // The physical address is the base minus the negative offset
    int physAddr = base + negOffset;
    
    return physAddr;
}
```
x??

---
#### Support for Sharing Memory Segments

System designers realized that sharing certain memory segments between different processes can save memory and improve efficiency. This requires additional hardware support to manage shared segments correctly.

:p How does supporting segment sharing help in saving memory?

??x
Supporting segment sharing helps save memory by allowing multiple processes to use the same segment of memory without duplicating it. For example, read-only code sections or data that are identical across different processes can be shared to reduce overall memory usage and improve performance.

This support involves updating the hardware to track which segments are shared and managing access to these shared segments appropriately.
x??

---

#### Code Sharing and Protection Bits
Background context: Code sharing is a common technique used in modern systems to allow the same code segment to be shared across multiple processes. This requires additional hardware support, specifically protection bits per segment that indicate read, write, or execute permissions.

:p How does code sharing work with protection bits?
??x
Code sharing works by setting specific protection bits on each segment to control access rights. For example, a code segment can be marked as read and execute only (ROX). When the same physical segment is mapped into multiple virtual address spaces of different processes, none of these processes can modify or execute the shared code, preserving isolation.

For instance, if process A and B both need to use the same code segment:
1. The hardware sets protection bits on the segment: Read-Execute.
2. Both processes A and B map this segment into their virtual address spaces.
3. Processes A and B see their own private memory views but share the same physical memory that cannot be modified.

This setup ensures that each process believes it has its own copy of the code, even though they are sharing the same physical memory region:
```java
// Pseudocode to set protection bits
setProtectionBit(segmentPointer, READ_EXECUTE);

// Process A and B mapping the segment
mapSegmentToVA(A, segmentPointer);
mapSegmentToVA(B, segmentPointer);
```
x??

---

#### Segmentation: Coarse vs. Fine Grained
Background context: While most systems use a few large segments (coarse-grained), some early systems like Multics allowed for fine-grained segmentation where the address space is divided into many smaller segments. This provides more flexibility in managing memory but requires additional hardware support.

:p What distinguishes coarse-grained from fine-grained segmentation?
??x
Coarse-grained segmentation divides the address space into a few large segments, while fine-grained segmentation uses many small segments. Coarse-grained is simpler and less flexible, suitable for systems with fewer distinct areas of memory usage like code, stack, heap. Fine-grained allows more precise control over memory regions but requires more complex hardware support.

For example:
- Coarse: Code (32K), Stack (28K), Heap (34K)
- Fine: Many segments like thousands for different functions, data structures

Code can demonstrate setting up fine-grained segmentation in a C-like syntax with multiple entries in a segment table:
```c
// Pseudocode to set up fine-grained segmentation
struct SegmentTableEntry {
    unsigned baseAddress;
    unsigned size;
    int protectionBits; // 1=Read-Execute, 2=Read-Write
};

SegmentTable[0] = {0x32K, 32K, READ_EXECUTE}; // Code segment
SegmentTable[1] = {0x28K, 28K, READ_WRITE}; // Stack segment
// More segments can be added as needed

void setupSegementTable() {
    for each entry in SegmentTable {
        setProtectionBits(entry.baseAddress, entry.protectionBits);
        mapSegmentToVA(currentProcessID, entry.baseAddress);
    }
}
```
x??

---

#### Operating System Support for Segmentation
Background context: To effectively manage segmented memory spaces, operating systems need to support the relocation of segments into physical memory as processes run. This involves tracking which segments are in use and managing the mapping between virtual addresses and physical ones.

:p How does an OS support segmentation?
??x
An OS supports segmentation by maintaining a segment table that tracks information about each segment such as its base address, size, and protection bits. The hardware is responsible for checking if accesses to segments are within bounds and permissible based on the protection settings.

For instance:
- Tracking which segments are in use: The OS updates this status based on process activities.
- Mapping virtual addresses to physical memory: This involves using segment tables to find the correct physical address when a virtual address is accessed.

Example pseudocode for managing segmentation:
```java
// Pseudocode for managing segmentation
public class SegmentManager {
    private SegmentTable segmentTable;

    public void mapSegment(Process process, int segmentIndex) {
        SegmentEntry entry = segmentTable.get(segmentIndex);
        if (entry != null && entry.protectionBits.permits(process.accessType)) {
            allocatePhysicalMemory(entry.baseAddress + process.virtualAddressOffset);
            process.updateSegmentMapping(entry.baseAddress);
        } else {
            throw new AccessViolationException("Access to this segment is not permitted");
        }
    }

    private void allocatePhysicalMemory(long physicalAddress) {
        // Logic to map virtual address to physical memory
    }
}
```
x??

---

#### Context Switch Handling During Segmentation
Background context explaining the concept. When a process undergoes a context switch, the operating system must save and restore segment registers to ensure that each process runs with its own virtual address space. This is crucial for maintaining the isolation between processes.

:p What should the OS do on a context switch during segmentation?
??x
The OS needs to save and restore the segment registers of the currently running process before performing a context switch, ensuring that each process maintains its unique virtual address space. This involves saving the values in the segment registers (like `cs`, `ds`, `es`, etc.) and restoring them when the process resumes execution.

```java
// Pseudocode for saving and restoring segment registers during context switch

void saveContextRegisters() {
    // Save current segment registers
    cs = readCS();
    ds = readDS();
    es = readES();
    fs = readFS();
    gs = readGS();

    // Perform the context switch to another process
}

void restoreContextRegisters() {
    // Restore segment registers for the original process
    writeCS(cs);
    writeDS(ds);
    writeES(es);
    writeFS(fs);
    writeGS(gs);
}
```
x??

---

#### Managing Free Space in Physical Memory (External Fragmentation)
Background context explaining the concept. With segmentation, physical memory can quickly become filled with small gaps of free space, making it difficult to allocate new segments or grow existing ones. This is known as external fragmentation.

:p What problem arises due to segmentation in physical memory management?
??x
The problem that arises is external fragmentation, where physical memory becomes fragmented into many small chunks of free space. This makes it challenging for the operating system to find a contiguous block of memory large enough to satisfy segment allocation requests.

```java
// Example code snippet showing how to detect gaps in physical memory

public boolean checkFreeSpace(int size) {
    // Assume 'memory' is an array representing used/available slots in memory
    int i = 0;
    while (i < memory.length) {
        if (!memory[i]) { // If the slot is free
            int freeLength = 1;
            while (i + freeLength < memory.length && !memory[i + freeLength]) {
                freeLength++;
            }
            if (freeLength >= size) {
                return true; // Found a contiguous block of free space of the required size
            }
        }
        i++;
    }
    return false; // No suitable free space found
}
```
x??

---

#### Compaction as a Solution to External Fragmentation
Background context explaining the concept. One solution to external fragmentation is compaction, where the operating system rearranges segments in memory to create large contiguous free spaces.

:p What is one solution to manage external fragmentation?
??x
One solution to manage external fragmentation is compaction, which involves stopping processes, copying their data to a contiguous region of memory, and updating segment registers. This process can then provide a larger extent of free memory for allocation.

```java
// Pseudocode for the compaction process

void compactMemory() {
    // Stop all running processes temporarily
    stopProcesses();

    // Copy each process's segments to a new contiguous location in memory
    for (Process p : activeProcesses) {
        copySegmentsToContiguousLocation(p);
        updateSegmentRegisters(p);
    }

    // Resume processes now that they are located contiguously
    resumeProcesses();
}

void copySegmentsToContiguousLocation(Process p) {
    // Logic to copy segments of process p from current locations to a new, contiguous location
}

void updateSegmentRegisters(Process p) {
    // Update segment registers (cs, ds, etc.) to point to the new physical addresses
}
```
x??

---

#### Free-List Management Algorithms for External Fragmentation
Background context explaining the concept. Another solution is using free-list management algorithms that try to keep large extents of memory available for allocation.

:p What are some approaches to managing external fragmentation?
??x
Some approaches to managing external fragmentation include various free-list management algorithms such as best-fit, worst-fit, first-fit, and more complex schemes like the buddy system. These algorithms attempt to minimize gaps in memory by efficiently managing free space.

```java
// Example code snippet for a simple best-fit allocation algorithm

public int allocate(int size) {
    // Assume 'freeSpaces' is an array representing available free spaces of various sizes
    int bestFit = -1;
    int smallestGap = Integer.MAX_VALUE;

    for (int i = 0; i < freeSpaces.length; i++) {
        if (freeSpaces[i] >= size && freeSpaces[i] < smallestGap) {
            smallestGap = freeSpaces[i];
            bestFit = i;
        }
    }

    // Allocate the found space
    if (bestFit != -1) {
        freeSpaces[bestFit] -= size;
        return bestFit; // Return the index of allocated space
    } else {
        return -1; // No suitable space was found
    }
}
```
x??

---

#### External Fragmentation
Background context explaining the concept of external fragmentation. Discuss how variable-sized segments lead to memory being chopped up into odd-sized pieces, making it difficult to satisfy allocation requests.

:p What is external fragmentation?
??x
External fragmentation occurs when free memory gets fragmented into small, non-contiguous blocks that are too small to be utilized for requested allocations. This happens because segments are of varying sizes, and as a result, free space in the memory can become scattered throughout, making it hard to find large enough chunks.

For example:
```java
// Imagine a scenario where multiple allocations and deallocations have fragmented the memory
MemoryBlock mem = new MemoryBlock(1024);
mem.allocate(300); // Allocates 300 bytes
mem.deallocate(300);

// Now there is an available block of 724 bytes, but no request for such a size can be satisfied.
```
x??

---

#### Segmentation and Sparse Address Spaces
Discuss the advantages of segmentation in supporting sparse address spaces. Explain how segmentation avoids memory wastage between logical segments.

:p How does segmentation help with sparse address spaces?
??x
Segmentation helps manage sparse address spaces by avoiding huge potential waste of memory between logical segments. Instead of having a continuous large block, memory can be divided into smaller, logically meaningful segments that better match the actual usage pattern, reducing overall memory overhead.

For example:
```java
// Consider a scenario with a large but sparsely used heap
SegmentedMemoryManager mem = new SegmentedMemoryManager(1024 * 1024);
mem.createSegment("Heap", 512 * 1024);

// Here, only the needed part of the segment is in memory at any time, reducing overhead.
```
x??

---

#### Code Sharing with Segmentation
Explain how code sharing can be facilitated by placing code within separate segments.

:p How does segmentation enable code sharing?
??x
By placing code within a separate segment, multiple running programs can share that segment, thereby saving memory and improving efficiency. This is because the same code doesn't need to be loaded repeatedly into different processes.

For example:
```java
// Assume we have an application that uses a common library
SegmentedMemoryManager mem = new SegmentedMemoryManager();
mem.createSegment("Library", 2048 * 1024);

Process p1 = new Process(mem, "Program1");
Process p2 = new Process(mem, "Program2");

// Both programs can share the same segment for the library code.
```
x??

---

#### Flexibility of Segmentation
Discuss why segmentation isn't flexible enough to support a fully generalized, sparse address space. Provide an example illustrating this limitation.

:p Why is segmentation not sufficiently flexible?
??x
Segmentation struggles with providing full flexibility because it still requires the entire segment (e.g., a large heap) to reside in memory even if only parts of it are used. This can lead to inefficiencies where substantial portions of memory go unused, especially for sparsely used address spaces.

For example:
```java
// Example: A large but sparsely used heap
SegmentedMemoryManager mem = new SegmentedMemoryManager(1024 * 1024);
mem.createSegment("Heap", 512 * 1024);

// If the program only uses a small part of this heap, the entire segment must be kept in memory.
```
x??

---

#### Solution to Flexibility Issues
Suggest an approach that might address the limitations of segmentation discussed earlier. Provide a brief description.

:p What solutions could we consider for flexibility issues?
??x
To address the limitations of segmentation, one potential solution is to introduce **page-based memory management** or **demand paging**, where memory pages can be allocated and deallocated dynamically based on actual usage. This approach allows for better utilization of memory by keeping only necessary parts in physical memory.

For example:
```java
// Example: Using demand paging
PageBasedMemoryManager mem = new PageBasedMemoryManager(1024 * 1024);
mem.createSegment("Heap", 512 * 1024);

// Only the pages of the heap that are actually used will be kept in memory.
```
x??

---

#### References and Historical Context
Provide brief descriptions for each reference, highlighting their significance.

:p What is the significance of the references provided?
??x
- **[CV65]**: This paper introduced Multics and discussed its segmentation system. It was one of the first to propose a comprehensive memory management solution.
- **[DD68]**: This early paper in 1968 detailed dynamic linking techniques, which were ahead of their time and eventually became widely used in modern systems due to advancements like X-windows libraries.
- **[G62]**: An early paper on fact segmentation that explored basic concepts without references to other work, indicating the pioneering nature of this research.
- **[H61]**: A foundational paper by Holt that laid some groundwork for understanding and implementing segmentation.

x??

#### Segmentation Basics
Background context: This section covers segmentation, a memory management technique that divides memory into fixed-size blocks called segments. Each segment has its own base address and bounds, allowing for variable-sized regions within the virtual address space.

:p What is the highest legal virtual address in segment 0 with the given parameters?
??x
The highest legal virtual address in segment 0 can be calculated by adding the base address (b) to the length of the segment minus one. For example, if -b0 is set to 512 and -l0 is 20, then the highest legal address would be $512 + 20 - 1 = 531$.

x??

---

#### Virtual Address Space
Background context: The virtual address space consists of multiple segments each with a base address (b) and length (l). The total size of the address space is determined by the parameter `-a`.

:p What are the lowest and highest illegal addresses in this entire address space?
??x
The lowest illegal address would be 0, as it's below the virtual address space. The highest illegal address would be $2^{address\space size} - 1 $(considering a typical 32-bit system where the maximum address is $2^{32} - 1 = 4294967295$).

x??

---

#### Address Translation Simulation
Background context: The `segmentation.py` program simulates memory segmentation and translation. Parameters like `-a`, `-p`, `-b`, `-l`, etc., control the address space, physical memory size, base addresses, lengths of segments, and random seed for address generation.

:p What is a valid set of parameters to generate the specified translation results?
??x
To generate the specified translations (valid, valid, violation, ..., violation, valid, valid) with an 16-byte address space in a 128-byte physical memory:

```plaintext
segmentation.py -a 16 -p 128 -A 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 
               --b0 ? --l0 ? --b1 ? --l1 ?
```

For the given stream:
- Segment 0: Base = 0, Length = 8 (valid, valid)
- Segment 1: Base = 8, Length = 8 (violation)

So,
```plaintext
--b0 0 --l0 8 --b1 8 --l1 8
```

x??

---

#### Randomly Generated Virtual Addresses

:p How can you configure the simulator to generate about 90% valid virtual addresses?
??x
To achieve approximately 90% valid virtual addresses, the segments should cover most of the address space while leaving a small portion as invalid. For example:

```plaintext
segmentation.py -a 16 -p 128 -A 0,15 --b0 0 --l0 14
```

Here, segment 0 covers the first 14 out of 16 addresses (93.75%), leaving only a small invalid range at the end.

x??

---

#### No Valid Addresses

:p How can you run the simulator such that no virtual addresses are valid?
??x
To ensure all generated virtual addresses are invalid, set up segments to cover the entire address space without leaving any gaps for valid addresses. For example:

```plaintext
segmentation.py -a 16 -p 128 -A 0,15 --b0 1 --l0 14
```

This configuration sets segment 0 from base 1 to 15 (leaving the first address invalid).

x??

---

