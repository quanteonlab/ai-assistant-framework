# Flashcards: Operating-Systems_-Three-Easy-Pieces_processed (Part 51)

**Starting Chapter:** 50. Andrew File System AFS

---

#### Introduction to AFS (Andrew File System)
Background context: The Andrew File System was introduced at Carnegie-Mellon University in the 1980s, designed primarily for scaling distributed file systems. It aimed to support as many clients as possible on a server by optimizing protocol design and user-visible behavior.
:p What is the main goal of AFS?
??x
The primary objective of AFS was to scale a distributed file system so that servers could handle a large number of client connections effectively. This was achieved through the optimization of protocols between clients and servers, ensuring efficient caching mechanisms and minimizing server load.
x??

---

#### Differences Between AFSv1 and NFS
Background context: The original version (AFSv1) of AFS had some basic design in place but did not scale as desired. This led to a redesign resulting in the final protocol known as AFSv2 or simply AFS, which addressed these scalability issues.
:p How does AFS differ from NFS in terms of user-visible behavior?
??x
AFS differed significantly from NFS by focusing on ensuring reasonable and user-visible behaviors right from the start. Specifically, AFS designed its cache consistency model to be simple and easy to understand, whereas NFS required complex low-level implementation details for cache consistency.
x??

---

#### Protocol Highlights of AFSv1
Background context: The protocol highlights of AFSv1 include key operations such as TestAuth, GetFileStat, Fetch, Store, SetFileStat, and ListDir. These operations are crucial for understanding how data is managed between the client and server.
:p What are some of the key protocols in AFSv1?
??x
Some of the key protocols in AFSv1 include:
- TestAuth: Used to validate cached entries.
- GetFileStat: Retrieves stat information for a file.
- Fetch: Downloads the contents of a file from the server.
- Store: Stores this file on the server.
- SetFileStat: Sets the stat info for a file.
- ListDir: Lists the contents of a directory.

These protocols ensure efficient data management and consistency between the client and the server.
x??

---

#### Caching Mechanism in AFSv1
Background context: In AFSv1, files are cached entirely on the local disk of the client machine when opened. Subsequent read and write operations use this local cache, minimizing network communication and improving performance.
:p How does AFS handle file caching?
??x
In AFSv1, when a user opens a file, the entire file is fetched from the server and stored locally on the client's disk. This allows subsequent read and write operations to be handled using the local file system without needing network communication, thus improving performance.

Code example:
```java
// Pseudocode for opening a file in AFSv1
public void openFile(String filePath) {
    // Fetch the entire file from the server if it doesn't exist locally
    File localFile = new File(filePath);
    if (!localFile.exists()) {
        fetchFromServer(filePath, localFile);
    }
    // Start using the cached file for read and write operations
}

private void fetchFromServer(String filePath, File localFile) {
    // Logic to download the entire file from the server
    // This can be simplified in practice but illustrates the process
}
```
x??

---

#### Flushing Mechanism in AFSv1
Background context: Upon closing a file that has been modified, it is flushed back to the server. This ensures that changes are propagated and maintained across all clients.
:p What happens when a user closes an edited file in AFSv1?
??x
When a user closes an edited file in AFSv1, if the file has been modified, it is flushed back to the server to ensure that the latest version of the file is stored on the server and available to other clients.
```
// Pseudocode for closing a file in AFSv1
public void closeFile(String filePath) {
    File localFile = new File(filePath);
    if (hasFileBeenModified(localFile)) {
        // Flush changes back to the server
        flushToServer(filePath, localFile);
    }
}

private boolean hasFileBeenModified(File file) {
    // Check if the local file is modified compared to the version on the server
    return true; // This would be replaced with actual logic in practice
}

private void flushToServer(String filePath, File localFile) {
    // Logic to upload changes from the local file back to the server
    // Again, this simplifies the process for illustration purposes
}
```
x??

---

#### NFS vs. AFS Caching Mechanisms
AFS caches whole files, while NFS caches blocks of files. NFS can cache every block of an entire file but does so in client memory rather than on local disk.

:p What is the primary difference between how NFS and AFS handle caching?
??x
In NFS, blocks are cached in client memory, whereas in AFS, whole files are cached on the local disk. This means that while both systems cache data for performance reasons, their granularity differs significantly.
x??

---

#### File Fetch Protocol in AFS
When a client application first calls `open()`, the AFS client sends a Fetch protocol message to the server. The Fetch message contains the entire pathname of the desired file.

:p How does the AFS client handle the initial fetch of a file when a user opens it?
??x
The AFS client sends a Fetch protocol message that includes the full path to the file. This message is sent to the file server, which then locates and transfers the entire file back to the client, where it is cached on local disk.

```c
// Pseudocode for Fetch Protocol Message Handling in Client-Side Code
void fetchFile(char* pathname) {
    // Send Fetch protocol message with full path
    sendToServer(FETCH_PROTOCOL, pathname);
    
    // Cache entire file locally after receiving
    cacheEntireFile(pathname);
}
```
x??

---

#### Local Read and Write Operations in AFS

Once a file is cached on the client's local disk, subsequent read and write operations are handled locally. No further communication with the server occurs for these operations.

:p How does AFS handle read and write operations after fetching a file?
??x
After a file is fetched to the client's local disk, any subsequent `read()` or `write()` calls are performed directly on the cached copy without involving the server. The client-side code redirects these system calls to the local file.

```c
// Pseudocode for Read and Write Handling in AFS
int readLocalFile(char* filename) {
    // Check if file is already cached locally
    if (isCached(filename)) {
        return readFileFromDisk(filename);
    } else {
        fetchFile(filename); // Fetch file first if not cached
        return readFileFromDisk(filename);
    }
}

void writeLocalFile(char* filename, char* data) {
    // Check if file is already cached locally
    if (isCached(filename)) {
        writeFileToDisk(filename, data);
    } else {
        fetchFile(filename); // Fetch file first if not cached
        writeFileToDisk(filename, data);
    }
}
```
x??

---

#### File Update Protocol in AFS

When a client is done using a file and it has been opened for writing, the client sends a `Store` protocol message to the server to update the file on permanent storage. The `Store` message includes both the entire updated file and its pathname.

:p What happens when a user finishes modifying a file in AFS?
??x
When the client is done with a file that has been opened for writing, it sends a `Store` protocol message to the server. This message contains the full path of the file and the complete contents of the updated file. The server then stores this information permanently.

```c
// Pseudocode for File Update Protocol in AFS
void storeFile(char* filename) {
    // Read the entire local copy of the file
    char* fileData = readFileFromDisk(filename);
    
    // Send Store protocol message to server with path and data
    sendToServer(STORE_PROTOCOL, filename, fileData);
}
```
x??

---

#### Protocol Messages in AFS

AFS uses several protocol messages such as Fetch, TestAuth, and Store. The Fetch message is used for initial file transfer, the TestAuth message checks if a file has been modified by another client, and the Store message updates the server with the latest version of a file.

:p What are some key protocol messages used in AFS?
??x
AFS uses several important protocol messages:
- **Fetch**: Used to initially fetch a file from the server.
- **TestAuth**: Checks if a file has been modified by another client.
- **Store**: Updates the server with the latest version of a file.

These messages help manage file transfers and ensure consistency between clients and the server.

```c
// Pseudocode for Protocol Messages in AFS
void fetchFile(char* filename) {
    sendToServer(FETCH_PROTOCOL, filename);
}

bool testAuth(char* filename) {
    sendToServer(TESTAUTH_PROTOCOL, filename);
    return receivedConfirmation();
}

void storeFile(char* filename) {
    char* fileData = readFileFromDisk(filename);
    sendToServer(STORE_PROTOCOL, filename, fileData);
}
```
x??

---

#### Performance Optimization with Cache Checking

AFS uses a TestAuth protocol message to check if the file has been modified by another client. If not, it can use its local cache, thereby avoiding unnecessary network transfers.

:p How does AFS optimize performance using cache checking?
??x
AFS optimizes performance by using a `TestAuth` protocol message to determine whether a file has been modified since it was last cached. If the file has not changed, AFS uses the locally-cached copy, thus saving time and bandwidth by avoiding unnecessary network transfers.

```c
// Pseudocode for Cache Checking in AFS
bool isFileModified(char* filename) {
    sendToServer(TESTAUTH_PROTOCOL, filename);
    return receivedModificationConfirmation();
}

void useLocalCacheIfNotModified(char* filename) {
    if (!isFileModified(filename)) {
        // Use local cache
        readFromDisk(filename);
    } else {
        // Fetch file from server
        fetchFile(filename);
    }
}
```
x??

---

#### Path-Traversal Costs in AFSv1
Background context: In AFSv1, when a client performs a Fetch or Store protocol request, it passes the entire pathname to the server. The server then needs to traverse this path from the root directory level by level to locate and access the desired file.

:p What is the main problem with path-traversal costs in AFSv1?
??x
The primary issue is that the server spends a significant amount of CPU time walking down directory paths for every client request. This overhead increases as more clients simultaneously access the system, making it difficult to scale.
x??

---

#### Excessive TestAuth Messages in AFSv1
Background context: AFSv1 generated a large number of TestAuth protocol messages to check whether local files were valid. Most of these checks resulted in confirmation that the file had not changed since being cached.

:p What is the main problem with TestAuth messages in AFSv1?
??x
The excessive TestAuth messages created significant network traffic and CPU overhead for servers, as they frequently confirmed unchanged cached copies of files to clients.
x??

---

#### Scalability Issues in AFSv1
Background context: The combination of high path-traversal costs and excessive TestAuth messages limited the scalability of AFS. Servers could only service a small number of clients without becoming overloaded.

:p How did these issues limit the scalability of AFS?
??x
High path-traversal costs and frequent TestAuth messages caused servers to spend too much time on non-productive tasks, leading to a bottleneck that prevented the system from handling more clients efficiently.
x??

---

#### Design Challenges for AFSv2
Background context: The designers faced challenges in designing a scalable file protocol. They needed to reduce server interactions (by minimizing TestAuth messages) and make these interactions efficient.

:p What were the main design challenges for AFSv2?
??x
The primary challenge was to design a protocol that minimized server interactions while ensuring efficiency. This required reducing the number of TestAuth messages and optimizing how client/server communications were handled.
x??

---

#### Introduction of Callbacks in AFSv2
Background context: To address the issues, AFSv2 introduced the concept of callbacks. These are promises from the server to inform the client when a cached file has been modified.

:p What is a callback in the context of AFSv2?
??x
A callback in AFSv2 is a mechanism where the server informs the client that a cached file has been modified, reducing the need for frequent TestAuth messages.
x??

---

#### Benefits of Callbacks in AFSv2
Background context: By using callbacks, AFSv2 significantly reduced the number of client/server interactions and improved overall system efficiency.

:p How did callbacks improve the protocol in AFSv2?
??x
Callbacks improved the protocol by reducing the frequency of TestAuth messages. Instead of constantly checking if cached files are still valid, clients receive notifications from servers when changes occur, leading to more efficient use of server resources.
x??

---

#### Volume Management in AFSv2
Background context: Another issue addressed in AFSv2 was load balancing across servers. This was solved by allowing administrators to move volumes (collections of files) between different servers.

:p How did volume management address the scalability issues in AFS?
??x
Volume management addressed scalability by enabling administrators to balance server loads through migration of file collections (volumes). This ensured that no single server became overloaded.
x??

---

#### Context-Switch Problem in AFSv1
Background context: In addition to path-traversal and TestAuth issues, AFSv1 also faced problems due to the use of processes per client, leading to excessive context switching overhead.

:p What was another problem with AFSv1 related to client-server communication?
??x
AFSv1 had a context-switch problem where each server used a separate process for each client. This led to significant overhead in managing multiple context switches.
x??

---

#### Solution for Context-Switch Problem in AFSv2
Background context: The context-switch issue was resolved in AFSv2 by implementing the server with threads instead of processes, thus reducing the overhead associated with context switching.

:p How did AFSv2 solve the context-switch problem?
??x
AFSv2 solved the context-switch problem by building servers using threads rather than separate processes for each client. This reduced the overhead and improved overall performance.
x??

#### Cache Validity Check Mechanism
In AFSv2, a novel approach to cache management was introduced. Instead of continually polling the server to check if cached files are still valid (polling), the client assumes that the file is still valid until notified otherwise by the server. This mechanism reduces the load on the server and enhances efficiency.
:p How does AFSv2 handle caching validation for files?
??x
AFSv2 uses a callback system where, upon each fetch of a directory or file, it establishes a connection with the server to ensure that the client is notified of any changes in the cached state. This approach contrasts with traditional polling methods and significantly reduces the load on the server.
```java
// Pseudocode for establishing a callback
public void establishCallback(FileIdentifier fid) {
    // Code to request notification from the server when file changes
}
```
x??

---

#### File Identifier (FID)
AFSv2 introduced an innovative method of specifying which file the client is interested in using an FID instead of pathnames. An FID consists of three components: a volume identifier, a file identifier, and a uniquifier.
:p What is an FID used for in AFSv2?
??x
An FID is used to uniquely identify a file in the AFS without relying on pathnames. This allows clients to efficiently access files by caching directory contents and establishing callbacks with the server for notifications of changes.
```java
// Pseudocode for creating an FID
public FileIdentifier createFID(VolumeIdentifier volumeId, FileIdentifier fileId) {
    return new FileIdentifier(volumeId, fileId, uniquifier);
}
```
x??

---

#### Directory Access in AFSv2
When a client needs to access a file within AFS, it fetches directory contents and establishes callbacks. The client walks the pathname piece by piece, caching results to reduce load on the server.
:p How does the client handle directory access in AFSv2?
??x
The client accesses files through a series of Fetch requests to the server, starting from the root directory. For example, accessing `/home/remzi/notes.txt` involves fetching `home`, then `remzi`, and finally `notes.txt`. Each fetch operation caches results locally and sets up callbacks for change notifications.
```java
// Pseudocode for fetching a file in AFSv2
public FileIdentifier fetchFile(String path) {
    // Logic to split path into pieces, fetch directories, and files
}
```
x??

---

#### Callback Mechanism Overview
AFSv2 employs callback mechanisms where the server notifies the client about changes in cached data. This reduces the frequency of client-server communication, improving overall performance.
:p What is a key feature of AFSv2's caching mechanism?
??x
A key feature of AFSv2’s caching mechanism is the use of callbacks to notify clients when files change on the server. This avoids constant polling and enhances efficiency by reducing load on the server.
```java
// Pseudocode for handling file changes via callback
public void handleFileChange(FileIdentifier fid) {
    // Logic to update local cache and notify application of changes
}
```
x??

---

#### Cache Consistency Challenges
Cache consistency is a complex issue in distributed filesystems. While AFSv2's approach significantly reduces server load, it does not solve all cache inconsistency issues.
:p What are the challenges with cache consistency in AFS?
??x
Cache consistency remains a significant challenge in distributed filesystems like AFS. Although AFSv2’s callback mechanism improves efficiency by reducing server load, it may still face inconsistencies due to network delays or failures. Ensuring that cached data is always up-to-date requires careful management and additional mechanisms.
```java
// Pseudocode for cache consistency check
public void ensureCacheConsistency(FileIdentifier fid) {
    // Logic to compare local cache with server state and handle discrepancies
}
```
x??

#### File Consistency Challenges in Multi-Client Environments
Background context: The text discusses challenges related to file consistency when multiple clients are accessing a shared code repository. It highlights that simple file locking mechanisms or baseline consistency might not be sufficient for handling concurrent updates and conflicts, especially in scenarios where files need to be updated by different users simultaneously.
:p What is the main issue with using only underlying file systems for managing concurrent file accesses in multi-client environments like a code repository?
??x
The main issue is that simple file system mechanisms do not provide explicit control over who can access or modify files concurrently, which can lead to data inconsistencies and conflicts. For example, when multiple clients are checking in or out of code, it’s necessary to use more advanced mechanisms such as explicit file-level locking to ensure that only one client can make changes at a time.
x??

---

#### AFS Cache Consistency
Background context: The text explains how the Andrew File System (AFS) manages cache consistency through callbacks and whole-file caching. It outlines two important cases for understanding cache consistency: between different machines and on the same machine.
:p How does AFS ensure that cached files are updated when a file is modified by another client?
??x
AFS ensures that cached files are updated when a file is modified by another client through the use of callbacks. When a file is closed after being written to, the new version is flushed to the server. At this point, AFS breaks any existing callbacks for clients holding outdated copies of the file, forcing them to re-fetch the latest version.
x??

---

#### Consistency Between Different Machines
Background context: The text specifies that in AFS, updates are made visible at the server and cached copies are invalidated simultaneously when a file is closed. This ensures that all clients have access to the most recent version of the file upon their next interaction.
:p How does AFS handle consistency between different machines when updating files?
??x
AFS handles consistency between different machines by breaking callbacks for any client holding an outdated copy of the file immediately after the new version is flushed to the server. This ensures that once a file is closed and its updated version is available on the server, all clients opening it subsequently will receive the latest version without relying on stale cached data.
x??

---

#### Consistency Between Processes on the Same Machine
Background context: The text also addresses how AFS maintains consistency between different processes running on the same machine. It mentions that local accesses to files are fast because they do not require server interaction, as the file is already cached.
:p How does AFS ensure consistent access when multiple processes on a single client open and modify the same file?
??x
AFS ensures consistent access by managing file writes and caching locally. When a process opens and writes to a file, any changes are stored in local cache. Once the file is closed, these changes are flushed to the server, invalidating any outdated cached copies on other clients. Subsequent accesses from the same client will use the updated local copy without needing further interaction with the server.
x??

---

#### Cache Staleness
Background context: The text explains how AFS manages cache staleness, where a file’s new version is made visible to clients as soon as it is written to the server. It also mentions that breaking callbacks ensures clients do not read outdated versions of files.
:p What mechanism does AFS use to manage cache staleness and ensure clients see the latest version of a file?
??x
AFS manages cache staleness by invalidating cached copies on clients immediately after writing the new file version to the server. This is done through breaking callbacks, where the server contacts each client holding an outdated copy and informs them that their callback for the file is no longer valid. This ensures that subsequent file accesses from these clients will fetch the updated version.
x??

---

#### Example of File Access Sequence
Background context: The text provides a sequence example to illustrate how AFS handles file access, including opening, writing, closing, and re-opening files.
:p Describe the sequence of operations for accessing a file in AFS as shown in the provided comments.
??x
The sequence of operations for accessing a file in AFS is as follows:
1. Client 1 opens file `F` (P1).
2. Client 1 writes to file `A` (write(A)).
3. Client 1 closes the file (close()).
4. The updated file is flushed to the server, making it visible.
5. Server breaks callbacks for any clients with cached copies of `F`.
6. Client 2 opens the same file `F` (open(F)), and reads the new version which is A.

This sequence ensures that subsequent accesses from both clients will use the latest version of the file without relying on stale cached data.
x??

---
These flashcards are designed to help with understanding key concepts in AFS, including file consistency challenges, cache consistency mechanisms, and specific operations like managing callbacks and handling local vs. server interactions.

#### Cache Consistency in AFS
Background context: AFS (Andrew File System) manages file access across multiple machines. It ensures consistency through a combination of local caching and periodic updates to the server. This system involves processes on different clients and the server, where writes are visible locally but may not be immediately reflected on other clients until updated.
:p What is AFS and how does it handle cache consistency?
??x
AFS is designed for distributed systems where files can be cached locally to improve performance. The key aspects of its cache consistency model include local updates being visible immediately, while changes are eventually synchronized with the server. This ensures that a single machine behaves as expected but introduces challenges when multiple machines modify the same file.
The system uses techniques like callbacks and last writer wins (or closer wins) to manage inconsistencies.

```java
// Pseudocode for handling cache consistency in AFS
public class CacheManager {
    void handleCacheConsistency(String filePath, boolean isWriteOperation) {
        if (isWriteOperation) {
            // Perform local write operation
            System.out.println("Local write operation performed.");
            
            // Send a callback to the server
            sendCallbackToServer(filePath);
        } else {
            // Perform read operation and check cache validity
            String cachedContent = getCachedContent(filePath);
            if (isValid(cachedContent)) {
                System.out.println("Using local cache content: " + cachedContent);
            } else {
                // Fetch updated content from the server
                fetchUpdatedContentFromServer(filePath);
            }
        }
    }

    private void sendCallbackToServer(String filePath) {
        // Code to send a callback message to the server for file update
        System.out.println("Sending callback to server for " + filePath);
    }

    private boolean isValid(String content) {
        // Check if cached content is valid
        return true; // Simplified check, actual implementation would be more complex
    }
}
```
x??

---

#### Last Writer Wins Mechanism in AFS
Background context: In scenarios where multiple clients modify a file simultaneously on different machines, AFS uses a last writer wins (or closer wins) approach. This means the client that closes the file last will have its changes applied to the server first.
:p How does AFS handle writes from multiple clients modifying the same file?
??x
AFS employs a mechanism where the last client to close a file on different machines will have its changes committed to the server, effectively overriding any changes made by other clients. This ensures that only one version of the file remains on the server.
The key point is that writes are not flushed immediately but accumulate in each client's cache until the file is closed and updated.

```java
// Pseudocode for handling last writer wins in AFS
public class FileHandler {
    void handleWrite(String filePath) {
        // Accumulate changes locally
        System.out.println("Accumulating changes for " + filePath);
        
        // Simulate a write operation
        updateLocalCache(filePath, "some data");
    }

    void handleClose(String filePath) {
        // Determine the last closer and apply changes to server
        if (this == lastCloser) { // Hypothetical method to identify the last closer
            System.out.println("Applying final changes for " + filePath);
            updateServer(filePath, getLocalCacheContents());
        } else {
            System.out.println("Waiting for other clients to close.");
        }
    }

    private void updateLocalCache(String filePath, String data) {
        // Code to update local cache
    }

    private String getLocalCacheContents() {
        // Code to retrieve cached contents
        return "cached content"; // Hypothetical implementation
    }

    private void updateServer(String filePath, String content) {
        // Code to send changes to the server
    }
}
```
x??

---

#### Crash Recovery in AFS
Background context: AFS needs a robust crash recovery mechanism to handle situations where clients are rebooted or disconnected. When a client reboots after missing critical messages from the server, it must verify its cache contents and update them if necessary.
:p How does AFS ensure consistency when a client reboots?
??x
When a client reboots, AFS treats all cached files as potentially invalid since they may have missed important callback messages. The system requires clients to request validation from the server before using any local cache content. If the cache is valid, it can be used; otherwise, the latest version must be fetched.
This process ensures that the client has the most up-to-date information when rejoining the network.

```java
// Pseudocode for crash recovery in AFS
public class ClientRecovery {
    void recoverCache(String filePath) {
        // Send a TestAuth message to validate cache contents
        boolean isValid = testCacheValidity(filePath);
        
        if (isValid) {
            System.out.println("Using local cache content: " + getLocalCacheContent(filePath));
        } else {
            // Fetch updated content from the server
            fetchUpdatedContentFromServer(filePath);
        }
    }

    private boolean testCacheValidity(String filePath) {
        // Code to send a TestAuth message and check response
        return true; // Simplified, actual implementation would involve network calls
    }

    private String getLocalCacheContent(String filePath) {
        // Code to retrieve local cache content
        return "cached content"; // Hypothetical implementation
    }

    private void fetchUpdatedContentFromServer(String filePath) {
        // Code to request updated content from the server
        System.out.println("Fetching latest version of " + filePath);
    }
}
```
x??

---

#### Server Recovery After a Crash
Background context: AFSv2 faces challenges during server recovery due to its caching mechanism. When the server crashes, it loses information about which client machine has which files, leading to potential issues when the server restarts.

:p What are the main challenges in recovering from a server crash in AFSv2?
??x
The main challenges include ensuring that each client realizes the server has crashed and treats all cached contents as potentially invalid. This necessitates re-establishing file validity before using them, which can be complex due to the distributed nature of AFS.

```java
// Pseudocode for handling server crash recovery in AFSv2
public void handleServerCrash() {
    // Notify clients that the server has crashed
    notifyClientsOfCrash();

    // Clients must invalidate their cache contents and revalidate files before use
    for (Client client : getClientList()) {
        client.invalidateCache();
        client.revalidateFiles();
    }
}
```
x??

---

#### Scalability and Performance of AFSv2 vs. NFS
Background context: AFSv2 was designed to address the scalability issues faced by its predecessor, NFS. The new protocol allowed for better performance and supported a larger number of clients.

:p How did AFSv2 improve scalability compared to NFS?
??x
AFSv2 improved scalability by supporting about 50 clients instead of just 20, as was common with the original version of NFS. This improvement was due to more efficient caching mechanisms and better handling of file access patterns.

```java
// Pseudocode for comparing AFSv2 performance with NFS
public void comparePerformance() {
    // Example scenarios: small/medium/large files, sequential/read/write
    FileAccessScenario[] scenarios = {new SmallFileSequentialRead(), new MediumFileSequentialReRead(), ...};
    
    for (FileAccessScenario scenario : scenarios) {
        System.out.println("Performance in " + scenario.getName());
        if (scenario.isNFS()) {
            printTime(scenario.getTimeNFS());
        } else {
            printTime(scenario.getTimeAFS2());
        }
    }
}
```
x??

---

#### Client-Side Performance and Local Access
Background context: In AFSv2, client-side performance often approached local performance due to the caching mechanism. For typical file access patterns, most reads were served from local disk caches or memory.

:p Why did client-side performance in AFSv2 often come close to local performance?
??x
Client-side performance was close to local performance because for small and medium files, which fit into client memory, all accesses could be handled locally. Large files, although not entirely cached locally, were mostly accessed from the local disk cache before needing a network request.

```java
// Pseudocode for handling file access in AFSv2
public void handleFileAccess(File file) {
    if (file.getSize() <= clientMemoryLimit) {
        // Small or medium file; try to serve from memory first
        if (clientCacheContains(file)) {
            return fileFromCache();
        } else {
            fetchFromLocalDisk();
        }
    } else {
        // Large file; always read from local disk cache first
        if (clientCacheContains(file)) {
            return fileFromCache();
        } else {
            fetchFromLocalDisk();
            if (fileNeedsUpdate) {
                updateFileOnServer();
            }
        }
    }
}
```
x??

---

#### Server Crash Recovery Mechanism
Background context: When the server crashes, AFSv2 clients must revalidate their cached files to ensure they are up-to-date. This is achieved either through a message from the server or periodic heartbeat messages.

:p How do AFSv2 clients handle server crash recovery?
??x
AFSv2 clients handle server crash recovery by treating all cached contents as potentially invalid and revalidating them before use. This can be done via:
- The server sending a "don’t trust your cache" message upon restart.
- Clients periodically checking the server's availability (heartbeats).

```java
// Pseudocode for client-side crash recovery handling
public void handleServerCrashRecovery() {
    // Option 1: Server sends a message
    if (serverIsAvailable()) {
        invalidateCache();
        revalidateAllFiles();
    }

    // Option 2: Client checks periodically with heartbeat messages
    while (!serverIsAvailable()) {
        Thread.sleep(heartbeatInterval);
    }
    invalidateCache();
    revalidateAllFiles();
}
```
x??

---

#### First Access to a File Does Not Hit Cache
Background context: In this scenario, we are observing file access times for NFS and AFS when there is no cache hit on the first read of a file. Subsequent reads might benefit from local caching.

:p What happens during the first access to a file in both NFS and AFS?
??x
During the first access, the time to fetch the file from the remote server dominates, making it similar for both systems. This is because neither system has cached copies of the file locally yet.
x??

---

#### Performance Comparison on Large File Re-Reads
Background context: The performance comparison focuses on large files that are re-read multiple times. AFS uses a local disk cache, whereas NFS only caches in client memory.

:p Why might AFS be faster than NFS during a large-file re-read?
??x
AFS is faster because it can access the file from its local disk cache. In contrast, NFS would need to fetch the entire file again from the remote server if the cached blocks do not fit into client memory.
??x

The performance difference arises due to AFS's ability to reuse data from a local cache, while NFS needs to re-fetch the whole file, especially for large files that exceed available memory.
x??

---

#### Sequential Writes Performance
Background context: This section discusses how both systems handle sequential writes of new files. Both systems buffer writes in their respective memory caches and eventually write to the server.

:p How do NFS and AFS handle sequential writes?
??x
Both NFS and AFS buffer writes in client-side memory but ensure that the data is eventually written to the server. However, AFS uses local file system caching, while NFS relies on a more general memory cache.
??x

For both systems:
- Writes are buffered in client-side memory.
- Data is flushed to the server when the file is closed (NFS) or periodically by the operating system (AFS).

Code Example: 
```java
// Pseudo-code for writing data using NFS and AFS
public void writeFile(FileSystem fs, String filePath, byte[] data) {
    try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(fs.open(filePath), "UTF-8"))) {
        writer.write(new String(data));
    } catch (IOException e) {
        e.printStackTrace();
    }
}
```
x??

---

#### Sequential File Overwrite Performance
Background context: This topic focuses on the performance impact of overwriting a sequential file. AFS might be slower because it needs to fetch the entire old file before overwriting.

:p Why does AFS perform worse in sequential file overwrites?
??x
AFS performs worse during file overwrites because the client must first fetch the entire old file from local storage, only to overwrite it. This additional step can significantly slow down the process.
??x

The overhead of fetching the entire file before overwriting can be substantial for large files, especially compared to a direct overwrite operation in NFS.

Code Example:
```java
// Pseudo-code for overwriting a file using AFS
public void overwriteFile(FileSystem fs, String filePath, byte[] newData) {
    try (BufferedInputStream fis = new BufferedInputStream(fs.open(filePath)); 
         FileOutputStream fos = new FileOutputStream(new File(filePath), false)) {
        int read;
        byte[] buffer = new byte[1024];
        while ((read = fis.read(buffer)) != -1) {
            // Store the data temporarily
            fos.write(buffer, 0, read);
        }
        fos.write(newData); // Overwrite with new data
    } catch (IOException e) {
        e.printStackTrace();
    }
}
```
x??

---

#### NFS vs AFS Performance Differences
Background context explaining the differences between NFS and AFS, focusing on their performance characteristics. NFS overwrites blocks to avoid initial reads, while AFS fetches entire files even for small accesses.

:p What is a key difference in how NFS and AFS handle file access?
??x
NFS overwrites blocks to avoid an initial read, whereas AFS fetches the entire file upon opening it, even if only a small part of the file is accessed. This leads to better performance for workloads that access only parts of large files on NFS.
x??

---

#### Workload Considerations in AFS Design
Background context about the importance of choosing appropriate workloads when designing storage systems and how assumptions affect system design.

:p How do workload assumptions impact the design choices in file systems like AFS?
??x
Workload assumptions significantly influence the design decisions. For example, AFS was designed under the assumption that files are rarely shared and accessed sequentially. This led to a design that fetches entire files upon opening but is inefficient for workloads involving small sequential reads or writes.
x??

---

#### AFS vs NFS Block Handling
Background context explaining how block-based protocols like NFS perform I/O proportional to read/write size, compared to AFS which always loads the whole file.

:p Why does NFS generally outperform AFS in scenarios where only a portion of large files is accessed?
??x
NFS performs I/O operations that are proportional to the size of the reads or writes, making it more efficient for accessing small portions of large files. In contrast, AFS always fetches the entire file upon opening, leading to unnecessary overhead and reduced performance in such scenarios.
x??

---

#### Global Namespace in AFS
Background context about the global namespace feature in AFS and how it differs from NFS.

:p How does AFS ensure a consistent naming system across clients?
??x
AFS provides a true global namespace to clients, ensuring that all files are named consistently on all client machines. This is different from NFS, which allows each client to mount servers independently, leading to potential inconsistencies in file names unless carefully managed.
x??

---

#### Improvements in AFS
Background context about the enhancements and features added by the designers of AFS to improve usability.

:p What feature does AFS provide that simplifies managing files across different clients?
??x
AFS provides a true global namespace, ensuring that all files are named consistently on all client machines. This contrasts with NFS, where each client can mount servers independently, leading to inconsistencies in file naming unless managed through convention and administrative effort.
x??
--- 

These flashcards cover key concepts from the provided text related to AFS versus NFS performance, workload assumptions, block handling, global namespaces, and system enhancements.

#### AFS Security Mechanisms
AFS incorporates mechanisms to authenticate users and ensure that a set of files can be kept private if a user so desires. In contrast, NFS had primitive security support for many years.

:p How does AFS provide better security compared to NFS?
??x
AFS provides advanced security features such as user authentication and flexible access control lists (ACLs), allowing users to precisely manage who has access to which files. This is in stark contrast to NFS, which historically lacked robust security mechanisms until later versions like NFSv4 began to incorporate similar features.

```java
// Example of setting ACL in AFS
public void setAccessControlList(String filePath, String permission) {
    // Logic to update AFS for specific file permissions
    AFSManager.setPermissions(filePath, permission);
}
```
x??

---

#### AFS User Management and Flexibility
AFS allows users to have a great deal of control over who can access which files. This is facilitated through flexible user-managed access controls.

:p How does AFS enable more flexible file access management compared to traditional Unix file systems?
??x
In AFS, users have fine-grained control over permissions and access rights, enabling them to manage who can view or modify specific files independently of the underlying operating system. This is achieved through mechanisms like ACLs which allow for dynamic and granular permission settings.

```java
// Example of setting file permissions in AFS
public void grantAccess(String filePath, String username, String permissions) {
    // Logic to add a user with specific permissions to the file's ACL
    AFSManager.grantAccess(filePath, username, permissions);
}
```
x??

---

#### NFS vs. AFS Security
NFS historically lacked robust security mechanisms compared to AFS, which provided both authentication and fine-grained access controls.

:p How does AFS improve upon the security shortcomings of NFS?
??x
AFS enhances security by offering mechanisms for user authentication and flexible access control policies. Unlike NFS, where early versions were notoriously vulnerable, AFS was designed with a focus on secure file sharing, ensuring that users could define detailed permissions on files and directories.

```java
// Example of configuring NFS server to enable secure file transfers
public void configureNFSForSecurity(String mountPoint, String server) {
    // Logic to set up NFS security options such as authentication methods
    NFSConfig.setSecurityOptions(mountPoint, server);
}
```
x??

---

#### AFS Management Tools for Administrators
AFS provides tools that make it easier and more efficient for system administrators to manage servers compared to the traditional approaches used with NFS.

:p How does AFS simplify server management for administrators?
??x
AFS includes advanced administrative tools that streamline the process of managing distributed file systems. These tools reduce the complexity and overhead associated with maintaining multiple servers, allowing administrators to support a larger number of clients per server more efficiently.

```java
// Example of configuring AFS servers for better management
public void configureAFSServer(String host, String port) {
    // Logic to set up AFS servers with optimized configurations
    AFSAdmin.configureServer(host, port);
}
```
x??

---

#### AFS Protocol Design and Performance
The protocol design of AFS is designed to minimize server interactions by leveraging whole-file caching and callbacks. This approach allows each server to support many clients, reducing the number of servers needed.

:p How does AFS’s protocol design contribute to its performance benefits?
??x
AFS’s protocol minimizes server interactions through techniques like whole-file caching and callback mechanisms. These features enable a single server to handle more client requests by caching files locally, thus reducing network overhead and improving overall system performance.

```java
// Example of implementing a cache hit/miss logic in AFS
public boolean checkCacheHit(String filePath) {
    // Logic to check if the file is cached and return true or false
    if (isCached(filePath)) {
        return true;
    }
    return false;
}
```
x??

---

#### AFS Consistency Model
AFS offers a simple consistency model that is easy to understand and reason about, unlike NFS which can sometimes exhibit weird behavior.

:p How does the consistency model in AFS differ from NFS?
??x
The consistency model in AFS is straightforward and predictable. It ensures that file operations are completed before any changes are propagated, making it easier for users to manage and predict file system states. In contrast, NFS has been known to exhibit inconsistent behavior at times, particularly with network interruptions or server crashes.

```java
// Example of ensuring consistency during file operations in AFS
public void ensureFileConsistency(String filePath) {
    // Logic to make sure the file is consistent after an operation
    if (AFSManager.isConsistent(filePath)) {
        System.out.println("File is consistent.");
    } else {
        System.out.println("File needs a consistency check.");
    }
}
```
x??

---

#### AFS Declining Popularity and Evolution
Although still seen in some environments, AFS's influence on modern distributed file systems may come more from the ideas it pioneered rather than its current usage.

:p Why is AFS likely declining in popularity?
??x
AFS is likely declining because NFS became an open standard with widespread support across many vendors. As a result, NFS and CIFS (the Windows-based protocol) dominate the market. While AFS had advanced features like flexible access control and server management tools, its proprietary nature limited broader adoption.

```java
// Example of checking if AFS is still in use in an environment
public boolean checkAFSInstallation() {
    // Logic to determine if AFS is currently installed or used
    return AFSManager.isInstalled();
}
```
x??

---

#### AFS System Overview
Background context: The provided reference discusses the Andrew File System (AFS), which is a distributed file system. It was first described in the paper "The ITC Distributed File System: Principles and Design" by M. Satyanarayanan et al., presented at SOSP '85. AFS is still in use today, with improvements over time.
:p What are the key features of the Andrew File System (AFS)?
??x
The key features include distributed access control, automatic file replication, and a hierarchical namespace. These features allow for efficient and secure file sharing across multiple machines.
x??

---

#### AFS Simulation with `afs.py`
Background context: The homework involves using an AFS simulator (`afs.py`) to understand how the system works. This includes predicting client reads, server callbacks, cache states, and different workloads.
:p How do you run simple cases in the AFS simulation?
??x
To run simple cases, use the following command with various flags:
```sh
python afs.py -s <seed> -f <num_files> -C <num_clients> -r <read_ratio> -n <trace_length>
```
The `-s` flag sets the random seed for reproducibility, `-f` specifies the number of files, `-C` indicates the number of clients, and `-r` controls the read ratio. You can vary these to observe different behaviors.
x??

---

#### Predicting Callbacks
Background context: The simulation allows you to predict callbacks initiated by the AFS server. This involves understanding the timing and conditions under which these callbacks occur.
:p How do you predict each callback in the AFS simulation?
??x
To predict each callback, run the program with detailed feedback:
```sh
python afs.py -d 3 -c
```
The `-d 3` flag provides high-level detailed output, helping to understand when and why callbacks occur. You can use this information to guess the exact timing.
x??

---

#### Cache States in AFS Simulation
Background context: The cache state is crucial for understanding how data is managed across clients and servers. Detailed cache states can be observed using certain flags.
:p How do you predict the cache state at each step in the simulation?
??x
To observe the cache state, run the command with:
```sh
python afs.py -c -d 7
```
The `-c` flag enables cache tracing, and `-d 7` provides detailed debug output. This helps track changes in the cache across different steps.
x??

---

#### Specific Workloads with AFS Simulation
Background context: The simulation allows you to run specific workloads that involve both read and write operations from multiple clients. Understanding these can help predict outcomes based on different scheduling policies.
:p How do you construct a workload for client 1 reading file `a`?
??x
To simulate this, use the following command:
```sh
python afs.py -A oa1:w1:c1,oa1:r1:c1
```
This schedule indicates that Client 1 will write to `a` once and then read from it. To see different outcomes, vary the random seed.
x??

---

#### Specific Schedules in AFS Simulation
Background context: Different scheduling policies can lead to varying outcomes for client reads. Understanding these schedules helps predict final file values based on the sequence of operations.
:p How do you run with specific schedule interleavings?
??x
To test with specific schedules, use the `-S` flag followed by the desired pattern:
```sh
python afs.py -A oa1:w1:c1,oa1:r1:c1 -S 01,-S 100011 ,-S 011100
```
These patterns control the order of operations. For instance, `-S 01` means a read operation followed by a write.
x??

---

#### Final Values with Varying Schedules
Background context: The final value in AFS can depend on the interleaving of read and write operations from different clients. Understanding these interactions is crucial for predicting outcomes.
:p How do you determine the final value when running with `-S 011100`?
??x
To find out, run the simulation with:
```sh
python afs.py -A oa1:w1:c1,oa1:w1:c1 -S 011100
```
This schedule indicates a read followed by two writes. The final value depends on which write operation wins based on AFS's concurrency control mechanism.
x??

---

