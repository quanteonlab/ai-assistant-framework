# Flashcards: 029---Understanding-Distributed-Systems-2021-Roberto-Vitillo--_processed (Part 16)

**Starting Chapter:** Geo load balancing

---

#### L7 Load Balancing with Sidecar Pattern

Background context: In modern microservices architectures, managing load balancing and other network-related functionalities can be complex. The traditional approach involves a dedicated load balancer service that needs to be scaled out and maintained separately from the application services.

The sidecar pattern addresses this by integrating these functionalities directly into the clients (in this case, applications) as processes running alongside them. This approach offloads the load balancing responsibilities from a central service to individual instances of the application, reducing the complexity and scaling requirements for the centralized load balancer.

:p How does the sidecar pattern work in load balancing?
??x
The sidecar pattern works by placing a process (sidecar) on each application node that handles network traffic. This process implements various functionalities such as load balancing, rate limiting, authentication, monitoring, etc., directly within the application's deployment environment. By doing so, it distributes these responsibilities across all nodes instead of relying on a single, potentially bottlenecked service.

This approach leverages proxies like NGINX, HAProxy, or Envoy to handle network traffic locally. For example, when an HTTP request comes in, the sidecar process first handles authentication and rate limiting before forwarding the request to the appropriate backend service.

```java
// Example of a simplified sidecar proxy logic
public class SidecarProcess {
    public void handleRequest(HttpRequest request) {
        // Check if the client is authenticated
        boolean isAuthenticated = authenticateClient(request);
        
        // Rate limit requests based on client IP and time window
        int remainingRequests = rateLimitRequest(request);
        
        // If authentication and rate limiting are successful, forward to backend service
        if (isAuthenticated && remainingRequests > 0) {
            String response = forwardToBackend(request);
            sendResponse(response);
        } else {
            logFailedRequest(request);
            rejectClient();
        }
    }
}
```
x??

---

#### Geo Load Balancing

Background context: Even with local load balancing, there can be latency issues when clients are located far from the server. Geo load balancing distributes traffic to data centers based on geographical proximity to reduce response times and improve performance.

DNS geoloadbalancing is an extension of DNS that considers a client's location (inferred from its IP) and returns a list of the closest L4 load balancers. The load balancer also needs to take into account the capacity and health status of each data center before routing traffic.

:p How does Geo Load Balancing ensure clients communicate with the geographically closest load balancer?
??x
Geo load balancing ensures that clients communicate with the geographically closest load balancer by leveraging DNS resolution techniques. The DNS server returns a list of IP addresses (VIPs) associated with the nearest data center based on the client's inferred location.

This process involves:
1. Determining the client's geographical location from its IP address.
2. Consulting a predefined mapping to find the closest data centers.
3. Returning the VIPs of these data centers in response to DNS queries.

For example, if a client is located in New York and querying for a service hosted globally, the DNS server might return the VIP of a load balancer in New York or another nearby region rather than one on the other side of the world.

```java
// Pseudocode for Geo Load Balancing logic
public class GeoLoadBalancer {
    public List<String> resolveNearestLoadBalancers(String service) {
        // Step 1: Determine client's geographical location based on IP address
        Location clientLocation = determineClientLocation(clientIP);
        
        // Step 2: Map the client's location to a list of nearest data centers
        List<DataCenter> nearestDataCenters = findNearestDataCenters(clientLocation);
        
        // Step 3: Return VIPs of these data centers
        return getLoadBalancerVIPs(nearestDataCenters);
    }
}
```
x??

--- 

Note: The Java code examples are simplified and serve to illustrate the concepts rather than providing a complete implementation.

#### Single Leader Replication
Background context: This is a common approach to replicating data where client writes are exclusively sent to one leader node, which updates its local state and then asynchronously or synchronously replicates changes to followers. The Raft algorithm is an example of implementing this strategy.

:p What is the single leader replication model?
??x
In the single leader replication model, client writes are exclusively directed to a leader node, which handles the write operation and then asynchronously or synchronously updates its state before responding to the client. The leader then replicates these changes to one or more follower nodes.
??x

---

#### Asynchronous Replication
Background context: In asynchronous replication mode, when the leader receives a write request from a client, it sends out requests to followers for replication and responds back to the client even if the data hasn't been fully replicated. This approach is faster but not fault-tolerant.

:p What happens in an asynchronous replication scenario?
??x
In asynchronous replication, after receiving a write request from a client, the leader sends asynchronous requests to the followers to replicate the change. It responds back to the client before ensuring that the data has been fully replicated on all followers. This can lead to issues like data loss if the leader crashes before replicating the writes.
??x

---

#### Synchronous Replication
Background context: In synchronous replication, a write request is not considered complete until it is acknowledged by multiple replicas. This ensures strong consistency but may be slower due to waiting for all replicas.

:p What distinguishes synchronous replication from asynchronous?
??x
Synchronous replication requires that a write request be acknowledged by the majority of replicas before returning confirmation to the client. This approach provides stronger consistency guarantees but can introduce latency because it waits for multiple nodes to confirm the update.
??x

---

#### Trade-offs in Replication Strategies
Background context: Different replication strategies offer varying trade-offs between performance, fault tolerance, and data consistency.

:p What are the key trade-offs in replication strategies?
??x
Replication strategies balance factors such as performance (asynchronous vs. synchronous), fault tolerance (single leader with multiple followers), and data consistency (eventual vs. strong). Asynchronous replication is faster but can lead to inconsistencies if the leader fails, while synchronous ensures stronger consistency but introduces latency.
??x

#### Synchronous and Asynchronous Replication
Synchronous replication ensures that a write operation is acknowledged only after it has been successfully written to all replicas. Conversely, asynchronous replication allows a write to be acknowledged immediately, with the replica catching up later. This can lead to performance penalties but offers flexibility in handling high write throughput.
:p What are the key differences between synchronous and asynchronous replication?
??x
Synchronous replication requires all replicas to acknowledge a write operation before returning confirmation to the client. This ensures data consistency but incurs higher latency due to waiting for acknowledgments from each replica. Asynchronous replication, on the other hand, returns acknowledgment immediately after writing locally, allowing the system to handle more writes per second at the cost of potential data inconsistency in case of a failure.
x??

---

#### Multi-Leader Replication
Multi-leader replication allows multiple nodes to accept write operations, providing higher write throughput and redundancy across different geographic locations. However, it introduces complexity due to the possibility of conflicting writes when two leaders update the same piece of data simultaneously.
:p What is multi-leader replication, and what are its primary challenges?
??x
Multi-leader replication involves having multiple nodes capable of accepting write operations independently. This setup increases write throughput and can improve availability by distributing writes across different geographical locations. However, it poses a significant challenge in resolving conflicting writes that occur when two leaders update the same data item concurrently.
x??

---

#### Conflict Resolution Strategies in Multi-Leader Replication
Conflict resolution is crucial in multi-leader replication to handle situations where multiple nodes attempt to modify the same piece of data simultaneously. Common strategies include designing systems to avoid conflicts, using timestamps, or leveraging logical clocks for more reliable conflict detection and resolution.
:p What are some common methods used to resolve conflicts in multi-leader replication?
??x
Common methods for resolving conflicts in multi-leader replication include:
1. **Avoiding Conflicts**: Design the system so that conflicts can be prevented. For example, routing all requests from a specific region to a single leader within that region.
2. **Timestamp-Based Resolution**: Using timestamps to determine which write is more recent and should take precedence. However, this method may not be entirely reliable due to clock skew between nodes.
3. **Logical Clocks**: Implementing logical clocks to provide more accurate and consistent conflict resolution across distributed systems.
x??

---

#### Handling Conflicts in Multi-Leader Replication
In multi-leader replication, when a client requests a write operation that could potentially conflict with another leader's updates, the system must handle these conflicts. One approach is to store concurrent writes and present them to the next reader, allowing the application logic to resolve them.
:p How does the "push the can down the road" method work in resolving conflicts?
??x
The "push the can down the road" method involves storing conflicting writes and returning them to the client that reads the data later. The client then resolves these conflicts by updating the database with its chosen solution, effectively passing the responsibility of conflict resolution back to the application layer.
x??

---

#### Advanced Conflict Resolution Methods
Advanced conflict resolution methods include using logical clocks or custom conflict resolution algorithms provided by clients. These mechanisms help in reliably detecting and resolving conflicts without relying solely on timestamps that might not be perfectly synchronized across nodes.
:p What are some advanced techniques for conflict resolution in multi-leader replication?
??x
Some advanced techniques for conflict resolution in multi-leader replication include:
1. **Logical Clocks**: Using logical clocks to ensure more reliable detection of causality and resolving conflicts based on the order of events, rather than relying solely on timestamps.
2. **Custom Conflict Resolution Algorithms**: Allowing clients to provide their own logic for resolving conflicts when they encounter concurrent writes, enabling fine-grained control over how data is managed in distributed systems.
x??

---

