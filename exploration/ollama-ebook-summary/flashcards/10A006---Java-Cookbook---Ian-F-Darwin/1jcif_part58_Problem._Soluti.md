# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 58)

**Starting Chapter:** Problem. Solution. Discussion

---

#### ServerSocket Basics
In Java, `ServerSocket` is used to create a socket-based server that waits for client connections. It represents one end of a connection and is typically instantiated with just the port number. The server listens on the specified port for incoming client connections.

:p How do you initialize a `ServerSocket` in Java?
??x
You can initialize a `ServerSocket` by specifying the port number as follows:

```java
public class Listen {
    public static final short PORT = 9999;

    public static void main(String[] argv) throws IOException {
        ServerSocket sock;
        Socket clientSock;
        try {
            sock = new ServerSocket(PORT);
            while ((clientSock = sock.accept()) != null) {
                // Process it, usually on a separate thread
                process(clientSock);
            }
        } catch (IOException e) {
            System.err.println(e);
        }
    }

    static void process(Socket s) throws IOException {
        System.out.println("Accept from client " + s.getInetAddress());
        // The conversation would be here.
        s.close();
    }
}
```
x??

---

#### Listening on Specific Network Interfaces
Java `ServerSocket` can listen on a specific network interface by providing the address of that interface to its constructor. This allows services to be exposed only to certain networks, enhancing security and functionality.

:p How do you specify a network interface when creating a `ServerSocket`?
??x
To create a `ServerSocket` that listens on a specific network interface, use the following code:

```java
public class ListenInside {
    public static final short PORT = 9999;
    public static final String INSIDE_HOST = "acmewidgets-inside";
    public static final int BACKLOG = 10;

    public static void main(String[] argv) throws IOException {
        ServerSocket sock;
        Socket clientSock;
        try {
            sock = new ServerSocket(PORT, BACKLOG,
                    InetAddress.getByName(INSIDE_HOST));
            while ((clientSock = sock.accept()) != null) {
                // Process it.
                process(clientSock);
            }
        } catch (IOException e) {
            System.err.println(e);
        }
    }

    static void process(Socket s) throws IOException {
        System.out.println("Connected from " + INSIDE_HOST + ": "
                + s.getInetAddress());
        // The conversation would be here.
        s.close();
    }
}
```
x??

---

#### Port Number Restrictions
Certain ports are reserved for specific services, and on server-based operating systems, ports below 1024 require root or administrator privilege to use. This was an early security mechanism but provides little real security in today's environment.

:p What restrictions exist when using ports below 1024?
??x
Ports below 1024 are considered privileged and typically require root/administrator privileges to create a `ServerSocket`. This restriction is largely due to historical reasons and provides minimal security benefits in modern, single-user desktop environments. Example:

```java
// Code example for using non-privileged ports (above 1024)
public class Listen {
    public static final short PORT = 9999;

    public static void main(String[] argv) throws IOException {
        ServerSocket sock;
        Socket clientSock;
        try {
            sock = new ServerSocket(PORT);
            while ((clientSock = sock.accept()) != null) {
                // Process it, usually on a separate thread
                process(clientSock);
            }
        } catch (IOException e) {
            System.err.println(e);
        }
    }

    static void process(Socket s) throws IOException {
        System.out.println("Accept from client " + s.getInetAddress());
        // The conversation would be here.
        s.close();
    }
}
```
x??

---

#### Overview of Java Enterprise Edition (Java EE)
Java EE provides support for building scalable and well-structured, multitiered distributed applications. It includes features like the servlet framework, JSP, JSF, EJB3 remote access, and Java Messaging Service.

:p What are some key components provided by Java EE?
??x
Java EE offers several important components:
- **Servlet Framework**: A strategy object that can be installed into any standard Java EE web server.
- **JSP (JavaServer Pages)**: An original technology for building dynamic web pages.
- **JSF (JavaServer Faces)**: A newer, component-based approach to creating user interfaces.

Example code using JSP:

```java
// JSP Example
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>
<head>
    <title>Welcome</title>
</head>
<body>
<h1>Hello, World!</h1>
</body>
</html>
```
x??

---

#### ServerSocket Accept Method
The `accept()` method of `ServerSocket` blocks until a client connects. Once a connection is established, it returns a `Socket` object that can be used for both reading and writing.

:p What does the `accept()` method do?
??x
The `accept()` method of `ServerSocket` waits (blocks) until a client makes a connection request. When a client connects, `accept()` returns a `Socket` object representing the established connection. This object can be used to send and receive data.

Example:

```java
public class Listen {
    public static final short PORT = 9999;

    public static void main(String[] argv) throws IOException {
        ServerSocket sock;
        Socket clientSock;
        try {
            sock = new ServerSocket(PORT);
            while ((clientSock = sock.accept()) != null) {
                // Process it, usually on a separate thread
                process(clientSock);
            }
        } catch (IOException e) {
            System.err.println(e);
        }
    }

    static void process(Socket s) throws IOException {
        System.out.println("Accept from client " + s.getInetAddress());
        // The conversation would be here.
        s.close();
    }
}
```
x??

---

---
#### Digital Equipment Absorption by Compaq and HP
Background context: This information is part of a broader discussion on networking interfaces and their management. The absorption of companies like Digital Equipment by Compaq, and subsequently by HP, illustrates how technological and corporate changes can impact naming conventions in network interfaces.

:p How does the absorption of Digital Equipment by Compaq and then by HP affect the naming of network interfaces?
??x
The absorption of Digital Equipment by Compaq and later by HP did not significantly impact the naming conventions for network interfaces. The engineers responsible for such systems typically do not change names as a result of corporate mergers, so the interface names may continue to reflect the original company’s nomenclature.
x??
---
#### InetAddress.getByName()
Background context: This method is used in Java to resolve hostnames into IP addresses. It can read from various sources including configuration files or DNS resolvers.

:p How does `InetAddress.getByName()` function in Java?
??x
`InetAddress.getByName()` is a method that looks up the given hostname and returns an `InetAddress` object representing its IP address. This process involves checking system-dependent locations such as `/etc/hosts`, Windows configuration files, or DNS resolvers.

Example code:
```java
try {
    InetAddress address = InetAddress.getByName("example.com");
    System.out.println(address.getHostAddress());
} catch (UnknownHostException e) {
    e.printStackTrace();
}
```
x??
---
#### NetworkInterface Class for Finding Network Interfaces
Background context: The `NetworkInterface` class in Java provides methods to list and query network interfaces on a system. This is useful for obtaining information about multiple network connections, which can be necessary on servers or systems with multiple network cards.

:p How does the `NetworkInterface` class help in finding out networking arrangements?
??x
The `NetworkInterface` class helps find out the computer's networking arrangement by listing all local interfaces and providing methods to retrieve associated addresses. This is particularly useful for server environments where specific network interfaces need to be identified based on their roles or configurations.

Example code:
```java
import java.net.NetworkInterface;
import java.util.Enumeration;

public class NetworkInfo {
    public static void main(String[] args) throws Exception {
        Enumeration<NetworkInterface> nets = NetworkInterface.getNetworkInterfaces();
        while (nets.hasMoreElements()) {
            NetworkInterface netIf = nets.nextElement();
            System.out.println("Name: " + netIf.getName());
        }
    }
}
```
x??
---
#### Example of Using `NetworkInterface` Class
Background context: The provided code demonstrates how to use the `NetworkInterface` class to print out network interface names on a system.

:p What does the program in Example 13-3 do when run?
??x
The program in Example 13-3 lists all local network interfaces by printing their names. If the machine is named "laptop," it will also print its network address; otherwise, it suggests modifying the code to accept the name from the command line.

Example of modified code:
```java
import java.net.NetworkInterface;
import java.util.Enumeration;

public class NetworkInfo {
    public static void main(String[] args) throws Exception {
        Enumeration<NetworkInterface> nets = NetworkInterface.getNetworkInterfaces();
        while (nets.hasMoreElements()) {
            NetworkInterface netIf = nets.nextElement();
            System.out.println("Name: " + netIf.getName());
            
            // Additional logic to print the network address if needed
            if ("laptop".equals(System.getenv("HOSTNAME"))) {
                for (byte b : netIf.getInetAddresses()) {
                    System.out.println("IP Address: " + b);
                }
            }
        }
    }
}
```
x??
---

#### Network Interface Discovery and Address Retrieval
This section discusses how to list network interfaces on a machine, retrieve their display names, and get their associated IP addresses. The `NetworkInterface` class provides methods like `getDisplayName()` for interface name retrieval and `getInetAddresses()` for getting the IP addresses.
:p What is the purpose of the code snippet in Example 13-3?
??x
The code lists all network interfaces on a machine, prints their display names, and retrieves their associated IP addresses. This example demonstrates how to enumerate and access details about network interfaces using Java's `NetworkInterface` class.
```java
public static void main(String[] a) throws IOException {
    Enumeration<NetworkInterface> list = NetworkInterface.getNetworkInterfaces();
    while (list.hasMoreElements()) {
        // Get one NetworkInterface
        NetworkInterface iface = list.nextElement();
        // Print its name
        System.out.println(iface.getDisplayName());
        Enumeration<InetAddress> addrs = iface.getInetAddresses();
        // And its address(es)
        while (addrs.hasMoreElements()) {
            InetAddress addr = addrs.nextElement();
            System.out.println(addr);
        }
    }

    // Try to get the Interface for a given local (this machine's) address
    InetAddress destAddr = InetAddress.getByName("laptop");
    try {
        NetworkInterface dest = NetworkInterface.getByInetAddress(destAddr);
        System.out.println("Address for " + destAddr + " is " + dest);
    } catch (SocketException ex) {
        System.err.println("Couldn't get address for " + destAddr);
    }
}
```
x??

---
#### Echo Server Implementation
This section explains how to implement a simple echo server in Java. An echo server listens on a specified port and echoes back any message it receives from the client.
:p What does the `EchoServer` class do?
??x
The `EchoServer` class implements an echo server that listens for incoming client connections, reads messages sent by clients, and sends them back to the client. It handles one complete connection at a time before accepting another connection.
```java
public static void main(String[] args) {
    int p = ECHOPORT;
    if (args.length == 1) {
        try {
            p = Integer.parseInt(args[0]);
        } catch (NumberFormatException e) {
            System.err.println("Usage: EchoServer [port#]");
            System.exit(1);
        }
    }
    new EchoServer(p).handle();
}

public EchoServer(int port) {
    try {
        sock = new ServerSocket(port);
    } catch (IOException e) {
        System.err.println("I/O error in setup");
        System.err.println(e);
        System.exit(1);
    }
}

protected void handle() {
    Socket ios = null;
    while (true) {
        try {
            System.out.println("Waiting for client...");
            ios = sock.accept();
            System.err.println("Accepted from " + ios.getInetAddress().getHostName());
            try (
                BufferedReader is = new BufferedReader(new InputStreamReader(ios.getInputStream(), "8859_1"));
                PrintWriter os = new PrintWriter(new OutputStreamWriter(ios.getOutputStream(), "8859_1"), true);
            ) {
                String echoLine;
                while ((echoLine = is.readLine()) != null) {
                    System.err.println("Read " + echoLine);
                    os.print(echoLine + "\r");
                    os.flush();
                    System.err.println("Wrote " + echoLine);
                }
                System.err.println("All done.");
            }
        } catch (IOException e) {
            System.err.println(e);
        }
    }
}
```
x??

---
#### Daytime Binary Server Implementation
This section provides an example of a binary-based server that sends the current time as a 32-bit integer to clients. It demonstrates handling multiple client connections in a loop.
:p What does the `DaytimeServer` class do?
??x
The `DaytimeServer` class implements a binary-based echo server on port 37, which sends the current time in seconds since the epoch (1970-01-01) as a 32-bit integer to clients. It handles multiple client connections in a loop.
```java
public static void main(String[] argv) {
    new DaytimeServer(PORT).runService();
}

public DaytimeServer(int port) {
    try {
        sock = new ServerSocket(port);
    } catch (IOException e) {
        System.err.println("I/O error in setup " + e);
        System.exit(1);
    }
}

protected void runService() {
    Socket ios = null;
    DataOutputStream os = null;
    while (true) {
        try {
            System.out.println("Waiting for connection on port " + PORT);
            ios = sock.accept();
            System.err.println("Accepted from " + ios.getInetAddress().getHostName());
            os = new DataOutputStream(ios.getOutputStream());
            long time = System.currentTimeMillis() / 1000; // Daytime Protocol is in seconds
            // Convert to Java time base.
            time += RDateClient.BASE_DIFF;
            // Write it, truncating cast to int since it is using the Internet Daytime protocol which uses 4 bytes.
            os.writeInt((int) time);
            os.close();
        } catch (IOException e) {
            System.err.println(e);
        }
    }
}
```
x??

---

