# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 59)

**Starting Chapter:** 13.4 Returning Object Information Across a Network Connection. Problem. Solution. 13.5 Handling Multiple Clients

---

---
#### Network Object Transfer
Background context: In Java, you can return an object across a network connection using serialization. Serialization is the process of converting objects into streams of bytes so that they can be stored or transmitted and then reconstructed back into objects.

This example uses `ObjectOutputStream` to serialize and send an object over a network socket.

:p How does the DaytimeObjectServer class handle incoming connections to return an object?
??x
The server listens for incoming connections on a specific port. When a connection is accepted, it constructs a new `LocalDateTime` object representing the current time and writes this object using an `ObjectOutputStream`. The object is then serialized and sent back to the client.

```java
public class DaytimeObjectServer {
    public static final short TIME_PORT = 1951; // Define the port for the service

    public static void main(String[] argv) {
        ServerSocket sock;
        Socket clientSock;

        try {
            sock = new ServerSocket(TIME_PORT); // Create a server socket
            while ((clientSock = sock.accept()) != null) { // Accept incoming connections
                System.out.println("Accept from " + clientSock.getInetAddress()); // Print the client's address

                ObjectOutputStream os = new ObjectOutputStream(clientSock.getOutputStream()); // Get output stream and create ObjectOutputStrea
                // Write the LocalDateTime object to the output stream
                os.writeObject(LocalDateTime.now());
                os.close(); // Close the output stream
            }
        } catch (IOException e) {
            System.err.println(e); // Handle any I/O exceptions that occur
        }
    }
}
```
x??
---

#### Object Serialization Process
Background context: The `ObjectOutputStream` is a part of Java's serialization framework. It writes objects to an output stream in a way that can be reconstructed at the receiving end.

:p What method does the server use to write the current date and time as an object?
??x
The server uses the `writeObject()` method from the `ObjectOutputStream`. This method serializes the specified object into the underlying output stream. In this case, it writes a `LocalDateTime` object representing the current date and time.

```java
// Code example
os.writeObject(LocalDateTime.now());
```
x??
---

#### Handling Client-Side Connection
Background context: The client-side code would typically involve creating an input/output stream connection to the server, reading the serialized object from this stream, and then deserializing it back into a usable form.

:p How does the server handle each incoming connection in terms of data transfer?
??x
For each incoming connection, the server accepts the socket, prints the client's address, creates an `ObjectOutputStream` to write a new `LocalDateTime` object representing the current time, and then closes the stream. This process is repeated for every new client connection.

```java
// Code example within while loop
System.out.println("Accept from " + clientSock.getInetAddress());
ObjectOutputStream os = new ObjectOutputStream(clientSock.getOutputStream()); // Create output stream
os.writeObject(LocalDateTime.now()); // Write LocalDateTime object to stream
os.close(); // Close the output stream after writing
```
x??
---

#### Exception Handling in Server Code
Background context: Proper exception handling is crucial in network programming to ensure that any issues are caught and handled gracefully, preventing the program from crashing.

:p How does the server handle potential I/O exceptions during its operation?
??x
The server handles potential I/O exceptions by catching `IOException` using a try-catch block. If an `IOException` occurs, it is printed to the standard error stream using `System.err.println(e)`.

```java
try {
    sock = new ServerSocket(TIME_PORT);
    while ((clientSock = sock.accept()) != null) {
        // Code for handling client connections
    }
} catch (IOException e) {
    System.err.println(e); // Handle and print any I/O exceptions that occur
}
```
x??
---

---
#### Handling Multiple Clients with Threads
In Java, to handle multiple clients, one approach is to use a thread for each client connection. This method is different from C's `select()` or `poll()` mechanism which informs the server about available file/socket descriptors.

Java does not provide `select()` due to platform limitations but instead relies on threads for handling concurrent connections. Each time a new connection is accepted, Java constructs and starts a new thread object to handle that client.

:p How can a Java-based server handle multiple clients using threads?
??x
A Java-based server handles multiple clients by accepting each new connection and starting a new thread to process it. Here’s an example of how the `runServer` method works:

```java
/** Run the main loop of the Server. */
void runServer( ) {
    while (true) {
        try {
            Socket clntSock = sock.accept(); // Accepts a new client connection
            new Handler(clntSock).start();  // Starts a new thread to handle this connection
        } catch(IOException e) {
            System.err.println(e);         // Prints the exception if any error occurs during acceptance
        }
    }
}
```

This code uses a loop that continuously accepts connections and creates a new `Handler` thread for each client. The `Handler` class, which extends `Thread`, processes the client's requests.

x??
---
#### Thread Subclassing in Java
To use threads in Java to handle multiple clients, one approach is to subclass the `Thread` class. If you choose this method, your handler class must extend `Thread`.

:p How does a thread subclass work in handling client connections in Java?
??x
A thread subclass works by extending the `Thread` class and overriding its methods or implementing necessary functionality within it. For example:

```java
class Handler extends Thread {
    private Socket sock;

    public Handler(Socket s) {
        this.sock = s;
    }

    @Override
    public void run() {
        try (BufferedReader in = new BufferedReader(new InputStreamReader(sock.getInputStream()));
             PrintWriter out = new PrintWriter(sock.getOutputStream(), true)) {

            String inputLine;
            while ((inputLine = in.readLine()) != null) {
                // Process the input line and send back an echo response
                out.println(inputLine);
            }
        } catch (IOException e) {
            System.err.println("Error handling client: " + e.getMessage());
        }
    }
}
```

In this example, `Handler` is a subclass of `Thread`. The `run()` method processes the input and sends an echo response back to the client. Each new connection creates a new instance of `Handler`, which starts processing as soon as it's created.

x??
---
#### Using Runnable Interface for Threads
Another way to use threads in Java is by implementing the `Runnable` interface instead of subclassing `Thread`. If you choose this method, you pass an instance of your `Runnable` object to a new `Thread` constructor and start it.

:p How can we handle client connections using the `Runnable` interface?
??x
Using the `Runnable` interface involves creating a class that implements `Runnable` and then passing its instance to a `Thread` constructor. Here's an example:

```java
class Handler implements Runnable {
    private Socket sock;

    public Handler(Socket s) {
        this.sock = s;
    }

    @Override
    public void run() {
        try (BufferedReader in = new BufferedReader(new InputStreamReader(sock.getInputStream()));
             PrintWriter out = new PrintWriter(sock.getOutputStream(), true)) {

            String inputLine;
            while ((inputLine = in.readLine()) != null) {
                // Process the input line and send back an echo response
                out.println(inputLine);
            }
        } catch (IOException e) {
            System.err.println("Error handling client: " + e.getMessage());
        }
    }
}
```

In this example, `Handler` implements the `Runnable` interface. The `run()` method is where you process the client's input and send back a response. A new thread is created by passing an instance of `Handler` to a `Thread` constructor:

```java
Thread t = new Thread(new Handler(clntSock));
t.start();
```

This approach also uses a loop in the `run()` method to continuously read from the client until there are no more lines.

x??
---

---
#### Echo Server Threaded Design
The server is designed to handle client connections concurrently using threads. Each client connection results in a new thread being created, allowing for parallel processing of client requests. The `Handler` class extends `Thread` and manages communication with each client.

:p What is the purpose of creating a separate thread for handling each client?
??x
The purpose is to enable concurrent processing of multiple clients, ensuring that responses are sent back to their respective clients without blocking other connections. This allows the server to handle many clients simultaneously.
```
public class Handler extends Thread {
    Socket sock;
    Handler(Socket s) { sock = s; }
    public void run() {
        try (BufferedReader is = new BufferedReader(new InputStreamReader(sock.getInputStream()));
             PrintStream os = new PrintStream(sock.getOutputStream(), true)) {
            String line;
            while ((line = is.readLine()) != null) {
                os.print(line + "\r");
                os.flush();
            }
            sock.close();
        } catch (IOException e) {
            System.out.println("IO Error on socket " + e);
            return;
        }
        System.out.println("Socket ENDED: " + sock);
    }
}
```
x??

---
#### Server Socket Accept Method
The `accept()` method in the `ServerSocket` class is not synchronized, meaning multiple threads can call it concurrently. This lack of synchronization could lead to race conditions and data corruption if accessed by many threads at once.

:p How does the `accept()` method handle concurrency issues?
??x
To mitigate concurrency issues with `accept()`, a `synchronized` keyword is used around this method call in the server's main loop. Only one thread at a time can execute within the synchronized block, ensuring that global data is not corrupted during these operations.

```java
public void runServer () {
    ServerSocket sock;
    Socket clientSocket ;
    try {
        sock = new ServerSocket(ECHOPORT);
        System.out.println("EchoServerThreaded ready for connections.");
        while (true) {
            synchronized(sock) {  // Synchronize on the server socket to control access
                clientSocket = sock.accept();
                new Handler(clientSocket).start();
            }
        }
    } catch (IOException e) {
        System.err.println("Could not accept " + e);
        System.exit(1);
    }
}
```
x??

---
#### Handling Multiple Clients with Fixed Number of Threads
Instead of creating a new thread for each client, you can pre-create a fixed number of threads. However, this approach requires careful management to avoid race conditions when accessing the `ServerSocket`'s `accept()` method.

:p How do you control access to the `accept()` method in a multi-threaded server?
??x
To control access to the `accept()` method, you can synchronize it using the `synchronized` keyword. This ensures that only one thread at a time can call `accept()`, preventing race conditions and data corruption.

```java
public void runServer () {
    ServerSocket sock;
    Socket clientSocket ;
    try {
        sock = new ServerSocket(ECHOPORT);
        System.out.println("EchoServerThreaded ready for connections.");
        while (true) {
            synchronized(sock) {  // Synchronize on the server socket to control access
                clientSocket = sock.accept();
                new Handler(clientSocket).start();
            }
        }
    } catch (IOException e) {
        System.err.println("Could not accept " + e);
        System.exit(1);
    }
}
```
x??

---
#### Threaded Object Management
In the server implementation, each `Handler` thread manages a client's communication. The `Handler` class extends `Thread`, and its `run()` method contains an infinite loop that reads from the input stream and writes to the output stream.

:p What is the role of the `Handler` class in the server?
??x
The `Handler` class represents a single-threaded object responsible for managing communication with one client. It extends `Thread`, allowing it to handle multiple clients concurrently by creating new threads as needed.

```java
class Handler extends Thread {
    Socket sock;
    Handler(Socket s) { sock = s; }
    public void run() {
        System.out.println("Socket starting: " + sock);
        try (BufferedReader is = new BufferedReader(new InputStreamReader(sock.getInputStream()));
             PrintStream os = new PrintStream(sock.getOutputStream(), true)) {
            String line;
            while ((line = is.readLine()) != null) {
                os.print(line + "\r");
                os.flush();
            }
            sock.close();
        } catch (IOException e) {
            System.out.println("IO Error on socket " + e);
            return;
        }
        System.out.println("Socket ENDED: " + sock);
    }
}
```
x??

---

#### Quick Client Connection Start Time Trade-off
Background context: The example discusses a server design that prioritizes faster client connection startup at the cost of increased server initialization time. This approach avoids repeatedly constructing new `Handler` or `Thread` objects for each request, which can improve overall performance.

:p What is the trade-off made by this server design?
??x
The server starts more quickly with clients, but it takes longer to start due to the overhead in setting up threads.
x??

---

#### Thread Pool for Client Handling
Background context: The example illustrates a thread pool mechanism where multiple threads are pre-created to handle client connections. This is similar to how Apache web servers operate by creating a pool of processes.

:p How does the `EchoServerThreaded2` class create and start its handler threads?
??x
The `EchoServerThreaded2` class creates a specified number of handler threads, starting each one with the `start()` method. The constructor initializes a server socket and then starts the handler threads using a loop.
```java
public EchoServerThreaded2(int port, int numThreads) {
    ServerSocket servSock;
    try {
        servSock = new ServerSocket(port);
    } catch (IOException e) {
        throw new RuntimeException("Could not create ServerSocket ", e);
    }
    // Create a series of threads and start them.
    for (int i = 0; i < numThreads; i++) {
        new Handler(servSock, i).start();
    }
}
```
x??

---

#### Synchronous Client Connection Handling
Background context: The `Handler` class handles client connections synchronously by waiting for a connection to be accepted and then processing the request in a loop.

:p How does the `Handler` class handle incoming client connections?
??x
The `Handler` class uses a synchronized block on the server socket to wait for an incoming connection. Once a connection is established, it prints details about the client's IP address and processes the input line by line, echoing each line back to the client.
```java
public void run() {
    while (true) {
        try {
            System.out.println(getName() + " waiting");
            Socket clientSocket;
            // Wait here for the next connection.
            synchronized (servSock) {
                clientSocket = servSock.accept();
            }
            System.out.println(
                    getName() + " starting, IP=" + 
                    clientSocket.getInetAddress());
            try (
                BufferedReader is = new BufferedReader(
                        new InputStreamReader(clientSocket.getInputStream()));
                PrintStream os = new PrintStream(clientSocket.getOutputStream(), true);
            ) {
                String line;
                while ((line = is.readLine()) != null) {
                    os.print(line + "\r");
                    os.flush();
                }
                System.out.println(getName() + " ENDED ");
                clientSocket.close();
            }
        } catch (IOException ex) {
            System.out.println(getName() + ": IO Error on socket " + ex);
            return;
        }
    }
}
```
x??

---

#### Server Thread Class Design
Background context: The `EchoServerThreaded2` class uses an inner class `Handler` to handle client connections in a thread. Each `Handler` object represents one thread responsible for processing a single connection.

:p What is the role of the `Handler` class in this server design?
??x
The `Handler` class extends `Thread` and handles individual client connections by starting when it is initialized. It processes incoming data from clients line-by-line, echoing each line back to the client.
```java
class Handler extends Thread {
    ServerSocket servSock;
    int threadNumber;

    public Handler(ServerSocket s, int i) {
        servSock = s;
        threadNumber = i;
        setName("Thread " + threadNumber);
    }

    public void run() {
        while (true) {
            try {
                System.out.println(getName() + " waiting");
                Socket clientSocket;
                // Wait here for the next connection.
                synchronized (servSock) {
                    clientSocket = servSock.accept();
                }
                System.out.println(
                        getName() + " starting, IP=" +
                        clientSocket.getInetAddress());
                try (
                    BufferedReader is = new BufferedReader(
                            new InputStreamReader(clientSocket.getInputStream()));
                    PrintStream os = new PrintStream(
                            clientSocket.getOutputStream(), true);
                ) {
                    String line;
                    while ((line = is.readLine()) != null) {
                        os.print(line + "\r");
                        os.flush();
                    }
                    System.out.println(getName() + " ENDED ");
                    clientSocket.close();
                }
            } catch (IOException ex) {
                System.out.println(getName() + ": IO Error on socket " + ex);
                return;
            }
        }
    }
}
```
x??

---

#### Welcome Message Structure
Background context: The provided Java code snippet constructs a welcome message for a web server, which is displayed to users who access the non-SSL version of the server. This involves combining HTML content with Java string concatenation and printing it over an output stream.

:p What does this Java code do?
??x
This Java code sets up a basic web server that greets users but informs them they are accessing a non-SSL version. It constructs an HTML response, prints its length, and sends the message back to the client.

Example of how the final HTML content looks:
```html
<h1>Welcome, </h1>, but...
<p>You have reached a desktop machine that does not run a real Web service.
Please pick another system.
Or view <a href="VIEW_SOURCE_URL">the WebServer0 source on github</a>.
<hr/><em>Java-based WebServer0</em><hr/>
</p>
```
x??

---

#### Java Secure Socket Extension (JSSE) Usage
Background context: The provided code snippet demonstrates how to use JSSE for secure communication over SSL/TLS. This is crucial for protecting data in transit from being intercepted or modified by unauthorized parties.

:p How can you implement SSL encryption using JSSE?
??x
To implement SSL encryption, you replace the standard `ServerSocket` with an instance created by `SSLServerSocketFactory`. The code snippet shows how to override the `getServerSocket()` method to use an SSL-enabled server socket. Additionally, it requires setting up a keystore for storing your SSL certificate.

Example of overriding `getServerSocket()`:
```java
protected ServerSocket getServerSocket(int port) throws Exception {
    SSLServerSocketFactory ssf = (SSLServerSocketFactory) SSLServerSocketFactory.getDefault();
    return ssf.createServerSocket(port);
}
```
x??

---

#### Self-Signed Certificate Setup
Background context: For demonstration purposes, a self-signed certificate can be used to secure the web server. This certificate is typically created for testing and development environments.

:p How do you set up a self-signed SSL certificate?
??x
To create a self-signed SSL certificate, follow these steps:
1. Generate a private key.
2. Create a CSR (Certificate Signing Request) using the private key.
3. Self-sign the CSR to generate a certificate.
4. Store the certificate in a keystore.

Example command for generating a self-signed certificate and storing it in a keystore:
```bash
keytool -genkeypair -alias mycert -keystore /home/ian/.keystore -storepass secrat -validity 365
```
x??

---

#### Running the JSSE Web Server
Background context: The provided code shows how to start an SSL-enabled web server using JSSE. This involves setting system properties for the keystore and running the `JSSEWebServer0` class.

:p How do you run the JSSE web server?
??x
To run the JSSE web server, you need to set the `-D` system property to specify the path to your keystore and its password. For example:
```bash
java -Djavax.net.ssl.keyStore=/home/ian/.keystore -Djavax.net.ssl.keyStorePassword=secrat JSSEWebServer0
```
This command tells Java where to find the SSL certificate and sets up the necessary environment for secure communication.

x??

---

#### Client Browser Behavior with Self-Signed Certificates
Background context: When accessing an SSL server with a self-signed certificate, modern browsers typically display warnings or caution messages due to the inherent risks associated with self-signed certificates. However, users can still proceed by accepting the certificate.

:p How does a client browser handle a self-signed certificate?
??x
When a client browser encounters a self-signed certificate, it often displays a warning message indicating that the security of the connection is not verified. Despite this caution, if the user confirms or accepts the risk, they are allowed to proceed with the connection.

Example of what might be displayed in a modern browser:
- A red warning icon and caution message.
- The option to "Proceed anyway" after accepting the risk.

x??

---

