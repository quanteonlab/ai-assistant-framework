# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 55)

**Starting Chapter:** 12.3 Finding and Reporting Network Addresses. Problem. Solution. 12.4 Handling Network Errors

---

#### Finding and Reporting Network Addresses
Background context: This section explains how to find and report network addresses using Java's `InetAddress` class. It covers methods like `getByName`, `getHostAddress`, and `getHostName`. Additionally, it mentions that these methods do not contact the remote host and thus do not throw exceptions related to network connections.
:p What method in InetAddress is used to get the numeric IP address corresponding to an InetAddress object?
??x
The `getHostAddress()` method returns the numeric IP address as a string for a given InetAddress object. This is useful when you need to obtain the IP address from an already obtained host name or address.

```java
String ipAddress = InetAddress.getByName("darwinsys.com").getHostAddress();
```
x??

---

#### Getting InetAddress by Address
Background context: The text explains how to get an `InetAddress` object using a network address string. This can be useful when you know the IP address of a host and want to find its hostname or other related information.
:p How do you obtain an InetAddress for a known IP address?
??x
You can use the `getByName(String hostnameOrAddress)` method, passing in the IP address as a string.

```java
String ipAddress = "8.8.8.8"; // Google's public DNS server
InetAddress inetAddress = InetAddress.getByName(ipAddress);
```
x??

---

#### Getting InetAddress by Host Name
Background context: This section describes how to get an `InetAddress` object using the hostname of a host. It explains that this method can be used to find out the IP address given a host name.
:p How do you obtain an InetAddress for a known host name?
??x
You can use the `getByName(String hostnameOrAddress)` method, passing in the hostname as a string.

```java
String hostName = "darwinsys.com";
InetAddress inetAddress = InetAddress.getByName(hostName);
```
x??

---

#### Getting Localhost Address
Background context: The text provides information on how to get the `InetAddress` for the local machine using the `getLocalHost()` method. This is particularly useful when you want to connect to a server running on the same machine.
:p How do you obtain the InetAddress representing localhost?
??x
You can use the `getLocalHost()` static method of the `InetAddress` class.

```java
final InetAddress localHost = InetAddress.getLocalHost();
```
x??

---

#### Getting Remote InetAddress from Socket
Background context: This section explains how to get an `InetAddress` object for a remote host by using a `Socket`. It is useful in scenarios where you have established a connection and want to know the address of the server.
:p How do you obtain the InetAddress of a remote host using a Socket?
??x
You can use the `getInetAddress()` method on an existing `Socket` object.

```java
String someServerName = "google.com";
try (Socket theSocket = new Socket(someServerName, 80)) {
    InetAddress remote = theSocket.getInetAddress();
}
```
x??

---

#### Getting All Addresses for a Host
Background context: The text describes how to get all `InetAddress` objects associated with a host name using the `getAllByName(String hostname)` method. This is useful when a server might be on multiple networks.
:p How do you obtain all InetAddress objects for a given host name?
??x
You can use the `getAllByName(String hostname)` static method of the `InetAddress` class.

```java
String[] addresses = InetAddress.getAllByName("google.com");
```
x??

---

#### Using Inet6Address for IPv6
Background context: The text mentions that if you are using IPv6, you should use `Inet6Address` instead of `InetAddress`. This is necessary to handle the different format and representation of IPv6 addresses.
:p How do you get an Inet6Address object?
??x
You can create an `Inet6Address` object by calling the appropriate constructor or methods from the `InetAddress` class, specifying that you are using IPv6.

```java
// Example: Creating an Inet6Address instance
String ip6Address = "2001:db8::1";
try {
    Inet6Address inet6Addr = (Inet6Address) InetAddress.getByName(ip6Address);
} catch (UnknownHostException e) {
    // handle exception
}
```
x??

---

---
#### Service Number Management for Programs
Background context: The text discusses the convenience of changing service numbers for programs on a machine or network without altering program code, especially when services are new or installed non-routinely. This capability is suggested as an improvement for Java in future releases.

:p How can Java improve to handle new or nonroutine services?
??x
Java could introduce mechanisms that allow developers to change the service definitions globally for all programs on a machine or network without modifying the program code. This would be particularly useful when dealing with new or non-routine services.
x??

---
#### Handling Network Errors in Java Programs
Background context: The text outlines how to handle network errors more precisely by catching different types of exceptions beyond `IOException`. It emphasizes using specific exception classes like `SocketException`, `ConnectException`, and `NoRouteToHostException`.

:p What is the purpose of handling network errors more precisely?
??x
The purpose is to provide more detailed error reporting, allowing developers to understand exactly what went wrong when a network operation fails. This can help in diagnosing and fixing issues more effectively.
x??

---
#### Example Code for Handling Network Errors: ConnectFriendly.java
Background context: The code snippet demonstrates enhanced handling of connection failures by catching specific types of `SocketException` subclasses.

:p How does the example code handle different types of network errors?
??x
The example code handles different types of network errors by catching and printing more detailed error messages. It uses exceptions like `UnknownHostException`, `NoRouteToHostException`, and `ConnectException` to provide precise feedback on what went wrong during a connection attempt.

```java
public class ConnectFriendly {
    public static void main(String[] argv) {
        String server_name = argv.length == 1 ? argv[0] : "localhost";
        int tcp_port = 80;
        try (Socket sock = new Socket(server_name, tcp_port)) {
            System.out.println(" *** Connected to " + server_name + " ***");
            /* Do some I/O here... */
        } catch (UnknownHostException e) {
            System.err.println(server_name + " Unknown host");
            return;
        } catch (NoRouteToHostException e) {
            System.err.println(server_name + " Unreachable");
            return;
        } catch (ConnectException e) {
            System.err.println(server_name + " connect refused");
            return;
        } catch (IOException e) {
            System.err.println(server_name + ' ' + e.getMessage());
            return;
        }
    }
}
```
x??

---

---
#### BufferedReader and PrintWriter Construction
Background context: When working with textual data over a network connection, you often need to read from or write to a socket. The `Socket` class provides methods for getting an input or output stream but does not directly provide reader or writer objects. You can convert these streams into readers and writers using conversion classes.

:p How do you construct a `BufferedReader` and `PrintWriter` from a `Socket`?
??x
To construct a `BufferedReader` and `PrintWriter`, you first get the input and output streams from the socket, then wrap those streams with appropriate reader and writer objects. For example:

```java
BufferedReader is = new BufferedReader(new InputStreamReader(sock.getInputStream()));
PrintWriter os = new PrintWriter(sock.getOutputStream(), true);
```

The `BufferedReader` is used for reading text line by line, while the `PrintWriter` handles writing text to the socket. The second argument in `PrintWriter` being set to `true` ensures that each write operation flushes the buffer automatically.

x??
---

#### Reading from Daytime Service
Background context: The example provided demonstrates how to read a single line of text from a "Daytime" service, which is available on some networked systems. This service sends the current date and time when connected to via TCP/IP.

:p How does the `DaytimeText` class read the current date and time from a server?
??x
The `DaytimeText` class connects to a Daytime server and reads one line of text, which contains the current date and time. Here's how it works:

```java
public static void main(String[] argv) {
    String server_name = argv.length == 1 ? argv[0] : "localhost";
    try (Socket sock = new Socket(server_name, TIME_PORT);
         BufferedReader is = new BufferedReader(new InputStreamReader(sock.getInputStream()))) {
        String remoteTime = is.readLine();
        System.out.println("Time on " + server_name + " is " + remoteTime);
    } catch (IOException e) {
        System.err.println(e);
    }
}
```

The `Socket` object is created to connect to the specified host and port. The input stream of the socket is wrapped with a `BufferedReader`, which reads one line of text using `readLine()`. This line contains the date and time from the server.

x??
---

#### Writing to Echo Server
Background context: The "Echo" server sends back whatever data it receives. It can be used for testing network connections or client-server interactions.

:p How does the `EchoClientOneLine` class communicate with an Echo server?
??x
The `EchoClientOneLine` class establishes a connection with an Echo server, sends a message, and prints the echoed response. Here's how it works:

```java
public static void main(String[] argv) {
    if (argv.length == 0)
        new EchoClientOneLine().converse("localhost");
    else
        new EchoClientOneLine().converse(argv[0]);
}

protected void converse(String hostName) {
    try (Socket sock = new Socket(hostName, 7)) { // echo server.
        BufferedReader is = new BufferedReader(new InputStreamReader(sock.getInputStream()));
        PrintWriter os = new PrintWriter(sock.getOutputStream(), true);
        
        os.println(mesg); // Send the message
        String response = is.readLine(); // Read the echoed message
        System.out.println("Received: " + response);
    } catch (IOException e) {
        System.err.println(e);
    }
}
```

The `Socket` object connects to the specified host and port. The output stream of the socket is wrapped with a `PrintWriter`, which sends a message using `println()`. The input stream is then used to read the echoed response from the server.

x??
---

