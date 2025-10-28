# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 54)

**Starting Chapter:** 12.1 HTTPREST Web Client. Solution

---

#### Connecting to a Server Using Socket in C

Background context: This concept covers how to establish a connection with a server using sockets in C. It involves creating a socket, specifying the server address and port, and then attempting to connect to the server.

:p How does one create a socket connection to a server in C?
??x
In C, you can use the `connect` function from the `<sys/socket.h>` library to establish a connection with a server. Here’s an example of how this is done:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> // For close()
#include <arpa/inet.h> // For sockaddr_in and inet_pton
#include <sys/socket.h>

int main() {
    int sock; // The file descriptor for the socket
    struct sockaddr_in server; // Structure to hold address details

    // Create a socket
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("socket creation failed");
        exit(1);
    }

    memset(&server, '\0', sizeof(server)); // Clear the structure

    server.sin_family = AF_INET; // IPv4 family
    server.sin_port = htons(80);  // Server port (HTTP)
    
    // Convert the IP address to binary form. Use inet_pton() for this.
    if (inet_pton(AF_INET, "127.0.0.1", &server.sin_addr) <= 0) {
        perror("Invalid address/ Address not supported");
        exit(2);
    }

    // Connect to the server
    if (connect(sock, (struct sockaddr *)&server, sizeof(server)) < 0) {
        perror("connecting to server");
        close(sock); // Close the socket if connection failed
        exit(4);
    }

    // Proceed with reading and writing on the socket.
    
    close(sock); // Close the socket after use

    return 0;
}
```
x??

---

#### Java One-Line Connection Example for a Server

Background context: In Java, establishing a simple TCP connection to a server can often be done in one line of code using the `Socket` class.

:p How can you establish a connection with a server using Java?
??x
In Java, creating and connecting to a server is relatively straightforward. Here's an example where we connect to a specific host and port:

```java
import java.net.Socket;

public class ConnectToServer {
    public static void main(String[] args) throws Exception {
        // Create a socket connection in one line of code.
        Socket sock = new Socket("localhost", 80);

        // Proceed with further operations using the 'sock' object.

        sock.close(); // Close the socket after use
    }
}
```
x??

---

#### TFTP Client Example

Background context: The Trivial File Transfer Protocol (TFTP) is a simple protocol used for booting diskless workstations. This example covers creating a basic client that can interact with a TFTP server.

:p What is an example of implementing the TFTP protocol in C?
??x
Implementing a basic TFTP client involves setting up connections, sending and receiving packets to download files from a server. Here’s how you might start:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> // For close()
#include <arpa/inet.h> // For sockaddr_in and inet_pton
#include <sys/socket.h>

int main() {
    int sock;
    struct sockaddr_in server;

    // Create a socket
    if ((sock = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(1);
    }

    memset(&server, '\0', sizeof(server));

    server.sin_family = AF_INET;
    server.sin_port = htons(69); // TFTP port
    if (inet_pton(AF_INET, "127.0.0.1", &server.sin_addr) <= 0) {
        perror("Invalid address/ Address not supported");
        exit(2);
    }

    // Send a request to the server
    const char *msg = "GET /file.txt";
    sendto(sock, msg, strlen(msg), 0, (struct sockaddr *)&server, sizeof(server));

    // Receive data from the server and process it
    // This is a simplified example

    close(sock); // Close the socket after use

    return 0;
}
```
x??

---

#### Web Services Client in Java for HTTP/REST

Background context: Web services clients can interact with web servers using HTTP. The provided text suggests familiarity with REST-based services, which involve sending HTTP requests and receiving responses.

:p How does one read from a URL to connect to a RESTful web service or download a resource over HTTP/HTTPS in Java?
??x
In Java, you can use the `HttpURLConnection` class to interact with URLs. Here’s how you might set up an HTTP GET request:

```java
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class HttpRestClient {
    public static void main(String[] args) throws Exception {
        URL url = new URL("http://example.com/api/resource");
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        
        // Set the request method to GET
        connection.setRequestMethod("GET");

        // Response code 200 is HTTP_OK
        if (connection.getResponseCode() == HttpURLConnection.HTTP_OK) {
            BufferedReader in = new BufferedReader(new InputStreamReader(connection.getInputStream()));
            String inputLine;
            StringBuffer response = new StringBuffer();

            while ((inputLine = in.readLine()) != null) {
                response.append(inputLine);
            }
            in.close();
            
            // Process the response
            System.out.println(response.toString());
        } else {
            System.err.println("Failed : HTTP error code : " + connection.getResponseCode());
        }

        connection.disconnect(); // Close the connection
    }
}
```
x??

---

#### Introduction to Java 11 HttpClient
Background context: Prior to Java 11, developers had to rely on either `URLConnection` or third-party libraries like Apache HTTP Client for making HTTP requests. With Java 11, a more flexible and easy-to-use API is available in the standard library.
:p What does Java 11 offer that was not available before?
??x
Java 11 introduces the `HttpClient` class as part of its standard library, providing an easier and more flexible way to make HTTP requests compared to `URLConnection`. It also supports HTTP/2.0, which is not supported by Apache HttpClient as of early 2020.
x??

---

#### Creating a HttpClient Object
Background context: The `HttpClient` object is created using the Builder pattern. This allows for configuration before the client is built and used for sending requests.
:p How do you create an `HttpClient` object in Java 11?
??x
You can create an `HttpClient` object by building it with desired configurations:
```java
HttpClient client = HttpClient.newBuilder()
                               .followRedirects(Redirect.NORMAL)
                               .version(HttpClient.Version.HTTP_1_1)
                               .build();
```
Here, the `.newBuilder()` method is used to start building the `HttpClient`. The configuration methods like `.followRedirects()` and `.version()` are called on this builder object. Finally, the client is built using the `.build()` method.
x??

---

#### Building an HttpRequest Object
Background context: An `HttpRequest` object is needed to send a request to a specific URL with certain headers. This object can be configured before it is sent to the `HttpClient`.
:p How do you build and configure an `HttpRequest` object?
??x
You can build and configure an `HttpRequest` object as follows:
```java
HttpRequest req = HttpRequest.newBuilder(URI.create(urlString + URLEncoder.encode(keyword)))
                            .header("User-Agent", "Dept of Silly Walks")
                            .GET()
                            .build();
```
The `.newBuilder()` method is used to start building the request. The `URI` for the request URL is created using `URI.create()`. Headers can be added using the `.header()` method, and the HTTP method (in this case, GET) is specified with `.GET()`. Finally, the request object is built with `.build()`.
x??

---

#### Sending a Request Synchronously
Background context: After building an `HttpRequest` object, you can send it to the `HttpClient` for synchronous execution. This means the program waits for the response before proceeding.
:p How do you send a request synchronously using Java 11 HttpClient?
??x
To send a request synchronously, you use the following method:
```java
HttpResponse<String> resp = client.send(req, BodyHandlers.ofString());
```
The `client.send()` method is called with two parameters: the `HttpRequest` object and the `BodyHandlers.ofString()` to handle the response body as a string. The result is stored in an `HttpResponse<String>` object.
x??

---

#### Sending a Request Asynchronously
Background context: For asynchronous requests, you do not block the execution of your program while waiting for the response. Instead, you use the `.sendAsync()` method and process the response when it arrives.
:p How do you send a request asynchronously using Java 11 HttpClient?
??x
To send a request asynchronously, you can use:
```java
client.sendAsync(req, BodyHandlers.ofString())
       .thenApply(HttpResponse::body)
       .thenAccept(System.out::println)
       .join();
```
The `sendAsync()` method is used to send the request in an asynchronous manner. The response handling chain is created using `.thenApply()` and `.thenAccept()`. Finally, `.join()` is called to wait for the completion of the asynchronous operation.
x??

---

#### Using URLConnection as Alternative
Background context: If you prefer not to use `HttpClient`, you can still achieve your goals using the older `URLConnection` class. This class provides methods to open and read from a URL.
:p How do you make an HTTP request using Java’s `URLConnection`?
??x
To make an HTTP request using `URLConnection`, you can follow these steps:
```java
URLConnection conn = new URL(HttpClientDemo.urlString + HttpClientDemo.keyword)
                        .openConnection();
try (BufferedReader is = new BufferedReader(new InputStreamReader(conn.getInputStream()))) {
    String line;
    while ((line = is.readLine()) != null) {
        System.out.println(line);
    }
}
```
The `URL` class is used to create a URL object, and its `openConnection()` method returns a `URLConnection`. The input stream from the connection is read using a `BufferedReader`, and each line is printed out.
x??

---

---
#### RESTful Java Services
Background context: This card covers information on implementing server-side components for REST services, as discussed in Bill Burke’s "RESTful Java with JAX-RS 2.0, 2nd Edition" (O’Reilly).
:p What book provides guidance on implementing server-side components for REST services?
??x
The book "RESTful Java with JAX-RS 2.0, 2nd Edition" by Bill Burke.
x??
---

---
#### Contacting a Socket Server Using TCP/IP
Background context: This card explains how to create a socket connection in Java using the `java.net.Socket` class for establishing a client-server communication over TCP/IP.
:p How do you create a socket connection in Java?
??x
To create a socket connection in Java, you use the `java.net.Socket` constructor by passing the hostname and port number. Here is an example of how it's done:

```java
import java.net.Socket;

public class ConnectSimple {
    public static void main(String[] argv) throws Exception {
        try (Socket sock = new Socket("localhost", 8080)) {
            System.out.println("*** Connected OK ***");
            // Do some I/O here...
        }
    }
}
```

This code snippet demonstrates the use of `try-with-resources` to ensure that the socket is closed automatically when done.
x??
---

