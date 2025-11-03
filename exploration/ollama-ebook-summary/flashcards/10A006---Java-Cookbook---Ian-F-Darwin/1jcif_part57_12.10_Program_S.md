# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 57)

**Starting Chapter:** 12.10 Program Sockets-Based Chat Client

---

---
#### Setting Up TFTP Server on Unix System
To set up a TFTP server for testing purposes, you need to configure the `/etc/inetd.conf` file and restart or reload the `inetd` service. This example demonstrates setting up an TFTP server using the provided configuration.

:p How do you configure the TFTP server on a Unix system?
??x
You would add or modify the following line in your `/etc/inetd.conf` file:
```
tftp    dgram   udp      wait    root    /usr/libexec/tftpd tftpd -s /tftpboot
```
Then, you need to restart or reload the `inetd` service. On Linux systems, this can be done using commands like `service inetd restart` or `systemctl restart inetd`. This configuration uses the `tftpd` daemon with a root directory `/tftpboot`.

To test if everything is set up correctly, you could put some files in the `/tftpboot` directory and use the `RemCat` client to retrieve them. For example:
```java
$ java network.RemCat localhost foo | diff - /tftpboot/foo
```
This command retrieves a file named `foo` from the TFTP server running on `localhost`, and then compares it with the original file in `/tftpboot` using `diff`. If there are no differences, everything is set up correctly.

x??
---
#### Using Java Chat Client - GUI Setup
The chat client class (`ChatClient`) sets up a graphical user interface (GUI) for chatting. It includes components like text areas for displaying messages and text fields for input. The login button triggers the `login()` method, which establishes a connection with the server.

:p How does the `ChatClient` class set up its GUI?
??x
The `ChatClient` class extends `JFrame` and sets up various UI elements such as text areas, text fields, and buttons. Here is an example of how it initializes these components:
```java
public ChatClient() {
    cp = this;
    cp.setTitle(TITLE);
    cp.setLayout(new BorderLayout());
    port = PORTNUM;

    // The GUI
    ta = new JTextArea(14, 80);
    ta.setEditable(false);        // readonly
    ta.setFont(new Font("Monospaced", Font.PLAIN, 11));
    cp.add(BorderLayout.NORTH, ta);

    JPanel p = new JPanel();
    
    // The login button
    p.add(loginButton = new JButton("Login"));
    loginButton.setEnabled(true);
    loginButton.addActionListener(e -> {
        login();
        loginButton.setEnabled(false);
        logoutButton.setEnabled(true);
        tf.requestFocus();  // set keyboard focus in the right place.
    });

    // The logout button
    p.add(logoutButton = new JButton("Logout "));
    logoutButton.setEnabled(false);
    logoutButton.addActionListener(e -> {
        logout();
        loginButton.setEnabled(true);
        logoutButton.setEnabled(false);
        loginButton.requestFocus();
    });

    p.add(new JLabel("Message here:"));
    tf = new JTextField(40);
    tf.addActionListener(e -> {
        if (loggedIn) {
            pw.println(ChatProtocol.CMD_BCAST + tf.getText());
            tf.setText("");
        }
    });
    
    p.add(tf);
    cp.add(BorderLayout.SOUTH, p);
    cp.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    cp.pack();
}
```
This code snippet initializes the main frame (`cp`), sets its title and layout, adds a `JTextArea` for displaying messages, and creates a panel with login/logout buttons and text fields. The components are arranged in a `BorderLayout`, ensuring that they are displayed appropriately.

x??
---
#### Error Handling in Chat Client
The chat client includes basic error handling to manage potential issues such as failed connections or disconnections from the server. It checks if the connection is established before attempting to send messages and handles exceptions gracefully.

:p How does the `ChatClient` handle errors during login and logout processes?
??x
During the login process, the `login()` method attempts to connect to the chat server using a `Socket`. If an exception occurs (e.g., due to network issues), it catches the exception and displays an error message. Here’s how error handling is implemented:
```java
public void login() {
    showStatus("In login.");
    if (loggedIn) return;
    try {
        sock = new Socket(serverHost, port);
        is = new BufferedReader(new InputStreamReader(sock.getInputStream()));
        pw = new PrintWriter(sock.getOutputStream(), true);
        showStatus("Got socket");
        // FAKE LOGIN FOR NOW - no password needed
        pw.println(ChatProtocol.CMD_LOGIN + userName);
        loggedIn = true;
    } catch (IOException e) {
        warn("Can't get socket to " + serverHost + "/" + port + ": " + e);
        cp.add(new JLabel("Can't get socket: " + e));
        return;
    }
    // Construct and start the reader: from server to textarea.
    // Make a Thread to avoid lockups.
    Runnable readerThread = new Runnable() {
        public void run() {
            String line;
            try {
                while (loggedIn && ((line = is.readLine()) != null))
                    ta.append(line + " ");
            } catch (IOException e) {
                showStatus("Lost another client. " + e);
                return;
            }
        }
    };
    threadPool.execute(readerThread);
}
```
In case of failure, the `login()` method catches `IOException` and sets up a warning label on the frame with an appropriate message.

For logout, the `logout()` method checks if the user is logged in before proceeding. If so, it closes the socket connection and resets the login status.
```java
public void logout() {
    if (!loggedIn) return;
    loggedIn = false;
    try {
        if (sock != null)
            sock.close();
    } catch (IOException ign) {
        // so what?
    }
}
```
If there is an issue closing the socket, a simple warning message is logged without crashing the application.

x??
---

#### ChatClient Main Method
Background context: The provided Java code snippet includes a main method for running the `ChatClient` application. This method initializes and displays the chat client window, making it usable as an independent Java Application.

:p What is the purpose of the main method in this `ChatClient` class?
??x
The main method initializes the `ChatClient` object, packs the layout, and makes it visible to the user. It serves as a starting point for running the application.
```java
public static void main(String[] args) {
    ChatClient room101 = new ChatClient();
    room101.pack();  // Pack the components in the container.
    room101.setVisible(true);  // Display the window to the user.
}
```
x??

---

#### ShowStatus Method
Background context: The `showStatus` method is used to display a message on the console. This can be useful for debugging or providing feedback during runtime.

:p What does the `showStatus` method do?
??x
The `showStatus` method prints a given message to the standard output (console). It's often used for logging messages or displaying information.
```java
public void showStatus(String message) {
    System.out.println(message);
}
```
x??

---

#### Warn Method Using JOptionPane
Background context: The `warn` method uses `JOptionPane.showMessageDialog` to display a warning dialog. This is useful when you need to alert the user with more than just text.

:p What does the `warn` method do?
??x
The `warn` method displays a message dialog box using `JOptionPane`. It's designed to show a warning or informational message to the user.
```java
private void warn(String message) {
    JOptionPane.showMessageDialog(this, message);
}
```
x??

---

#### LinkChecker Overview
Background context: The text discusses the implementation of a simple HTTP link checker (`KwikLinkChecker`). This tool checks if an HTTP link is accessible and returns status information.

:p What is `KwikLinkChecker` used for?
??x
`KwikLinkChecker` is a simple HTTP link checker that validates whether a given URL is accessible. It sends HTTP requests to the URLs, collects responses, and determines their statuses.
```java
public LinkStatus check(String urlString) {
    // Code to send request and process response
}
```
x??

---

#### Check Method Implementation
Background context: The `check` method within `KwikLinkChecker` checks an HTTP link. It sends a GET request and handles various exceptions, returning the status of the URL.

:p What does the `check` method do?
??x
The `check` method sends an HTTP GET request to the provided URL, processes the response, and returns a `LinkStatus` object indicating whether the URL is accessible and its status code.
```java
public LinkStatus check(String urlString) {
    try {
        // Code to send request and process response
    } catch (Exception e) {
        return new LinkStatus(false, "Error: " + e);
    }
}
```
x??

---

#### Handling HTTP Response Status Codes
Background context: The `check` method handles different HTTP status codes and returns appropriate statuses using a `LinkStatus` object. This helps in understanding the current state of the URL.

:p What are some status codes handled by the `check` method?
??x
The `check` method handles several HTTP status codes:
- 200: OK (Successful request)
- 403: Forbidden
- 404: Not Found
It returns a `LinkStatus` object indicating the success or failure of the URL based on these statuses.
```java
case 200:
    return new LinkStatus(true, urlString);
case 403:
    return new LinkStatus(false, "403: " + urlString);
case 404:
    return new LinkStatus(false, "404: " + urlString);
```
x??

---

#### Exception Handling in `check` Method
Background context: The `check` method includes a series of exception handling blocks to manage different types of errors that might occur during the URL request.

:p What exceptions are handled by the `check` method?
??x
The `check` method handles various exceptions such as `IllegalArgumentException`, `MalformedURLException`, `UnknownHostException`, `FileNotFoundException`, `ConnectException`, and general `IOException`. Each exception is caught and returns an appropriate `LinkStatus`.
```java
catch (IllegalArgumentException | MalformedURLException e) {
    return new LinkStatus(false, "Malformed URL: " + urlString);
}
catch (UnknownHostException e) {
    return new LinkStatus(false, "Host invalid/dead: " + urlString);
}
catch (FileNotFoundException e) {
    return new LinkStatus(false, "NOT FOUND (404) " + urlString);
}
```
x??

---

---
#### Introduction to Sockets
Sockets are fundamental components for implementing networking protocols. They allow communication between a client and server across different languages, such as C, C++, Perl, Python, or Java. Various high-level services like JDBC, RMI, CORBA, EJB, RPC, and NFS all rely on socket connections.
:p What is the role of sockets in networked applications?
??x
Sockets serve as the basic building blocks for networking protocols, enabling communication between clients and servers regardless of programming language. They are essential for implementing services like JDBC, RMI, CORBA, EJB, RPC, and NFS.
x??

---
#### ServerSocket Behavior
ServerSocket is used to establish a listening socket on the server side that waits for incoming client connections. Understanding how ServerSocket behaves is crucial even if you use higher-level services like RMI or JDBC in your applications.
:p What does a ServerSocket do?
??x
A ServerSocket sets up an endpoint where clients can connect, allowing the server to receive and process requests from multiple clients simultaneously. It listens for incoming connections on a specified port and handles each connection as a separate thread or process.
x??

---
#### Writing Data Over Sockets
There are various methods to write data over sockets in Java, including using streams like `DataOutputStream`, `PrintStream`, or custom byte array writing. These methods allow sending both primitive data types and complex objects.
:p How can you send data over a socket in Java?
??x
You can send data over a socket by using output streams such as `DataOutputStream` or `PrintStream`. For example, to write a string:

```java
import java.io.*;

public class DataSender {
    public static void sendData(String data) throws IOException {
        Socket socket = new Socket("localhost", 12345);
        OutputStream os = socket.getOutputStream();
        PrintWriter out = new PrintWriter(os, true);
        out.println(data); // Send the string to the server
        socket.close(); // Close the connection
    }
}
```
x??

---
#### Complete Implementation of a Chat Server
A chat server is a comprehensive example of using sockets for real-world applications. It involves setting up a `ServerSocket` to listen for incoming connections, handling multiple clients in separate threads, and managing client communication.
:p What does a complete implementation of a chat server involve?
??x
A complete chat server involves:
1. Setting up a `ServerSocket` on the server side to wait for clients.
2. Handling multiple client connections using threads or a thread pool.
3. Managing communication between clients through shared channels (like queues).
4. Ensuring proper message handling and synchronization.

Here is a simplified version of a chat server:

```java
import java.io.*;
import java.net.*;
import java.util.*;

public class ChatServer {
    public static void main(String[] args) throws IOException {
        ServerSocket serverSocket = new ServerSocket(12345);
        System.out.println("Chat server started on port 12345");

        while (true) {
            Socket clientSocket = serverSocket.accept();
            Thread clientHandler = new ClientHandler(clientSocket);
            clientHandler.start(); // Handle each client in a separate thread
        }
    }

    static class ClientHandler extends Thread {
        private final Socket socket;

        public ClientHandler(Socket socket) {
            this.socket = socket;
        }

        @Override
        public void run() {
            try (BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                 PrintWriter out = new PrintWriter(socket.getOutputStream(), true)) {

                String inputLine, outputLine;
                while ((inputLine = in.readLine()) != null) {
                    System.out.println("Received: " + inputLine);
                    // Process the message and send back a response
                    if ("exit".equals(inputLine.toLowerCase())) break;
                    out.println("Echo: " + inputLine);
                }
            } catch (IOException e) {
                System.err.println(e.getMessage());
            }
        }
    }
}
```
x??

---

