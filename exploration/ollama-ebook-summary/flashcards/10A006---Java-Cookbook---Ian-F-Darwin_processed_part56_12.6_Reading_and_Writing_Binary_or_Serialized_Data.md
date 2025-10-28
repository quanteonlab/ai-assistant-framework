# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 56)

**Starting Chapter:** 12.6 Reading and Writing Binary or Serialized Data. Problem. Solution. Discussion

---

#### Reading and Writing Text Data
Background context: In Java, when you need to send or receive text data over a network connection using sockets, you can use `PrintWriter` for sending data and `BufferedReader` (or `InputStreamReader`) for receiving it. The provided code snippet demonstrates the process of writing a message to a socket and reading the response.

:p What is the process of sending a message through a socket using PrintWriter?
??x
To send a message, you first obtain an output stream from the socket with `sock.getOutputStream()`. Then, wrap this stream in a `PrintWriter` object. You can then use the `print` method to append the message and a CRLF (carriage return line feed), which is necessary because `println` only appends `\r` on some platforms.

```java
// Get output stream from socket
Socket sock = new Socket("localhost", 12345);
PrintWriter os = new PrintWriter(sock.getOutputStream(), true);

// Send message with CRLF and flush to ensure it's sent immediately
os.print("Hello Server" + "\r\n");
os.flush();
```
x??

---

#### Reading and Writing Binary Data
Background context: When you need to send binary data or serialized Java objects over a network, using `DataInputStream`/`DataOutputStream`, `ObjectInputStream`/`ObjectOutputStream` is more appropriate. The example provided demonstrates reading the current time as an unsigned 32-bit integer from a server.

:p How do you read and write binary data using DataInputStream and DataOutputStream?
??x
To handle binary data, create instances of `DataInputStream` and `DataOutputStream` from the socket's input and output streams. If performance is critical for large amounts of data, wrap these with buffered streams like `BufferedInputStream` and `BufferedOutputStream`.

```java
// Example of using DataInputStream and DataOutputStream
Socket sock = new Socket("localhost", 37);
DataInputStream is = new DataInputStream(new BufferedInputStream(sock.getInputStream()));
DataOutputStream os = new DataOutputStream(new BufferedOutputStream(sock.getOutputStream()));
```
x??

---

#### Reading Binary Time from a Server
Background context: The server in this example provides the current time as an unsigned integer representing the number of seconds since 1900. This value needs to be adjusted by subtracting the difference between 1970 and 1900, multiplied by the number of seconds per day.

:p How do you read an unsigned integer from a `DataInputStream` in Java?
??x
Java does not have an unsigned integer type. To simulate this, read four bytes (unsigned), then combine them to form a single long value using bit shifting and bitwise OR operations.

```java
long remoteTime = ((is.readUnsignedByte() << 24) |
                   (is.readUnsignedByte() << 16) |
                   (is.readUnsignedByte() << 8) |
                   is.readUnsignedByte());
```
x??

---

#### Calculating the Time Difference Between Bases
Background context: The difference between 1970 and 1900 needs to be accounted for when converting time from a server providing data in a different base year.

:p How do you calculate the number of days between two years, considering leap years?
??x
To calculate the number of days between two years, factor in the number of leap years within that period. For instance, between 1900 and 1970, there are specific leap years (e.g., 1904, 1908, ..., 1968) which add extra days.

```java
long BASE_DAYS = ((1970 - 1900) * 365 + 
                  (1970 - 1900 - 1) / 4); // Leap year adjustment

// Convert to seconds and account for leap years
long BASE_DIFF = BASE_DAYS * 24 * 60 * 60;
```
x??

---

#### Object Serialization in Java
Background context: Object serialization is the process of converting objects into a format that can be transmitted over a network. This involves using `ObjectOutputStream` to write objects and `ObjectInputStream` to read them.

:p How do you use ObjectOutputStream and ObjectInputStream for object serialization?
??x
For object serialization, construct an `ObjectOutputStream` or `ObjectInputStream` from the socket’s input or output stream. These classes allow you to serialize Java objects directly over a network connection.

```java
// Example of using ObjectOutputStream and ObjectInputStream
Socket sock = new Socket("localhost", 37);
ObjectOutputStream os = new ObjectOutputStream(sock.getOutputStream());
ObjectInputStream is = new ObjectInputStream(sock.getInputStream());

// Write an object
os.writeObject(new SomeJavaObject());

// Read an object
SomeJavaObject obj = (SomeJavaObject)is.readObject();
```
x??

---

---
#### UDP vs TCP Connection
TCP and UDP are two different methods of transmitting data over a network, each with its own characteristics. The main difference lies in how they handle data transmission and error handling.

:p What is the primary difference between TCP and UDP in terms of data transmission?
??x
In TCP, the data is transmitted as a stream of bytes that can be fragmented into smaller units for transmission. However, on the receiving end, it is reassembled back into the original stream of bytes. This process ensures reliability but comes with more overhead.

In contrast, UDP transmits packets (chunks) of data independently and without reassembly. Each packet carries enough information to be understood separately from other packets, allowing for faster transmission but potentially leading to data loss or misordering.
??x
The answer with detailed explanations:
- TCP provides a reliable, ordered stream of bytes by ensuring that all sent data is received correctly and in the correct order.
- UDP sends packets independently, which can lead to better performance but at the cost of reliability since there's no guarantee that all packets will be delivered or in the correct order.

```java
// Example of TCP socket usage
public class TcpClient {
    public static void main(String[] args) throws IOException {
        Socket sock = new Socket("host", 12345);
        OutputStream out = sock.getOutputStream();
        InputStream in = sock.getInputStream();

        // Sending and receiving data
        out.write("Hello, Server!".getBytes());
        String response = new BufferedReader(new InputStreamReader(in)).readLine();
    }
}
```
```java
// Example of UDP socket usage
public class UdpClient {
    public static void main(String[] args) throws IOException {
        DatagramSocket sock = new DatagramSocket();

        // Creating and sending a packet
        byte[] buffer = "Hello, Server!".getBytes();
        InetAddress host = InetAddress.getByName("host");
        DatagramPacket packet = new DatagramPacket(buffer, buffer.length, host, 12345);
        sock.send(packet);

        // Receiving a packet
        byte[] receiveBuffer = new byte[1024];
        DatagramPacket receivePacket = new DatagramPacket(receiveBuffer, receiveBuffer.length);
        sock.receive(receivePacket);
        String response = new String(receivePacket.getData(), 0, receivePacket.getLength());
    }
}
```
x??
---

#### UDP Packet Structure
Datagram packets in UDP contain the actual data to be transmitted and metadata such as destination address and port number. This structure allows each packet to be sent independently without needing a connection.

:p What information does a typical UDP packet contain?
??x
A typical UDP packet contains:
- Data payload: The actual message or data being sent.
- Source Port: Identifies the sending application.
- Destination Port: Identifies the receiving application on the destination host.
- Length: Specifies the total length of the packet in bytes, including header and data.

:p How do you create a `DatagramPacket` object?
??x
You can create a `DatagramPacket` by providing:
- A byte array that contains the message/data to be sent or received.
- The length of the buffer (which must match the size of the byte array).
- The destination address and port number.

:p What is the purpose of setting the packet's length?
??x
The purpose of setting the packet's length is to specify the amount of data that will be written into the buffer. This ensures that only valid data within the specified range is sent or received.

```java
// Creating a DatagramPacket for sending
DatagramPacket packetToSend = new DatagramPacket(data, data.length, InetAddress.getByName("host"), 12345);
socket.send(packetToSend);

// Setting the length of a packet before receiving
packet.setLength(receivedData.length);
```
x??
---

#### Daytime Service with UDP
The Daytime service is a simple network service that sends the current date and time as text. In this context, both TCP and UDP are used to demonstrate how these protocols handle data transmission differently.

:p What is the purpose of the `DaytimeUDP` class in the provided code?
??x
The `DaytimeUDP` class is designed to connect to a Daytime service running on a specified host using UDP. Its main function is to send an empty packet to request the current date and time from the server and then receive and parse this data.

:p How does the `main` method in `DaytimeUDP` handle connections?
??x
The `main` method handles connections by:
1. Checking if a hostname argument is provided.
2. Resolving the hostname to an `InetAddress`.
3. Creating a `DatagramSocket`.
4. Creating a `DatagramPacket` with a buffer and specifying the server's address and port.
5. Sending an empty packet of maximum length minus one byte to initiate communication.
6. Receiving the response from the server.

:p What is the significance of sending an empty packet for UDP?
??x
Sending an empty packet in UDP can be significant because UDP doesn't establish a connection before data transmission, unlike TCP. By sending an initial packet, even if it's empty, you help the server identify your source and provide a basis for communication.

```java
// Example of creating and sending a DatagramPacket
DatagramSocket socket = new DatagramSocket();
byte[] buffer = new byte[1024];
DatagramPacket packet = new DatagramPacket(buffer, buffer.length, InetAddress.getByName("host"), 13);
packet.setLength(1023); // Sending empty packet except for one null byte
socket.send(packet);

// Example of receiving data with a DatagramPacket
byte[] receiveBuffer = new byte[1024];
DatagramPacket receivePacket = new DatagramPacket(receiveBuffer, receiveBuffer.length);
socket.receive(receivePacket);
String response = new String(receivePacket.getData(), 0, receivePacket.getLength());
```
x??
---

#### URI, URL, and URN Explanation
URI stands for Uniform Resource Identifier. It is a generic term used to identify resources on the internet. A URL (Uniform Resource Locator) is a specific type of URI that includes information about how to retrieve the resource; it contains three main components: the scheme (such as HTTP), the authority (hostname or IP address and port number), and the path.

A URN (Uniform Resource Name) is also a kind of URI but does not provide any means for locating the resource. It is mainly used to identify resources that may be located, moved, renamed, or cease to exist over time.

:p What are the differences between URI, URL, and URN?
??x
URI, URL, and URN are all types of identifiers, with URI being the most general term. A URL specifies a scheme, hostname, and path, while a URN uniquely identifies a resource without specifying how to access it.
x??

---

#### Java Network Daytime UDP Client
The provided code snippet demonstrates a basic implementation of a client that receives the current date/time from a server using UDP. The `DaytimeUDP` class implements this functionality.

:p What does this code do?
??x
This code creates a UDP client that sends a request to a server and prints the received date and time. It first sends a "Sent request" message, then waits for a response from the server. Upon receiving the response, it prints the packet's size and the actual content of the packet as a string.

```java
public class DaytimeUDP {
    public static void main(String[] args) {
        try (Socket sock = new Socket("aragorn", 13)) { // Connect to aragorn on port 13
            byte[] buffer = new byte[256]; // Buffer for receiving data
            System.out.println("Sent request");         // Send a packet
            sock.send(new DatagramPacket(buffer, buffer.length)); // Not shown in the snippet

            DatagramPacket packet = new DatagramPacket(buffer, buffer.length);
            sock.receive(packet);                     // Receive a packet and print it.
            System.out.println("Got packet of size " + packet.getLength());
            System.out.print("Date on " + host + " is "
                             + new String(buffer, 0, packet.getLength()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```
x??

---

#### URI and URL Normalization
The example in the text shows how to normalize URIs and URLs using Java's `URI` class. Normalization involves removing extraneous path segments, such as ".." or ".".

:p How does normalizing a URI work?
??x
Normalizing a URI ensures that it follows certain rules for path components and schemes, making it easier to compare two URIs. The provided code demonstrates how to normalize a URI by using the `normalize()` method from the `URI` class.

```java
public class URIDemo {
    public static void main(String[] args)
            throws URISyntaxException, MalformedURLException {
        URI u = new URI("https://darwinsys.com/java/../openbsd/../index.jsp");
        System.out.println("Raw: " + u);
        URI normalized = u.normalize();
        System.out.println("Normalized: " + normalized);
    }
}
```
x??

---

#### Relativizing URIs
The `relativize` method of the `URI` class is used to create a relative reference from one URI to another. This means it creates a URI that represents the difference between two URIs.

:p How does the relativize method work?
??x
The `relativize` method in Java's `URI` class takes another URI as an argument and returns a new URI that is a relative reference to the given URI, representing the path difference from the current URI. This can be useful for resolving paths in file systems or web applications.

Example:
```java
final URI BASE = new URI("https://darwinsys.com");
System.out.println("Relativized to " + BASE + ": " + BASE.relativize(u));
```

In this case, the `relativize` method is used to create a relative reference from the `BASE` URI to another URI `u`.

x??

---

#### Constructing URLs from URIs
A URL can be created from a URI by converting it back to a string and then using the `URL` class.

:p How do you convert a URI to a URL?
??x
To create a `URL` object from a `URI`, you first need to get the string representation of the `URI` and then use this string to construct a new `URL` object. The following code demonstrates how to do this:

```java
final URI BASE = new URI("https://darwinsys.com");
URL url = new URL(BASE.toString());
System.out.println("URL: " + url);
```

This converts the `URI` back into its string representation and then constructs a `URL` object from that string.

x??

---

#### TFTP UDP Client
The provided text describes a basic implementation of a client for the Trivial File Transfer Protocol (TFTP), which is primarily used in network booting. The code snippet demonstrates how to receive files using UDP.

:p What does the TFTP UDP client do?
??x
The TFTP UDP client receives files from a server using the UDP protocol. It connects to the specified host and port, sends a request, receives the file data, and prints it out in chunks. This is useful for network booting or transferring small files over networks.

This implementation uses basic socket operations to send and receive packets of data between the client and server, following the TFTP protocol rules.

x??

---

#### TFTP Protocol Overview
Background context explaining the TFTP protocol. It is a simple, unreliable, and connectionless file transfer protocol used over UDP. Key points include its use for transferring files between servers and clients, typically in network booting scenarios.

:p What is TFTP?
??x
TFTP stands for Trivial File Transfer Protocol, which is a simple, unreliable, and connectionless file transfer protocol that uses the User Datagram Protocol (UDP) for data transmission. It is often used for bootstrapping processes on devices or servers.
x??

---
#### Client-Server Communication in TFTP
Explanation of how TFTP operates between client and server using UDP.

:p How does communication work in TFTP?
??x
In TFTP, the client contacts the server on well-known UDP port 69. The client initiates by sending a read request (RRQ packet) from a generated port number to the server's port 69. The server responds with a data packet (OP_DATA), and further communication happens using the two generated ports.

Example of packet formats:
- Read Request (RRQ): OP_RRQ, sequence number, filename, null terminator, mode string
- Data Packet: OP_DATA, sequence number, 512 bytes of data

Server then sends data packets (OP_DATA), and client acknowledges them with ACK packets. This cycle continues until all the file is transferred.

Example packet format in Java:
```java
public class TftpPacket {
    private int opcode;
    private short seqNum;
    // Other fields like data, filename etc.
    
    public TftpPacket(int opcode, short seqNum) {
        this.opcode = opcode;
        this.seqNum = seqNum;
    }
}
```
x??

---
#### Packet Structure in TFTP
Explanation of the structure and contents of TFTP packets.

:p What is the format of a TFTP packet?
??x
A TFTP packet consists of:
- Opcode (1 byte): Indicates the type of packet (e.g., RRQ, DATA, ACK)
- Sequence Number (2 bytes): Used for ordering packets
- Data/Other Fields: Variable length depending on the packet type

For example, a data packet contains 512 bytes of data plus 4 bytes for opcode and sequence number. The last packet can vary in size from 4 to 515 bytes.

Example packet format:
```java
public class TftpPacket {
    private int opcode;
    private short seqNum;
    private byte[] data;

    public TftpPacket(int opcode, short seqNum, byte[] data) {
        this.opcode = opcode;
        this.seqNum = seqNum;
        this.data = data;
    }
}
```
x??

---
#### TFTP Packet Types
Explanation of different types of TFTP packets and their use.

:p What are the main packet types in TFTP?
??x
TFTP uses several packet types:
- RRQ (Read Request): Used by clients to request file reading, encoded as 1.
- WRQ (Write Request): For writing files, not discussed here but similar structure.
- DATA: Sent from server to client with data segments, opcode is 3.
- ACK: Acknowledgment packets sent by the client to the server.
- ERROR: Used for error reporting, if needed.

Example of handling RRQ:
```java
public void handleRRQ(String filename) {
    byte[] packet = new byte[PACKET_SIZE];
    TftpPacket rrq = new TftpPacket(1, 0, filename.getBytes());
    outp.setData(packet);
    outp.setAddress(servAddr);
    outp.setPort(TFTP_PORT);
    sock.send(outp);
}
```
x??

---
#### Client-Specific Considerations
Explanation of client-side implementation details.

:p What considerations should a TFTP client take into account?
??x
A TFTP client should consider:
- Generating its own port number for communication.
- Handling timeouts and retransmissions if needed.
- Ensuring proper handling of data packets, ACKs, and errors.

Example of handling timeouts (not implemented in the given code):
```java
try {
    sock.setSoTimeout(5000); // Set timeout to 5 seconds
} catch (SocketException e) {
    System.err.println("Could not set socket timeout: " + e.getMessage());
}
```
x??

---
#### Debugging and Troubleshooting
Explanation of how debugging can be done.

:p How can a developer debug TFTP client issues?
??x
Debugging involves:
- Logging packet contents and their sequence.
- Checking for incorrect data handling or ACKs.
- Monitoring timeouts and retries.

Example logging mechanism:
```java
if (debug) {
    System.err.println("Sending RRQ: " + Arrays.toString(rrq.packet));
}
```
x??

---
#### Server-Specific Considerations
Explanation of server-side implementation details.

:p What considerations should a TFTP server take into account?
??x
A TFTP server should consider:
- Properly handling RRQ and DATA packets.
- Ensuring file access is correct (permissions, etc.).
- Implementing timeouts to avoid indefinite waiting for ACKs.

Example timeout implementation in the server:
```java
try {
    Thread.sleep(5000); // Simulate timeout
} catch (InterruptedException e) {
    System.err.println("Server interrupted: " + e.getMessage());
}
```
x??

---

#### Reading Request Setup
Background context: The `readFile` method sets up and sends a read request to a TFTP server. It involves converting filenames and modes into bytes, then sending them over the network. Java handles byte order naturally for you.

:p What is the process for setting up a read request in this method?
??x
The method initializes a buffer and populates it with necessary information such as operation code (OP_RRQ), filename, and mode. It uses `getBytes()` to convert strings into bytes, then copies these byte arrays into the buffer at specific offsets.

```java
// Initialize buffer
buffer[0] = 0;
buffer[OFFSET_REQUEST] = OP_RRQ; // Request opcode

// Convert filename to bytes and copy into buffer
byte[] bTemp = path.getBytes(); 
System.arraycopy(bTemp, 0, buffer, p, path.length());
p += path.length();
buffer[p++] = 0; // Null terminate string

// Similarly convert mode (e.g., "octet") to bytes in buffer
bTemp = MODE.getBytes();
System.arraycopy(bTemp, 0, buffer, p, MODE.length());
p += MODE.length();
buffer[p++] = 0; // Null terminate mode
```
x??

---

#### Sending the Read Request
Background context: After setting up the buffer with necessary information (filename and mode), this part of the method sends the request to the TFTP server. The length of the buffer is set, and it is sent using `sock.send(outp)`.

:p How does the code send a read request to the TFTP server?
??x
The method sets the length of the output packet (`outp`) based on the current position in the buffer (`p`). It then sends this packet to the TFTP server via `sock.send(outp)`. This initiates the data transfer process.

```java
// Set length of output packet and send request
outp.setLength(p);
sock.send(outp);
```
x??

---

#### Receiving Data Packets
Background context: After sending the read request, this part of the method receives data packets from the TFTP server until it encounters a short packet (indicating end-of-file). It handles error responses and prints out the received data.

:p What is the loop used to receive data packets from the server?
??x
The method uses a `do-while` loop to continuously receive packets from the TFTP server. The loop continues as long as the incoming packet size matches the expected packet size (`PACKET_SIZE`). If an error response (OP_ERROR) is received, it prints the error message and exits.

```java
// Loop for receiving data until end-of-file packet
do {
    sock.receive(inp);
    if (debug)
        System.err.println("Packet # " + Byte.toString(buffer[OFFSET_PACKETNUM])
                           + " RESPONSE CODE " + Byte.toString(buffer[OFFSET_REQUEST]));
    
    if (buffer[OFFSET_REQUEST] == OP_ERROR) {
        System.err.println("rcat ERROR: " + new String(buffer, 4, inp.getLength() - 4));
        return;
    }
    
    if (debug)
        System.err.println("Got packet of size " + inp.getLength());
        
    // Print data from the packet
    System.out.write(buffer, 4, inp.getLength() - 4);
    
    // Acknowledge received packet with an ACK
    buffer[OFFSET_REQUEST] = OP_ACK;
    outp.setLength(4);
    outp.setPort(inp.getPort());
    sock.send(outp);
} while (inp.getLength() == PACKET_SIZE);
```
x??

---

#### Handling End-of-File Packet
Background context: Once the loop ends, it indicates that all data packets have been received. This part of the method handles the end-of-file condition and exits the loop.

:p What happens when the loop ends?
??x
When the loop ends (i.e., a packet smaller than `PACKET_SIZE` is received), the method prints out a message indicating completion, followed by exiting the loop.

```java
if (debug)
    System.err.println("** ALL DONE** Leaving loop, last size " + inp.getLength());
```
x??

---

