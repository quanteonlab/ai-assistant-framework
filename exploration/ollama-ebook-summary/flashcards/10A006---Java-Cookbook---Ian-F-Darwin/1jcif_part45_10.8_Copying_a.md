# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 45)

**Starting Chapter:** 10.8 Copying a File. Problem. Solution. 10.9 Reassigning the Standard Streams

---

#### Java 11 Files.copy() Methods
Background context explaining how to copy a file using the modern Java 11 `Files` class. The `Files.copy()` method provides several overloads to easily copy files between paths or from an input stream to a path.

:p What are some of the methods available in the `java.nio.file.Files` class for copying files?
??x
The `java.nio.file.Files` class offers multiple overloaded `copy` methods:
- Path copy(Path, Path, CopyOption ...)
- long copy(InputStream , Path, CopyOption ...)
- long copy(Path, OutputStream )

These methods simplify file copying significantly compared to older Java versions. Here's an example of using one of these methods:

```java
Path source = Paths.get("path/to/source/file");
Path target = Paths.get("path/to/target/file");

// Example usage with path objects:
try {
    Files.copy(source, target);
} catch (IOException e) {
    // Handle exception
}
```

x??

---

#### Using Explicit Read and Write Methods
Background context explaining that for older Java releases, you might need to use explicit read and write methods in the `InputStream`, `OutputStream` or `Readers`, `Writers`. This is a more manual process compared to using the modern `Files.copy()` method.

:p What approach would you take if your application uses an older version of Java (pre-Java 11)?
??x
For older versions of Java, you might need to use explicit read and write methods from the `InputStream`, `OutputStream` or `Readers`, `Writers`. Here’s a simple example using `InputStream` and `Path`:

```java
import java.io.*;

public class FileCopyExample {
    public static void main(String[] args) {
        Path source = Paths.get("path/to/source/file");
        Path target = Paths.get("path/to/target/file");

        try (InputStream is = new BufferedInputStream(new FileInputStream(source));
             OutputStream os = Files.newOutputStream(target)) {

            byte[] buffer = new byte[1024];
            int length;
            while ((length = is.read(buffer)) > 0) {
                os.write(buffer, 0, length);
            }
        } catch (IOException e) {
            // Handle exception
        }
    }
}
```

This code reads from a source file and writes to a target file using `InputStream` and `OutputStream`.

x??

---

#### FileIoDemo Class for Older JDK Versions
Background context explaining how the `com.darwinsys.util.FileIO` class can be used as an alternative in older JDK versions. This class provides helper methods that simplify common I/O operations.

:p How does the `FileIoDemo.java` program copy files using the `FileIO` utility?
??x
The `FileIoDemo.java` program uses the `com.darwinsys.util.FileIO.copyFile()` method to copy files from one location to another. Here’s an example:

```java
package com.darwinsys.io;

import java.io.IOException;

public class FileIoDemo {
    public static void main(String[] av) {
        try {
            FileIO.copyFile("FileIO.java", "FileIO.bak");
            FileIO.copyFile("FileIO.class", "FileIO-class.bak");
        } catch (IOException e) {
            System.err.println(e);
        }
    }
}
```

This program copies `FileIO.java` to a backup file named `FileIO.bak` and also copies the compiled class file (`FileIO.class`) to a backup file named `FileIO-class.bak`.

x??

---

#### Discussion on Copying Files in Older Java Versions
Background context explaining that older versions of Java did not provide built-in methods for common I/O operations, leading users to create their own utility classes like `com.darwinsys.util.FileIO`.

:p What was the situation with file copying in older JDK versions before the introduction of modern `Files.copy()` methods?
??x
Before the introduction of modern `Files.copy()` methods, older JDK versions lacked built-in utilities for common I/O operations such as file copying. Users had to implement their own utility classes or use external libraries.

For example, using a custom utility class like `FileIO` from the `com.darwinsys.util` package:

```java
package com.darwinsys.util;

import java.io.*;

public class FileIO {
    public static void copyFile(String src, String dst) throws IOException {
        Path source = Paths.get(src);
        Path target = Paths.get(dst);

        try (InputStream is = new BufferedInputStream(new FileInputStream(source));
             OutputStream os = Files.newOutputStream(target)) {

            byte[] buffer = new byte[1024];
            int length;
            while ((length = is.read(buffer)) > 0) {
                os.write(buffer, 0, length);
            }
        } catch (IOException e) {
            // Handle exception
        }
    }
}
```

This utility class manually handles file copying by reading from an `InputStream` and writing to an `OutputStream`.

x??

---

#### Reassigning Standard Streams in Java
This section discusses how to change the standard input, output, and error streams (`System.in`, `System.out`, `System.err`) to redirect or reassign them. This is useful for managing where your program writes its console outputs and errors without manually changing each print or println statement.

:p How can you change where Java programs write their console outputs?
??x
You can use the `System.setOut(PrintStream out)` method to replace `System.out`, and `System.setErr(PrintStream err)` to replace `System.err`. This allows you to redirect these streams, for example, to log files instead of the console.

```java
String LOGFILENAME = "error.log";
// Redirect System.err to a file named 'error.log'
System.setErr(new PrintStream(new FileOutputStream(LOGFILENAME)));
System.out.println("Please look for errors in " + LOGFILENAME);
```
x??

---
#### Merging Standard Streams
This technique is useful when you want to combine the outputs of `System.out` and `System.err`.

:p How can you merge standard output and error streams?
??x
You can use `System.setErr(System.out)` or `System.setOut(System.out)` to redirect both streams to the same destination. This means any errors will be written to the same place as normal outputs.

```java
// Merge stderr and stdout to the same file
System.setErr(System.out);
```
x??

---
#### Using Streams from Other Processes
This involves using streams that are connected to or from another process, such as a network socket or URL.

:p How can you use streams from other processes?
??x
Streams from other processes can be used by creating an `InputStream` or `PrintStream` and passing it to the appropriate set method in the `System` class. For example, if you have a `Process` that outputs data, you can redirect its output to your application's standard output stream.

```java
// Example of using a process's output stream
Process p = Runtime.getRuntime().exec("some_command");
InputStream procOut = p.getInputStream();
PrintStream sysOut = new PrintStream(System.out);
new Thread(() -> {
    byte[] buffer = new byte[1024];
    int len;
    while ((len = procOut.read(buffer)) != -1) {
        sysOut.write(buffer, 0, len);
    }
}).start();
```
x??

---

#### TeePrintStream Class Overview
TeePrintStream is an extension of PrintStream that duplicates output to both a secondary stream and its intended destination. This mimics the Unix `tee` command, allowing for simultaneous writing to standard streams and log files.

:p What does the TeePrintStream class do?
??x
The TeePrintStream class extends the functionality of Java's PrintStream by duplicating any written data to an additional output stream (like a file), while also directing it to its intended destination. This can be useful for logging errors or standard outputs to both console and file simultaneously.
```java
public class TeePrintStream extends PrintStream {
    // Class implementation goes here
}
```
x??

---

#### Constructor Methods of TeePrintStream
The constructors of the TeePrintStream class initialize a new instance that can duplicate output to an existing PrintStream and either an OutputStream or a filename.

:p What are the constructor methods in TeePrintStream?
??x
There are several constructors provided for initializing a TeePrintStream. These include:

1. `public TeePrintStream(PrintStream orig, OutputStream os, boolean flush)` - This is the main constructor that sets up the TeePrintStream with an existing PrintStream, an OutputStream, and a boolean to control auto-flush.
2. `public TeePrintStream(PrintStream orig, OutputStream os) throws IOException` - A simplified version of the first constructor where auto-flush is set to true by default.
3. `public TeePrintStream(PrintStream os, String fn) throws IOException` - Constructs a TeePrintStream with an existing PrintStream and a filename for output.
4. `public TeePrintStream(PrintStream orig, String fn, boolean flush) throws IOException` - Similar to the first but allows specifying auto-flush behavior.

```java
public class TeePrintStream extends PrintStream {
    public TeePrintStream(PrintStream orig, OutputStream os, boolean flush) 
        throws IOException {
        super(os, true);
        fileName = UNKNOWN_NAME;
        parent = orig;
    }
    
    // Other constructors here...
}
```
x??

---

#### Writing Mechanism in TeePrintStream
The write methods of the TeePrintStream class are overridden to ensure that data is written both to the original stream and another destination.

:p How does the write method work in TeePrintStream?
??x
In TeePrintStream, the `write` methods are overridden to perform a "tee" operation. This means the data being written to one output stream (the parent) is also copied to another output stream (this instance).

```java
public void write(int x) {
    parent.write(x);  // Write once;
    super.write(x);   // Write somewhere else.
}
```

Similarly, there's an overloaded version for writing byte arrays:

```java
public void write(byte[] x, int o, int l) {
    parent.write(x, o, l);  // Write once;
    super.write(x, o, l);   // Write somewhere else.
}
```
x??

---

#### CheckError Method in TeePrintStream
The `checkError` method ensures that any error in either the original or the duplicate stream will be detected.

:p What does the checkError method do?
??x
The `checkError` method checks if there has been an error in either the parent (original) PrintStream or the current instance of TeePrintStream. If either stream reports an error, then `checkError` returns true.

```java
public boolean checkError () {
    return parent.checkError() || super.checkError();
}
```
x??

---

#### Close and Flush Methods in TeePrintStream
The `close` and `flush` methods ensure that both the original and duplicate streams are closed or flushed appropriately.

:p How do the close and flush methods work in TeePrintStream?
??x
Both the `close` and `flush` methods call their counterparts on both the parent PrintStream and this instance of TeePrintStream to ensure that all buffered data is written out and resources are properly released.

```java
public void close() {
    parent.close();
    super.close();
}

public void flush() {
    parent.flush();
    super.flush();
}
```
x??

---

