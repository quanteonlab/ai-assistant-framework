# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 47)

**Starting Chapter:** Discussion. 10.15 Reading and Writing JAR or ZIP Archives. Problem. Discussion

---

#### Writing Binary Data to a File Using DataOutputStream

Background context: When you need to write binary data, such as integers and floating-point numbers, into a file, `DataOutputStream` is an excellent choice. It allows for writing primitive types directly into a stream. This method ensures that the data written is consistent across different platforms.

:p How do you use `DataOutputStream` to write binary integer and double values to a file?

??x
You can use `DataOutputStream` along with `FileOutputStream` to write primitive types like integers and doubles into a file as binary data. The following example demonstrates how to achieve this:

```java
import java.io.*;

public class WriteBinary {
    public static void main(String[] argv) throws IOException {
        int i = 42;
        double d = Math.PI;
        String FILENAME = "binary.dat";

        // Create a DataOutputStream that writes to the file
        try (DataOutputStream os = new DataOutputStream(new FileOutputStream(FILENAME))) {
            // Write an integer and a double to the output stream
            os.writeInt(i);
            os.writeDouble(d);
        }

        System.out.println("Wrote " + i + ", " + d + " to file " + FILENAME);
    }
}
```
x??

---

#### Reading Binary Data from a File Using DataInputStream

Background context: `DataInputStream` is used for reading binary data written by `DataOutputStream`. It provides methods that read primitive types and interpret the byte stream as the native type of the platform.

:p How do you use `DataInputStream` to read binary integer and double values from a file?

??x
To read binary integers and doubles from a file, you can open an input stream with `FileInputStream`, wrap it in `DataInputStream`, and then use its methods like `readInt()` and `readDouble()`.

```java
import java.io.*;

public class ReadBinary {
    public static void main(String[] argv) throws IOException {
        String FILENAME = "binary.dat";

        try (DataInputStream is = new DataInputStream(new FileInputStream(FILENAME))) {
            // Read an integer from the input stream
            int i = is.readInt();
            // Read a double from the input stream
            double d = is.readDouble();

            System.out.println("Read: " + i + ", " + d);
        }
    }
}
```
x??

---

#### Creating and Extracting from a JAR or ZIP Archive

Background context: Java provides classes like `ZipFile` and `ZipEntry` to handle operations on JAR and ZIP archives. These classes allow you to create, read, and extract files from these archive formats.

:p How do you use the `ZipFile` class to list entries in a JAR or ZIP file?

??x
To work with a `ZipFile`, you first need to construct an instance using either the file name or a `File` object. You can then enumerate through its entries and process each one as needed.

```java
import java.util.zip.*;
import java.io.*;

public class ListArchiveEntries {
    public static void main(String[] argv) throws IOException {
        String fileName = "example.zip";

        try (ZipFile zippy = new ZipFile(fileName)) {
            // Get an enumeration of the entries in the archive
            Enumeration<? extends ZipEntry> all = zippy.entries();
            while (all.hasMoreElements()) {
                ZipEntry entry = all.nextElement();
                if (entry.isDirectory()) {
                    System.out.println("Directory: " + entry.getName());
                } else {
                    System.out.println("File: " + entry.getName());
                }
            }
        }
    }
}
```
x??

---

#### Extracting Files from a ZIP Archive

Background context: The `ZipFile` class also allows you to extract files from the archive. This is useful for scenarios where you need to access individual files within an archive without unzipping it completely.

:p How do you use `ZipEntry` and `ZipFile` to extract files from a ZIP archive?

??x
To extract files from a ZIP archive, you can open a `ZipFile`, retrieve the relevant `ZipEntry`, and then read its contents into a file. The following example demonstrates how to extract all entries in an archive:

```java
import java.util.zip.*;
import java.io.*;

public class ExtractArchiveEntries {
    public static void main(String[] argv) throws IOException {
        String fileName = "example.zip";

        try (ZipFile zippy = new ZipFile(fileName)) {
            // Get an enumeration of the entries in the archive
            Enumeration<? extends ZipEntry> all = zippy.entries();
            while (all.hasMoreElements()) {
                ZipEntry entry = all.nextElement();

                if (!entry.isDirectory()) {
                    System.out.println("Extracting: " + entry.getName());
                    try (InputStream is = zippy.getInputStream(entry)) {
                        // Create the directory structure if needed
                        String dirName = new File(entry.getName()).getParent();
                        if (dirName != null && !new File(dirName).exists()) {
                            Files.createDirectories(Paths.get(dirName));
                        }

                        // Write to a file
                        try (FileOutputStream fos = new FileOutputStream(entry.getName())) {
                            byte[] buffer = new byte[8192];
                            int length;
                            while ((length = is.read(buffer)) > 0) {
                                fos.write(buffer, 0, length);
                            }
                        }
                    }
                }
            }
        }
    }
}
```
x??

---

#### Checking File Types in ZIP Files
Background context: The given Java snippet checks whether a file or directory within a ZIP archive is being processed. This can be useful for understanding the structure of a ZIP file and performing specific actions based on the type of entry.

:p How does the code differentiate between directories and files in a ZIP archive?
??x
The code uses an `if` statement to check if the entry is a directory or not. If it is, it prints "Directory" followed by the zipName; otherwise, it prints "File" followed by the zipName.

```java
if (e.isDirectory()) {
    System.out.println("Directory " + zipName);
} else {
    System.out.println("File " + zipName);
}
```
x??

---

#### Using getResource() for Classpath Resources
Background context: The provided text explains how to use `getResource()` and `getResourceAsStream()` methods from Java's `Class` or `ClassLoader` classes to load resources without referring to their absolute file paths. This is useful in server environments, unit testing, and when the resource might be located within a JAR.

:p How can you use `getClass().getResourceAsStream(String)` to read a properties file?
??x
You can use `getClass().getResourceAsStream("foo.properties")` to get an InputStream to a properties file. This method looks for the specified resource relative to the class loader of the given class and returns it as an InputStream.

```java
InputStream is = getClass().getResourceAsStream("foo.properties");
```

This line of code will attempt to find `foo.properties` in the classpath or JAR files, depending on where your application is running. You can then use this InputStream to read properties from the file.
x??

---

#### Understanding getResource() Methods Variations
Background context: The text lists three variations of `getResource()` methods that can be used to locate resources within a classpath, either as a single URL or as an enumeration of URLs.

:p What are the three forms of `getResource()` and what do they return?
??x
The three forms of `getResource()` are:
1. `public InputStream getResourceAsStream(String name)` - Returns an InputStream for the resource.
2. `public URL getResource(String name)` - Returns a URL for the resource.
3. `public Enumeration<URL> getResources(String name) throws IOException` - Returns an enumeration of URLs for all resources that match the given string.

Each form serves different purposes:
- The first is useful for accessing single resources.
- The second can be used to interpret the URL in various ways.
- The third is intended for finding multiple resources with a given name across directories and JAR files.
x??

---

#### Resource Paths in Maven Projects
Background context: In Maven projects, resources are typically placed in `src/main/resources` or directly under the package directory. This ensures they are included in the classpath when building the project.

:p How do you place resources for absolute and relative paths?
??x
For an absolute path in a Maven project:
- Place the file under `src/main/resources`.
This is where Maven will look for resource files that need to be packaged into your JAR or WAR file.

For a relative path:
- Place the file directly in the package directory.
The resource can then be accessed via its class name, like `MyPackage/two.txt`.

Example placement:
```
src/main/resources/one.txt
src/main/java/MyPackage/two.txt
```

Accessing these from code would look like this:

```java
InputStream isOne = getClass().getResourceAsStream("/one.txt"); // Leading slash indicates resources directory.
InputStream isTwo = getClass().getResourceAsStream("two.txt");  // No leading slash, relative to package.
```
x??

---

#### getResource() and getResourceAsStream()
Background context: The `getResource()` and `getResourceAsStream()` methods are used to retrieve resources from the classpath. These methods are particularly useful for accessing configuration files, templates, or other non-code assets that may be packaged with an application.

:p What do `getResource()` and `getResourceAsStream()` return if no matching resource is found?
??x
If `getResource()` or `getResourceAsStream()` cannot find a matching resource, they will return null. This is important to check for in your code to avoid NullPointerExceptions.
```java
URL resource = MyClass.class.getResource("/path/to/resource.txt");
if (resource == null) {
    // Handle the case where no resource was found
}
```
x??

---

#### getResource() vs getResources()
Background context: `getResource()` returns a URL or ClassLoader object, which can be used to load the resource. On the other hand, `getResources()` returns an Enumeration of URLs that match the specified name.

:p What does `getResources()` return if no matching resources are found?
??x
If no matching resources are found, `getResources()` will return an empty Enumeration.
```java
Enumeration<URL> urls = MyClass.class.getResources("/path/to/resource.txt");
if (!urls.hasMoreElements()) {
    // Handle the case where no resource was found
}
```
x??

---

#### getResource() Path Convention for Packages
Background context: When dealing with package structures (e.g., `package/subpackage`), you need to replace slashes (`/`) with periods (`.`) in your path.

:p How should the file path be formatted when using `getResource()` or `getResourceAsStream()` for a resource under a package structure?
??x
You should use periods (`.`) instead of slashes (`/`) in the file path. For example, if you have a resource named `config.properties` located at `com/example/package/subpackage/config.properties`, you would reference it like this: `com.example.package.subpackage.config`.
```java
URL resource = MyClass.class.getResource("com.example.package.subpackage.config");
if (resource != null) {
    // Use the resource
}
```
x??

---

#### Using java.nio.file.Files for File Information
Background context: The `java.nio.file.Files` class provides methods to retrieve information about files and directories. These methods are useful for determining file existence, size, permissions, etc.

:p What does the `Files.exists()` method return?
??x
The `Files.exists()` method returns a boolean value indicating whether the specified path denotes an existing file or directory.
```java
boolean exists = Files.exists(Path.of("/path/to/file.txt"));
if (exists) {
    // File exists
} else {
    // File does not exist
}
```
x??

---

#### Getting File Information with `Files` Class
Background context: The `Files` class contains a variety of static methods for getting information about files and directories. These include methods like `size()`, `lastModifiedTime()`, `owner()`, etc.

:p How can you check if a file is readable using the `Files` class?
??x
You can use the `isReadable(Path)` method from the `Files` class to determine if a file is readable by the current user.
```java
Path path = Path.of("/path/to/file.txt");
if (Files.isReadable(path)) {
    // The file is readable
} else {
    // The file is not readable
}
```
x??

---

#### Creating Paths with `Path.of()`
Background context: The `Path.of()` method can be used to create a `java.nio.file.Path` object from a string path. This method supports various overloads for different types of paths.

:p How do you create a `Path` object using the `Path.of()` method?
??x
You can create a `Path` object by calling `Path.of("path/to/resource")`.
```java
Path path = Path.of("/path/to/resource.txt");
```
x??

---

#### Using `Files` for Directory and File Operations
Background context: The `Files` class also provides methods that make changes to the filesystem or open files. These are classified as operational methods.

:p How can you check if a file is writable using the `Files` class?
??x
You can use the `isWritable(Path)` method from the `Files` class to determine if a file is writable by the current user.
```java
Path path = Path.of("/path/to/file.txt");
if (Files.isWritable(path)) {
    // The file is writable
} else {
    // The file is not writable
}
```
x??

---

#### Using `Files` for File Content Type Detection
Background context: You can use the `probeContentType(Path)` method to try and determine the MIME type of data in a given path.

:p How can you determine the MIME type of a file using `Files`?
??x
You can use the `probeContentType(Path)` method from the `Files` class. This method attempts to return the MIME type of the data at the specified path.
```java
Path path = Path.of("/path/to/file.txt");
try {
    String contentType = Files.probeContentType(path);
    if (contentType != null) {
        // A MIME type was determined
    } else {
        // No MIME type could be determined
    }
} catch (IOException e) {
    // Handle the exception
}
```
x??

#### Checking File Properties
Background context: This section describes how to use Java NIO.2 (java.nio.file) package to check various properties of files and directories on a filesystem.

:p What does `Files.exists(Path.of("/"));` do?
??x
This method checks if the path `/` exists in the file system. On most Unix-like systems, this will return true as it points to the root directory.

```java
import java.nio.file.Files;
import java.nio.file.Path;

public class FilePropertiesCheck {
    public static void main(String[] args) {
        Path path = Path.of("/");
        System.out.println("exists: " + Files.exists(path));
    }
}
```
x??

---

#### Directory and Executable Checks
Background context: This example demonstrates checking if a directory exists or an executable file is present.

:p What does `Files.isDirectory(Path.of("/"));` do?
??x
This method checks whether the path `/` refers to a directory. On Unix-like systems, this will return true as it points to the root directory containing several directories.

```java
import java.nio.file.Files;
import java.nio.file.Path;

public class DirectoryCheck {
    public static void main(String[] args) {
        Path path = Path.of("/");
        System.out.println("isDirectory: " + Files.isDirectory(path));
    }
}
```
x??

---

#### Reading File Content Type and Size
Background context: This section shows how to determine the content type of a file and its size.

:p What does `Files.probeContentType(Path.of("lines.txt"));` do?
??x
This method attempts to determine the MIME content type (e.g., "text/plain") of the file located at `"lines.txt"`. On a Unix system, it might return `text/plain`.

```java
import java.nio.file.Files;
import java.nio.file.Path;

public class ContentTypeCheck {
    public static void main(String[] args) {
        Path path = Path.of("lines.txt");
        System.out.println("probeContentType: " + Files.probeContentType(path));
    }
}
```
x??

---

#### Checking File Size
Background context: This example illustrates how to find out the size of a file.

:p What does `Files.size(Path.of("lines.txt"));` do?
??x
This method returns the number of bytes in the file located at `"lines.txt"`. On a Unix system, it might return 78 bytes as shown in the table.

```java
import java.nio.file.Files;
import java.nio.file.Path;

public class FileSizeCheck {
    public static void main(String[] args) {
        Path path = Path.of("lines.txt");
        System.out.println("size: " + Files.size(path));
    }
}
```
x??

---

#### Deleting a File
Background context: This section covers how to delete a file using the `Files.delete` method.

:p What does `Files.deleteIfExists(Path.of("no_such_file_as_skjfsjljwerjwj"));` do?
??x
This method attempts to delete the file located at `"no_such_file_as_skjfsjljwerjwj"`. Since this path is likely invalid, it will not throw an exception but return `false`.

```java
import java.nio.file.Files;
import java.nio.file.Path;

public class FileDeletionCheck {
    public static void main(String[] args) {
        Path path = Path.of("no_such_file_as_skjfsjljwerjwj");
        boolean result = Files.deleteIfExists(path);
        System.out.println("Delete Result: " + result);
    }
}
```
x??

---

#### Checking for File Existence
Background context: This example demonstrates checking if a file or directory exists.

:p What does `Files.notExists(Path.of("no_such_file_as_skjfsjljwerjwj"));` do?
??x
This method checks whether the path `"no_such_file_as_skjfsjljwerjwj"` does not exist. Since this is likely an invalid or non-existent file, it will return `true`.

```java
import java.nio.file.Files;
import java.nio.file.Path;

public class NonExistenceCheck {
    public static void main(String[] args) {
        Path path = Path.of("no_such_file_as_skjfsjljwerjwj");
        System.out.println("notexists: " + Files.notExists(path));
    }
}
```
x??

---

#### Checking File Writability
Background context: This section shows how to check if a directory is writable.

:p What does `Files.isWritable(Path.of("/tmp"));` do?
??x
This method checks whether the path `/tmp` is writable. On most Unix systems, this will return true since `/tmp` is typically a writable directory.

```java
import java.nio.file.Files;
import java.nio.file.Path;

public class WritableCheck {
    public static void main(String[] args) {
        Path path = Path.of("/tmp");
        System.out.println("isWritable: " + Files.isWritable(path));
    }
}
```
x??

---

#### Checking File Symbolic Links
Background context: This example illustrates checking if a file is a symbolic link.

:p What does `Files.isSymbolicLink(Path.of("/var"));` do?
??x
This method checks whether the path `/var` is a symbolic link. On most Unix systems, this will return false as `/var` typically points to an actual directory and not a symbolic link.

```java
import java.nio.file.Files;
import java.nio.file.Path;

public class SymbolicLinkCheck {
    public static void main(String[] args) {
        Path path = Path.of("/var");
        System.out.println("isSymbolicLink: " + Files.isSymbolicLink(path));
    }
}
```
x??

---

#### Constructing a Path Object
Background context: In Java, you can construct a `Path` object using various methods. One of these methods is `Path.of(String, String...)`, which constructs a path from given elements.

:p How do you create a `Path` object in Java?
??x
You can create a `Path` object by calling the static method `of()` on the `Path` class and passing one or more string arguments. For example:
```java
Path p = Path.of("/home/user/documents");
```
This creates a path starting from the root directory to `/home/user/documents`.

x??

---

#### Checking File Existence
Background context: To check if a file exists, you can use the `Files.exists(Path)` method.

:p How do you determine if a file or directory exists using Java's NIO API?
??x
You can check if a file or directory exists by calling the static method `exists()` on the `Files` class and passing a `Path` object. For example:
```java
if (Files.exists(p)) {
    System.out.println("File found.");
} else {
    System.out.println("File not found.");
}
```
This code snippet checks if the file or directory represented by `p` exists.

x??

---

#### Getting Canonical Name of a Path
Background context: The canonical name of a path is its normalized form, which resolves all symbolic links and relative paths to an absolute path. This method helps in ensuring that paths are consistent across different environments.

:p How do you obtain the canonical name of a `Path` object?
??x
You can get the canonical name of a `Path` by calling the `normalize()` method on it. For example:
```java
Path p = Path.of("/home/user/documents");
Path canonicalName = p.normalize();
System.out.println("Canonical name: " + canonicalName);
```
This code normalizes the path to its canonical form, resolving any symbolic links and relative paths.

x??

---

#### Checking File Readability and Writability
Background context: You can check if a file is readable or writable using the `Files.isReadable(Path)` and `Files.isWritable(Path)` methods. These methods return true if the respective permission is granted.

:p How do you determine if a file is readable in Java?
??x
You can determine if a file is readable by calling the method `isReadable()` on the `Files` class and passing a `Path` object. For example:
```java
if (Files.isReadable(p)) {
    System.out.println(fileName + " is readable.");
} else {
    System.out.println(fileName + " is not readable.");
}
```
This code checks if the file or directory represented by `p` can be read.

x??

---

#### Determining File Type and Size
Background context: You can determine whether a path refers to a file, directory, or other type using methods like `Files.isRegularFile(Path)` and `Files.isDirectory(Path)`. Additionally, you can get the size of a regular file using `Files.size(Path)`.

:p How do you check if a path is a regular file in Java?
??x
You can use the method `isRegularFile()` on the `Files` class to determine if a path refers to a regular file. For example:
```java
if (Files.isRegularFile(p)) {
    long fileSize = Files.size(p);
    System.out.printf("File size is %d bytes, content type %s", fileSize, Files.probeContentType(p));
} else {
    System.out.println("It's not a file.");
}
```
This code checks if the path represented by `p` points to a regular file and prints its size and content type.

x??

---

#### Getting Last Modified Time
Background context: The modification time of a file can be obtained using the method `Files.getLastModifiedTime(Path)`. This returns a `FileTime` object, which represents the last modified timestamp.

:p How do you get the last modified time of a file in Java?
??x
You can retrieve the last modified time of a file by calling the method `getLastModifiedTime()` on the `Files` class and passing a `Path` object. For example:
```java
final FileTime d = Files.getLastModifiedTime(p);
System.out.println("Last modified " + d);
```
This code retrieves and prints the last modified time of the file or directory represented by `p`.

x??

---

#### Converting Path to File Object
Background context: To work with legacy Java I/O APIs that expect a `File` object, you can convert a `Path` to a `File` using the `toFile()` method.

:p How do you convert a `Path` to a `File` in Java?
??x
You can convert a `Path` to a `File` by calling the `toFile()` method on the `Path` object. For example:
```java
File oldType = p.toFile();
```
This code snippet converts the path `p` to an equivalent `File` object.

x??

---

