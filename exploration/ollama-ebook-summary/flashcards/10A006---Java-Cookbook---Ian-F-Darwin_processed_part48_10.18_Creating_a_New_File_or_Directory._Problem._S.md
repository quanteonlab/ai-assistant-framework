# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 48)

**Starting Chapter:** 10.18 Creating a New File or Directory. Problem. Solution. 10.19 Changing a Files Name or Other Attributes

---

#### Creating a New File or Directory
Background context: This section discusses how to create new files and directories using Java’s `Files` and `File` classes. It highlights the advantages of using these methods over traditional file handling techniques, such as `FileOutputStream` or `FileWriter`.

:p What is the recommended way to create an empty file in Java?
??x
The `java.nio.file.Files` class provides a method called `createFile(Path)` that can be used to create a new empty file. This method ensures that the file is created without requiring additional handling for writing data.

Example code:
```java
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class CreateEmptyFile {
    public static void main(String[] args) {
        Path path = Paths.get("/path/to/file.txt");
        try {
            Files.createFile(path);
            System.out.println("File created successfully.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```
x??

---

#### Creating a Directory
Background context: This section explains how to create directories using the `Files` class. It mentions that there are different methods available, such as `createDirectory()` and `createDirectories()`, which can be used depending on whether you want to create a single directory or multiple nested directories.

:p How do you create a directory in Java?
??x
To create a directory, you can use the `Files.createDirectory(Path)` method. If you need to create multiple nested directories, you should use `Files.createDirectories(Path)`. This ensures that all necessary parent directories are also created if they don't already exist.

Example code:
```java
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class CreateDirectory {
    public static void main(String[] args) {
        Path dirPath = Paths.get("/path/to/new/directory");
        
        try {
            // Creates a single directory if it doesn't exist
            Files.createDirectory(dirPath);
            System.out.println("Single directory created successfully.");
            
            // Creates multiple nested directories if they don't exist
            Path multiDirPath = Paths.get("/path/to/multiple/nested/directories");
            Files.createDirectories(multiDirPath);
            System.out.println("Nested directories created successfully.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```
x??

---

#### Creating Multiple Empty Files
Background context: The provided text discusses how to create multiple empty files using the `Creat` class. This is done by iterating over an array of filenames and calling the `Files.createFile(Path)` method for each filename.

:p How does the `main` method in the `Creat` class work?
??x
The `main` method in the `Creat` class iterates through an array of filenames provided as command-line arguments. For each filename, it checks if a valid filename is provided and then uses the `Files.createFile(Path)` method to create an empty file.

Example code:
```java
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Creat {
    public static void main(String[] argv) throws IOException {
        // Ensure that a filename (or something) was given in argv[0]
        if (argv.length == 0) {
            throw new IllegalArgumentException("Usage: Creat filename [...]");
        }

        for (String filename : argv) {
            Path path = Paths.get(filename);
            try {
                Files.createFile(path);
                System.out.println("File " + filename + " created successfully.");
            } catch (IOException e) {
                System.err.println("Failed to create file: " + filename);
                e.printStackTrace();
            }
        }
    }
}
```
x??

---

#### Creating Directories and Writing to Them
Background context: The text mentions the need to create a directory before writing files into it. It describes using `Files.createDirectory()` or `Files.createDirectories()` methods for this purpose.

:p How do you ensure that a directory exists before writing files in Java?
??x
Before writing files, you should ensure that the target directory exists. You can use the `Files.createDirectory(Path)` method to create a single directory if it does not exist, or `Files.createDirectories(Path)` if you need to create multiple nested directories.

Example code:
```java
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class EnsureDirExists {
    public static void main(String[] args) {
        Path dirPath = Paths.get("/path/to/new/directory");
        
        try {
            // Creates a single directory if it doesn't exist
            Files.createDirectory(dirPath);
            System.out.println("Directory created successfully.");
            
            // Creates multiple nested directories if they don't exist
            Path multiDirPath = Paths.get("/path/to/multiple/nested/directories");
            Files.createDirectories(multiDirPath);
            System.out.println("Nested directories created successfully.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```
x??

---

#### Path Construction and File Creation
Background context: The provided text discusses how to construct a `Path` object using `Path.of()` method, which is used for file or directory paths. It also mentions that while constructing a `Path`, no changes are made to the disk; however, creating a file with `Files.createFile()` does modify the filesystem by actually creating a file.

:p How do you construct a `Path` object and create a file in Java?
??x
You can construct a `Path` object using `Path.of(arg)`. However, this method only creates a path reference and doesn't affect the disk. To create an actual file on the filesystem, you need to use `Files.createFile(p)`, where `p` is the `Path` object.

```java
for (String arg : argv) {
    final Path p = Path.of(arg);
    try {
        final Path created = Files.createFile(p);
        System.out.println(created);
    } catch (IOException e) {
        e.printStackTrace();
    }
}
```

x??

---

#### OpenOptions in File Creation
Background context: The text mentions that `Files.createFile()` has an overload method that takes a second argument of type `OpenOption`. `OpenOption` is an empty interface implemented by the `StandardOpenOption` enumeration, which provides various options for file operations.

:p What are `OpenOptions`, and how do they affect file creation in Java?
??x
`OpenOptions` are additional parameters used when creating or opening a file. They allow you to specify different behaviors such as creating new files if they don't exist, appending data, etc. For example, the `StandardOpenOption.CREATE` option can be used along with `Files.createFile()`.

```java
for (String arg : argv) {
    final Path p = Path.of(arg);
    try {
        final Path created = Files.createFile(p, StandardOpenOption.CREATE);
        System.out.println(created);
    } catch (IOException e) {
        e.printStackTrace();
    }
}
```

x??

---

#### Directory Creation: createDirectory() vs. createDirectories()
Background context: The text explains that `createDirectory()` and `createDirectories()` are two methods for creating directories, with the former only creating a single directory if it doesn't exist, while the latter creates all necessary intermediate directories.

:p What is the difference between `createDirectory()` and `createDirectories()` in Java's NIO.2 API?
??x
`createDirectory()` attempts to create a single directory specified by the given `Path`. If the directory already exists or if there are any issues, an exception will be thrown. On the other hand, `createDirectories()` creates all necessary directories along the path up to and including the target directory.

```java
try {
    Path dir = Path.of("/Users/ian/once/twice/again");
    Files.createDirectories(dir); // Creates /Users/ian/, once/, twice/, and again/
} catch (IOException e) {
    e.printStackTrace();
}
```

x??

---

#### Changing File Attributes
Background context: The text discusses methods for changing a file's attributes, such as setting it to read-only or modifying its modification time. This is achieved through various `Files` class methods that offer different options.

:p How do you change the attributes of a file in Java?
??x
To change the attributes of a file, you can use methods like `Files.setAttribute()`, which allows setting attributes such as readonly status and modification time. Here’s an example:

```java
try {
    Path path = Path.of("/path/to/file");
    Files.setAttribute(path, "read-only", true); // Set to read-only
} catch (IOException e) {
    e.printStackTrace();
}
```

x??

---

#### Renaming a File Using Java NIO.2
Background context: The `Files.move()` method is used to rename or move files in Java, similar to Unix commands like `mv`. It takes two paths as arguments - the existing file and the new name where it should be moved/renamed.
:p How do you rename or move a file using Java NIO.2?
??x
To rename or move a file, use the `Files.move()` method which requires two Path objects: one for the current location of the file and another for the desired location (new name). Here is an example:

```java
public class Rename {
    public static void main(String[] argv) throws IOException  {
        // Construct the Path object. Does NOT create a file on disk.
        final Path p = Path.of("MyCoolDocument"); // The file we will rename

        // Setup for the demo: create a new "old" file
        final Path oldName = Files.exists(p) ? p : Files.createFile(p);

        // Rename the backup file to "mydoc.bak"
        // Renaming requires a Path object for the target.
        final Path newName = Path.of("mydoc.bak");
        Files.deleteIfExists(newName); // In case previous run left it there
        Path p2 = Files.move(oldName, newName);
        System.out.println(p + " renamed to " + p2);
    }
}
```
x??

---

#### Changing File Attributes in Java NIO.2
Background context: The `Files` class provides several methods for changing file attributes such as executable, readable, writable permissions and last modified time.
:p How do you change the attributes of a file using the `Files` class?
??x
You can change various attributes of a file using different setter methods provided by the `Files` class. Here are some examples:

```java
// Set executable permission for the file
boolean result = Files.setExecutable(p, true);
System.out.println("Set Executable: " + result);

// Set readable and writable permissions
result = Files.setReadable(p, false); // or true for read/write access
result = Files.setWritable(p, true);

// Change last modified time of a file
long newTime = System.currentTimeMillis(); // Example timestamp
Files.setLastModifiedTime(p, FileTime.from(Instant.ofEpochMilli(newTime)));
```
x??

---

#### Using `setLastModifiedTime()` Method
Background context: The `setLastModifiedTime()` method allows you to set the last-modified time of a file or directory. It is useful in scenarios such as backup/restore programs.
:p How do you use the `setLastModifiedTime()` method?
??x
The `setLastModifiedTime()` method can be used to modify the last modified timestamp of a file or directory. The method takes an argument representing the time in milliseconds since the Unix epoch (January 1, 1970).

```java
// Example usage:
long newTime = System.currentTimeMillis(); // Current system time in milliseconds
Files.setLastModifiedTime(p, FileTime.from(Instant.ofEpochMilli(newTime)));

// Getting original value:
long originalTime = p.toFile().lastModified();
```
x??

---

#### Setting Permissions Using `setExecutable()`
Background context: The `setExecutable()` method can be used to set or clear the executable permission for a file.
:p How do you use the `setExecutable()` method?
??x
The `setExecutable()` method is used to enable or disable the execute permission on a file. It has two variants - one with a single boolean parameter and another that also takes an ownerOnly flag.

```java
// Enable executable permission for everyone:
boolean result = Files.setExecutable(p, true);
System.out.println("Set Executable: " + result);

// Set execute permission only for the owner:
result = Files.setExecutable(p, false, true);
```
x??

---

#### Setting Readability Using `setReadable()`
Background context: The `setReadable()` method allows you to set or clear the read permission on a file.
:p How do you use the `setReadable()` method?
??x
The `setReadable()` method is used to enable or disable the readable permission for a file. It also has two variants - one with a single boolean parameter and another that takes an ownerOnly flag.

```java
// Enable read permission for everyone:
boolean result = Files.setReadable(p, true);
System.out.println("Set Readable: " + result);

// Set read permission only for the owner:
result = Files.setReadable(p, false, true);
```
x??

---

#### Setting Writability Using `setWritable()`
Background context: The `setWritable()` method allows you to set or clear the writable permission on a file.
:p How do you use the `setWritable()` method?
??x
The `setWritable()` method is used to enable or disable the write permission for a file. It also has two variants - one with a single boolean parameter and another that takes an ownerOnly flag.

```java
// Enable write permission for everyone:
boolean result = Files.setWritable(p, true);
System.out.println("Set Writable: " + result);

// Set write permission only for the owner:
result = Files.setWritable(p, false, true);
```
x??

---

