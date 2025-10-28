# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 61)

**Starting Chapter:** See Also. 13.11 Network Logging with Log4j. Problem. Discussion

---

---
#### Logging in Java Using Log4j 2.13
Log4j 2 is a flexible and powerful logging framework for Java applications, offering a wide range of features such as hierarchical configuration, multi-appender support, and more sophisticated levels of control over message handling.

The core concept revolves around obtaining a `Logger` object from the `LogManager` class to log messages. Each logger has an associated level (e.g., DEBUG, INFO) which determines whether a message is logged based on its importance.
:p What method is used to obtain a Logger in Log4j 2?
??x
The static method `LogManager.getLogger()` is used to obtain a `Logger` object. This method allows you to specify the name of the class for which you want to get the logger, or you can use the default no-argument constructor if you don't need to customize it.
```java
private static Logger myLogger = LogManager.getLogger();
```
x??

---
#### Configuring Log4j 2 without a Properties File
In Log4j 2, logging output is controlled by a configuration file. If no such file exists or is specified via `-Dlog4j.configurationFile=URL`, the logger will not produce any output unless explicitly configured.

:p What happens if there is no log4j2.properties file present?
??x
If there is no `log4j2.properties` file, and no other configuration file is provided via the system property `-Dlog4j.configurationFile=URL`, then the logging framework will not produce any output for your application.
```java
public class Log4JDemo {
    private static Logger myLogger = LogManager.getLogger();
    
    public static void main(String[] args) {
        Object o = new Object();
        myLogger.info("I created an object: " + o);
    }
}
```
The above program will not produce any logging output if run without a configuration file.
x??

---
#### Levels of Logging in Log4j 2
Log4j 2 uses the `Level` class to define message levels, which determine how messages are filtered and handled. The standard levels from least to most severe are: DEBUG, INFO, WARN, ERROR, and FATAL.

:p What are the five main logging levels in Log4j 2?
??x
The five main logging levels in Log4j 2 are:
- DEBUG (least severe)
- INFO
- WARN
- ERROR
- FATAL (most severe)

These levels help control which messages get logged based on their severity. For example, a logger with a level set to `WARN` will only log WARN and FATAL messages.
```java
private static Logger myLogger = LogManager.getLogger();

public static void main(String[] args) {
    Object o = new Object();
    myLogger.debug("This is a debug message: " + o); // This will be discarded if the logger level is INFO or higher.
}
```
x??

---
#### Using Loggers in Java Applications
Using loggers effectively involves creating instances of `Logger` and using their methods to log messages. The most common methods are `debug()`, `info()`, `warn()`, `error()`, and `fatal()`.

:p How do you use a logger to log an error message?
??x
To log an error message, you would typically call the `error()` method of your logger instance:
```java
private static Logger myLogger = LogManager.getLogger();

public static void main(String[] args) {
    Object o = new Object();
    try {
        // Some code that may throw exceptions.
    } catch (Exception ex) {
        myLogger.error("Caught Exception: " + ex, ex); // Logs the exception message and stack trace.
    }
}
```
This logs an error along with any associated exception information. The second parameter `ex` is used to provide a detailed view of what went wrong.
x??

---
#### Network Logging in Log4j 2
Log4j 2 supports network-based logging through its `net` package, which allows you to send log messages over the network to remote servers.

:p How can you configure Log4j 2 for network logging?
??x
To configure Log4j 2 for network logging, you would typically use a `SocketAppender`. Here is an example configuration snippet:
```properties
appender.socket.type = Socket
appender.socket.name = RemoteSocket
appender.socket.targetHost = localhost
appender.socket.targetPort = 5140
appender.socket.layout.type = PatternLayout
appender.socket.layout.pattern = %d{ISO8601} [%t] %-5p %c{2}: %m%n

rootLogger.level = info
rootLogger.appenderRef.name = RemoteSocket
```
This configuration sets up a socket appender that sends logs to `localhost` on port `5140`, using a specific layout format.

In your Java application, you would configure the appender as follows:
```java
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;
import org.apache.logging.log4j.core.config.AppenderRef;
import org.apache.logging.log4j.core.appender.SocketAppender;

// Setup your configuration and add the socket appender.
Configuration config = ... // Your Configuration setup
LoggerConfig loggerConfig = config.getLoggerConfig("myLogger");
SocketAppender socketAppender = new SocketAppender();
socketAppender.setRemoteHost("localhost");
socketAppender.setPort(5140);
loggerConfig.addAppender(socketAppender, Level.INFO, null);
```
x??

---

---
#### Log4j2 Configuration File Format
Log4j2 configuration files can be written in properties or XML format. Properties are simpler, while XML offers more flexibility and is preferred for complex configurations. The example provided uses a properties file to configure logging levels and appenders.

Properties file content:
```
# Ensure file gets copied for Java Cookbook
# WARNING - log4j2.properties must be on your CLASSPATH,
# not necessarily in your source directory.
rootLogger.level = info 
rootLogger.appenderRef.stdout.ref = STDOUT

appender.console.type = Console
appender.console.name = STDOUT
appender.console.layout.type = PatternLayout
appender.console.layout.pattern =  %p %c - %m%n
appender.console.filter.threshold.type = ThresholdFilter
appender.console.filter.threshold.level = debug
```

:p What does the Log4j2 configuration file do?
??x
The configuration file sets up a root logger with an info level, which means it logs messages at the INFO level and above. It also configures a console appender to output these log messages in a specific format.

For example:
```properties
rootLogger.level = info 
```
Sets the logging level for the root logger to INFO.
```java
appender.console.type = Console
appender.console.layout.pattern =  %p %c - %m%n
```
Configures the console appender and sets a pattern layout that includes the log level (%p), category (%c), message (%m), and new line (%n).
x??

---
#### Log4j2 Logger Levels
Log4j2 uses a hierarchical logging model where levels are defined from most critical to least. Commonly used levels include DEBUG, INFO, WARN, ERROR, and FATAL.

Example configuration:
```
rootLogger.level = debug  # Logs all messages at level DEBUG and above.
```

:p What is the difference between different log levels in Log4j2?
??x
In Log4j2, different log levels represent varying severity of logs. For instance:

- **DEBUG**: Detailed information, typically of interest only when diagnosing problems.
- **INFO**: Confirmation that things are working as expected.
- **WARN**: An indication that something unexpected happened or indicative of some problem in the near future (e.g., 'disk space low'). The software is still functioning.
- **ERROR**: Due to a more serious problem, the software has not been able to perform some function.
- **FATAL**: So severe that the application cannot continue running.

Setting `rootLogger.level = debug` ensures that all logs at the DEBUG level and above are captured. This can be adjusted in the configuration file as needed for different environments or deployment scenarios.
x??

---
#### Handling Exceptions with Logging
Logging caught exceptions is crucial for debugging and maintaining applications. Log4j2 provides a straightforward way to log exceptions using the `error` method, which includes both the exception message and stack trace.

Example code:
```java
public class Log4JDemo2 {
    private static Logger myLogger = LogManager.getLogger();
    public static void main(String[] args) {
        try {
            Object o = new Object();
            myLogger.info("I created an object: " + o);
            if (o == null) {  // Bogo condition to demonstrate logging
                throw new IllegalArgumentException("Just testing");
            }
        } catch (Exception ex) {
            myLogger.error("Caught Exception: " + ex, ex);
        }
    }
}
```

:p How does Log4j2 handle exceptions in a log message?
??x
Log4j2 handles exceptions by logging the caught exception using the `error` method. This method captures both the exception's message and its stack trace.

In the example provided:
```java
myLogger.error("Caught Exception: " + ex, ex);
```
This line logs an error message along with the full stack trace of the exception. The first parameter is a custom message that describes the situation, while the second parameter (`ex`) passes the actual exception object to Log4j2 for logging.
x??

---
#### Configuring Logging Levels Programmatically
In some cases, you might want to dynamically change the log level based on user input or external factors. You can do this by programmatically altering the logger's level.

Example code:
```java
public class Example {
    public static void main(String[] args) {
        Logger rootLogger = LogManager.getRootLogger();
        
        // Set logging level to DEBUG
        Level debugLevel = Level.DEBUG;
        rootLogger.setLevel(debugLevel);
        
        // Log a message at the INFO level
        rootLogger.info("This is an info log message.");
    }
}
```

:p How can you change the logging level dynamically in Log4j2?
??x
You can change the logging level dynamically by accessing the `rootLogger` and setting its level using the `setLevel` method. Here's how:

```java
public class Example {
    public static void main(String[] args) {
        Logger rootLogger = LogManager.getRootLogger();
        
        // Set logging level to DEBUG
        Level debugLevel = Level.DEBUG;
        rootLogger.setLevel(debugLevel);
        
        // Log a message at the INFO level
        rootLogger.info("This is an info log message.");
    }
}
```

In this example:
- `LogManager.getRootLogger()` retrieves the root logger.
- `Level.DEBUG` creates a new instance of the DEBUG level.
- `rootLogger.setLevel(debugLevel)` sets the logging level to DEBUG, allowing all messages at and above this level to be logged.

This approach allows you to adjust the logging verbosity at runtime based on various conditions or user inputs without needing to modify the configuration file.
x??

---
#### SocketAppender for Remote Logging
SocketAppender is used in scenarios where logs need to be sent from a client application to a server. This can be particularly useful for centralized log management.

Example setup:
```properties
appender.socket.type = Socket
appender.socket.name = REMOTE_LOGGING
appender.socket.layout.type = PatternLayout
appender.socket.layout.pattern = %p %c - %m%n
appender.socket.filter.threshold.type = ThresholdFilter
appender.socket.filter.threshold.level = info

appender.socket.locationInfo = false
appender.socket.reconnectionDelay = 10000
```

:p How does SocketAppender work in Log4j2?
??x
SocketAppender works by allowing log messages to be sent over a network connection from the client application to a remote server. This is particularly useful for centralized logging where logs from multiple machines need to be aggregated.

In the example setup:
- `appender.socket.type = Socket` specifies that this appender uses a socket.
- `appender.socket.name = REMOTE_LOGGING` gives this appender a unique name.
- `appender.socket.layout.type = PatternLayout` sets the layout for formatting log messages, such as including the level, category, and message.
- `appender.socket.filter.threshold.level = info` ensures that only INFO level and above messages are sent.

Additional settings like `locationInfo` control whether stack traces are included, and `reconnectionDelay` specifies how long to wait before trying to reconnect if a connection is lost.
x??

---

#### Log4j2 Configuration File - `log4j2-network.properties`
Log4j2 is a logging framework that supports various types of appenders, including socket-based networking. The provided configuration file (`log4j2-network.properties`) demonstrates how to set up log messages to be sent over a network using sockets.

:p What does the `log4j2-network.properties` configuration file do?
??x
The `log4j2-network.properties` configuration file sets up a socket-based appender for Log4j2, allowing log messages to be sent to a listener running on a specific host and port. This is useful for centralized logging or for sending logs to remote servers.

Code example:
```properties
# Log4J2 properties file for the NETWORKED logger demo programs.
rootLogger.level = info
rootLogger.appenderRef.stdout.ref = STDOUT

appender.console.type = Socket
appender.console.name = STDOUT
appender.console.host = localhost
appender.console.port = 6666
appender.console.layout.type = PatternLayout
appender.console.layout.pattern = %m%n
appender.console.filter.threshold.type = ThresholdFilter
appender.console.filter.threshold.level = debug
```
x??

---

#### Java System Property - `log4j.configurationFile`
Java system properties allow you to specify configuration files for logging frameworks like Log4j2. The provided script sets the `log4j.configurationFile` property to point to the `log4j2-network.properties` file.

:p How is the `log4j.configurationFile` property used in this scenario?
??x
The `log4j.configurationFile` property is set using a Java system property to specify the configuration file for Log4j2. This ensures that Log4j2 uses the specified configuration when initializing its logging setup.

Example command:
```bash
java -Dlog4j.configurationFile=log4j2-network.properties \
-classpath ".:${build}:${log4j2_jar}" logging.Log4JDemo
```

x??

---

#### Running Demo Programs with Java Classpath
The provided script runs two demo programs (`Log4JDemo` and `Log4JDemo2`) using a specific classpath that includes the Log4j2 JAR files.

:p How are the demo programs run in this scenario?
??x
The demo programs, such as `logging.Log4JDemo`, are run with a Java command that sets the classpath to include both the current directory and the compiled classes along with the required Log4j2 JARs. This ensures that the correct version of the configuration file is used.

Example commands:
```bash
echo "==> Log4JDemo"
java -Dlog4j.configurationFile=log4j2-network.properties \
-classpath ".:${build}:${log4j2_jar}" logging.Log4JDemo

echo "==> Log4JDemo2"
java -Dlog4j.configurationFile=log4j2-network.properties \
-classpath ".:${build}:${log4j2_jar}" logging.Log4JDemo2
```

x??

---

#### Setting Up a Listener with `nc` (Netcat)
To receive the log messages sent by the demo programs, you need to set up a listener. On Unix systems, `netcat` (or `nc`) can be used for this purpose.

:p How do you set up a listener using netcat (`nc`)?
??x
You can use `netcat` to listen on a specific port and receive log messages sent over the network. The `-l` option tells `nc` to listen, and the `-k` option keeps it listening even after connections are closed.

Example command:
```bash
$ nc -kl 6666
```

x??

---

#### Handling Log Messages in Demo Programs
The demo programs (`Log4JDemo` and `Log4JDemo2`) generate log messages and handle exceptions, demonstrating the use of logging with Log4j2.

:p What does the Log4JDemo program do?
??x
The `logging.Log4JDemo` program creates a Java object, logs information about its creation, and then throws an exception to demonstrate how Log4j2 handles logging in different scenarios. The log messages are sent over a network socket connection to a listener.

Example code:
```java
public class Log4JDemo {
    public static void main(String[] args) {
        System.out.println("I created an object: " + new Object());
        // Simulating some operation that might fail
        throw new IllegalArgumentException("Just testing");
    }
}
```

x??

---

#### Exception Handling in Demo Programs
The demo programs include exception handling to demonstrate how log messages are recorded when exceptions occur.

:p What happens if an exception is thrown in the `Log4JDemo2` program?
??x
If an exception, such as `IllegalArgumentException`, is thrown in the `logging.Log4JDemo2` program, Log4j2 captures the stack trace and logs it along with a message. This helps in diagnosing issues during development.

Example code:
```java
public class Log4JDemo2 {
    public static void main(String[] args) {
        try {
            System.out.println("I created an object: " + new Object());
        } catch (Exception e) {
            // Exception handling logic here
            throw e;
        }
    }
}
```

x??

---

---
#### Expensive String Operations in Loggers
Background context: In logging, especially when using expensive operations like `toString()` or string concatenations within a log message, it is important to avoid performing these operations unless necessary. This can lead to performance issues if the resultant string is not used.

The preferred method nowadays is to create the string inside a lambda expression so that the string operations are only performed when the logging level allows it.

:p How does creating a string inside a lambda expression help with logging performance?
??x
Creating a string inside a lambda expression ensures that the string concatenation and other expensive operations are only executed if the logger is set to a level that requires logging. This means that if, for example, `Log.info()` is called but the log level is not info or higher, the string concatenation will not occur.

Example in code:
```java
myLogger.info(() -> String.format("Value %d from Customer %s", customer.value, customer));
```
In this case, only if the logger's level allows `info` logs, the lambda expression inside `Log.info()` will be evaluated. Otherwise, no string operations are performed.

x?
---
#### Logging with Java Util Logging
Background context: The Java logging mechanism (package java.util.logging) is an alternative to Log4j and has similar capabilities for logging messages and exceptions.

:p How do you acquire a Logger object in the Java logging framework?
??x
You acquire a `Logger` object by calling `Logger.getLogger()` with a descriptive string. This string can be any unique identifier, such as your package or class name.

Example in code:
```java
Logger myLogger = Logger.getLogger("com.darwinsys");
```
This method returns an instance of the `Logger` class, which you can use to log messages and handle exceptions.

x?
---
#### Using Log Records with Java Util Logging
Background context: The Java logging framework provides various methods for logging different levels of severity. Each logger has a level set to determine what kind of logs it will allow.

:p How does the `log` method with a `Supplier` argument help in avoiding unnecessary computations?
??x
The `log` method that accepts a `Supplier` argument allows you to delay the creation and computation of log messages until they are actually needed. This is beneficial because it avoids performing expensive operations unless the logging level requires those logs.

Example in code:
```java
myLogger.log(Level.INFO, () -> String.format("Value %d from Customer %s", customer.value, customer));
```
Here, the `Supplier` inside `log` ensures that `String.format` is only called if the logger's current log level allows logging at the `INFO` level.

x?
---
#### Logging Exceptions with Java Util Logging
Background context: Exception handling in logging often involves catching exceptions and logging them for debugging purposes. The Java logging framework provides methods to log caught exceptions, allowing you to capture detailed information about errors that occur during program execution.

:p How can you log a caught exception using the `log` method in java.util.logging?
??x
You can use the `log` method with a specific level and pass an `Exception` object directly. This will create a log record with the appropriate severity level and attach the thrown exception to it.

Example in code:
```java
logger.log(Level.SEVERE, "Caught Exception", t);
```
Here, `t` is the `Throwable` instance representing the caught exception. The message and the exception are logged at the `SEVERE` level.

x?
---

---
#### Java 8 Lambda Demonstration
Background context explaining how Java 8 introduced lambda expressions to avoid unnecessary object creation. Lambdas allow you to pass functions as parameters, which can reduce code bloat and improve performance by eliminating the need for anonymous inner classes.

:p What is demonstrated in this example regarding logging with Java 8 Lambdas?
??x
The example demonstrates using a lambda expression with `Logger`'s `finest` method to log information about an object without creating an intermediate anonymous class. When `myLogger.finest(() -> "I created this object: " + o);` is called, the lambda itself acts as the logging message.

If you change the logging level from `finest` to `info`, both the system trace and the logging output will be visible because the lambda expression would then be evaluated. However, in this case, since `finest` is not enabled, the lambda expression's string concatenation is optimized away by the JVM, avoiding unnecessary object creation.

```java
public class JulLambdaDemo {
    public static void main(String[] args) {
        Logger myLogger = Logger.getLogger("com.darwinsys.jullambda");
        Object o = new Helper();
        // Lambda to log information about the object
        myLogger.finest(() -> "I created this object: " + o);
    }

    static class Helper {
        public String toString() {
            System.out.println("JulLambdaDemo.Helper.toString()");
            return "failure.";
        }
    }
}
```
x??

---
#### Security Considerations in Network Programming
Background context explaining the importance of security when writing server-side applications. Misconfigured or poorly written server programs can compromise network security, making it crucial to understand and implement robust security measures.

:p What are some key points regarding security in server-side Java applications?
??x
Security is critical for server-side Java applications because a single misconfiguration or poor implementation can compromise the entire network's security. For example, logging sensitive information at incorrect levels can expose data that should remain confidential. Additionally, ensuring proper authentication and authorization mechanisms, as well as securing data transmission through encryption, are essential.

Books like "Java Network Programming" by Elliotte Rusty Harold provide a good foundation on this topic, while more specialized books such as "Firewalls and Internet Security" by William R. Cheswick et al., and the series of books with "Hacking Exposed" in the title can offer deeper insights into security best practices.

x??

---
#### Server-Side Java Using Sockets
Background context explaining how server-side applications using sockets handle network communication. Sockets allow for bidirectional data transfer between a client and server, facilitating various types of network interactions.

:p What is an example technology that could be used to implement a chat server in Java?
??x
An RMI (Remote Method Invocation) chat server can be implemented using the `java.rmi` package. This approach allows remote objects to be accessed across different machines as if they were local, enabling complex distributed applications where methods on remote objects can be invoked.

Here's a simplified example of how you might set up an RMI chat server:

```java
// Server side (RMI Chat Server)
import java.rmi.Naming;
import java.rmi.registry.LocateRegistry;

public class RMIServer {
    public static void main(String[] args) {
        try {
            // Create the chat service object and bind it to a registry
            ChatService service = new ChatServiceImpl();
            LocateRegistry.createRegistry(1099);
            Naming.rebind("ChatService", service);
            System.out.println("RMI chat server started.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

// Client side (RMI Chat Client)
import java.rmi.Naming;

public class RMIClient {
    public static void main(String[] args) {
        try {
            // Look up the chat service from the registry
            ChatService service = (ChatService) Naming.lookup("rmi://localhost/ChatService");
            // Use the service to send a message
            String message = "Hello, RMI!";
            service.sendMessage(message);
            System.out.println("Message sent: " + message);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

x??

---

