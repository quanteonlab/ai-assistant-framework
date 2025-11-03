# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 67)

**Starting Chapter:** 15.7 Packaging Web Tier Components into a WAR File

---

#### Creating a JAR File
Background context: This section explains how to create and run Java programs packaged as JAR files. The `javac` command is used to compile Java source files, while the `jar` command is used to package these into JAR files along with their manifest information.

:p How do you prepare a Java program for packaging into a JAR file?
??x
To prepare your Java program for packaging into a JAR file, first, compile the Java source code using the `javac` command. For example:

```shell
C:> javac HelloWorld.java
```

This compiles the `HelloWorld.java` file and generates the corresponding class file.

Then, create a manifest file that includes the main-class attribute to specify which class should be executed when running the JAR file.
x??

---

#### Creating Manifest File Entry
Background context: The manifest file is crucial for specifying metadata about your JAR file. It contains information like the main class, classpath settings, and other attributes.

:p How do you create a manifest entry to indicate the main-class?
??x
To create a manifest entry indicating the main-class, use a text editor to write a line in the format `Main-Class: com.somedomainhere.HelloWorld`. This tells the Java runtime which class should be executed as the starting point of your application.

For example, if you are running the `HelloWorld` class from the `com.somedomainhere` package, you would include this entry:

```
Main-Class: com.somedomainhere.HelloWorld
```

Place this line in a file named `manifest.stub`.
x??

---

#### Packaging with jar Command
Background context: The `jar` command is used to create JAR files. You can use it to package the compiled classes along with any manifest information.

:p How do you package the program into a JAR file?
??x
To package the program into a JAR file, first, ensure that your manifest file (`manifest.stub`) contains the `Main-Class` attribute as described earlier. Then, run the `jar` command to create the JAR file:

```shell
C:> jar cvmf manifest.stub hello.jar HelloWorld.class
```

This command creates a JAR file named `hello.jar`, including the `HelloWorld.class` file and references the `manifest.stub` for metadata.

The `c` option stands for creating a new archive, `v` enables verbose mode to display what is being done, `m` uses a manifest file, and `f` specifies the name of the JAR file.
x??

---

#### Running the Program with java -jar
Background context: Once you have created your JAR file, you can run it using the `java -jar` command. This tells Java to execute the main method from the specified class within the JAR.

:p How do you run a program packaged in a JAR file?
??x
To run a program packaged in a JAR file, use the `java -jar` option:

```shell
C:> java -jar hello.jar
```

This command tells Java to execute the main method from the specified class within the `hello.jar` file.

For example, if your manifest specifies that `com.somedomainhere.HelloWorld` is the main-class, running this command will start the application.
x??

---

#### Automating with Maven
Background context: Maven is a build automation tool for Java projects. It can handle packaging and deployment of applications, including creating JAR files.

:p How does Maven facilitate the creation and execution of a JAR file?
??x
Maven can automate the process of creating and executing a JAR file by configuring your `pom.xml` (Project Object Model) file. Here is an example configuration:

```xml
<project ...>
    ...
    <packaging>jar</packaging>
    ...
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-jar-plugin</artifactId>
                <version>2.4</version>
                <configuration>
                    <archive>
                        <manifest>
                            <addclasspath>true</addclasspath>
                            <mainClass>${main.class}</mainClass>
                        </manifest>
                    </archive>
                </configuration>
            </plugin>
        </plugins>
    </build>
    ...
</project>
```

In this configuration, the `packaging` element is set to `jar`, and the Maven JAR plugin is configured to add a classpath and specify the main class. This setup allows you to build your project with:

```shell
mvn clean package
```

The resulting JAR file will have the correct manifest headers, including the main-class.
x??

---

#### Maven Assembly Plugin for Runnable JAR
Background context: When you want to package a Java application with its dependencies into a single executable JAR file using Maven, you need more than just the standard packaging plugin. The Maven assembly plugin can be used to create an uber-JAR (a JAR that contains all dependencies) along with a manifest file specifying the main class.

:p What does the Maven assembly plugin do in the context of creating a runnable JAR?
??x
The Maven assembly plugin is configured to package your application and its dependencies into a single JAR file, making it easier to distribute or run. It uses the `jar-with-dependencies` descriptor reference to include all dependencies within the JAR.

To configure this, add the following snippet in the `<build>` section of your `pom.xml`:

```xml
<plugins>
    <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-assembly-plugin</artifactId>
        <version>2.6</version>
        <configuration>
            <descriptorRefs>
                <descriptorRef>jar-with-dependencies</descriptorRef>
            </descriptorRefs>
            <archive>
                <manifest>
                    <addDefaultImplementationEntries>true</addDefaultImplementationEntries>
                    <mainClass>${main.class}</mainClass>
                </manifest>
                <manifestEntries>
                    <Vendor-URL>http://YOURDOMAIN.com/SOME_PATH/</Vendor-URL>
                </manifestEntries>
            </archive>
        </configuration>
    </plugin>
</plugins>
```

The `jar-with-dependencies` descriptor ensures that all dependencies are included in the JAR. The manifest file is configured to add a default implementation entry and specify the main class, along with additional metadata like vendor URL.

x??

---

#### Creating a WAR File for Web Tier Components
Background context: If you have web-tier resources (such as servlets, JSPs, etc.) that need to be packaged together, Maven provides the capability to create a WAR file. This is typically used for deploying applications on a Java EE server.

:p How do you package web-tier components into a WAR file using Maven?
??x
To package your web-tier components into a WAR file, you can use the `maven-war-plugin` in your Maven project. The plugin will handle packaging all necessary resources (such as servlet classes, JSPs, static content) into a WAR archive.

Here is an example configuration for the `maven-war-plugin`:

```xml
<build>
    <plugins>
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-war-plugin</artifactId>
            <version>3.2.3</version>
            <!-- Additional configurations can be added here -->
        </plugin>
    </plugins>
</build>
```

:p What command would you run to package the project into a WAR file?
??x
To package your Maven project into a WAR file, you would use the following command:

```sh
mvn clean install war:war
```

This command performs a `clean` and `install` first (which builds and installs the artifact in the local repository) and then packages it as a WAR.

x??

---

#### Creating a WAR File Using Maven
Background context: To package Java web applications, particularly those using Servlets and JSP pages, one common approach is to use the Maven build tool. The `war` packaging type in Maven allows you to package your application as a Web Application Archive (WAR) file.

:p How do you configure Maven to create a WAR file?
??x
To configure Maven for creating a WAR files, you need to specify `<packaging>war</packaging>` in your `pom.xml` file. This tells Maven that the project is a web application and should be packaged as a WAR.
```xml
<project>
    ...
    <packaging>war</packaging>
    ...
</project>
```
x??

---

#### Directory Structure for Web Application
Background context: The directory structure of a typical Java web application using Servlets includes various directories like `classes`, `lib`, and `WEB-INF` to organize resources such as `.class` files, external libraries, and configuration files.

:p What is the typical directory structure for a Java web application?
??x
The typical directory structure for a Java web application might look like this:

```
Project Root Directory
├── README.asciidoc
├── index.html  - typical web pages
│── signup.jsp  - ditto
├── WEB-INF
    ├── classes - Directory for individual .class files
    ├── lib     - Directory for Jar files needed by app
    └── web.xml - web app Descriptor ("Configuration file")
```
x??

---

#### Maven Project Structure
Background context: When using Maven to build a Java web application, the project structure should reflect the intended use of `mvn package` to create a WAR file. The `src/main/webapp` directory is specifically used for placing web application resources.

:p What does the Maven project structure look like for a web application?
??x
The Maven project structure for a web application typically looks like this:

```
Project Root Directory
├── README.asciidoc
├── pom.xml
└── src
    └── main
        ├── java
        │   └── foo
        │       └── WebTierClass.java
        └── webapp
            ├── WEB-INF
            │   ├── classes
            │   ├── lib
            │   └── web.xml
            ├── index.html
            └── signup.jsp
```
x??

---

#### Packaging with Maven
Background context: The `mvn package` command in Maven compiles the source code, places it into the appropriate directories, and creates a WAR file that can be deployed to a web server.

:p How do you package your project as a WAR using Maven?
??x
To package your project as a WAR using Maven, you use the `mvn package` command. This command will compile the source code, place it into the appropriate directories, and create the WAR file under the `target` directory.
```bash
mvn package
```
x??

---

#### Deployment to Web Server
Background context: Once the WAR file is created, you can deploy it to a web server that supports Java applications. The exact steps for deployment depend on the specific web server being used.

:p How do you deploy a WAR file to a web server?
??x
To deploy a WAR file to a web server, follow these general steps:
1. Copy the generated `*.war` file from the `target` directory in your Maven project.
2. Place the WAR file into the appropriate directory on your web server (e.g., `/webapps` for Tomcat).
3. Restart the web server if necessary to load the new application.

For detailed steps, consult the documentation of the specific web server you are using.
x??

---

