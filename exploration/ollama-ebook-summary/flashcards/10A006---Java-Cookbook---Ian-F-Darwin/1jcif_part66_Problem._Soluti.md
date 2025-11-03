# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 66)

**Starting Chapter:** Problem. Solution. Discussion

---

#### Overriding vs. Overloading Concepts
Background context explaining the difference between overriding and overloading in Java. The key points are that **overriding** involves a method with the same name, parameters, and return type in a subclass, while **overloading** involves methods with the same name but different parameters (number, type, or order).

If applicable, add code examples with explanations:
```java
// Example of Overriding
class Top {
    public void myMethod(Object o) {
        // Method body
    }
}

class Bottom extends Top {
    @Override
    public void myMethod(String s) {  // This is a mistake, should be Object to override.
        // Do something here...
    }
}
```
:p How does the method in `Bottom` class relate to overriding and overloading?
??x
The `myMethod` in `Bottom` class attempts to override a method from `Top`, but it incorrectly uses `String` instead of `Object`. This results in a compile-time error, indicating that the method is not an override.

To correctly override, you should use `public void myMethod(Object o)` as follows:
```java
class Bottom extends Top {
    @Override
    public void myMethod(Object s) {  // Corrected to match the signature of the parent class.
        // Do something here...
    }
}
```
x??

---
#### JavaBean Requirements and Implementation
Background context on what a JavaBean is, including naming conventions for properties. The main points are that all public properties should be accessible via get/set accessor methods. Examples include `getText()` and `setText(String)`.

Example of set/get method pair:
```java
public class LabelText extends JPanel implements java.io.Serializable {
    // ...
    public String getText() {  // Getter
        return theTextField.getText();
    }
    
    public void setText(String text) {  // Setter
        theTextField.setText(text);
    }
}
```
:p What are the minimum requirements for a JavaBean according to the passage?
??x
The minimum requirements for a JavaBean include:
- A no-argument constructor.
- Use of set/get paradigm for public methods.
- Implementation of `java.io.Serializable` (though not enforced by all containers).
- Packaging into a JAR file, if needed.

For example, `LabelText` class includes the following to be usable as a JavaBean:
```java
public LabelText() {  // No-argument constructor
    this("(LabelText)", 12);
}

public String getText() {
    return theTextField.getText();
}

public void setText(String text) {
    theTextField.setText(text);
}
```
x??

---
#### Code Example of `LabelText` Class
Background context on a sample JavaBean, `LabelText`, that combines a label and a text field.

Example class:
```java
// package com.darwinsys.swingui;
public class LabelText extends JPanel implements java.io.Serializable {
    private static final long serialVersionUID = -8343040707105763298L;

    // Fields and methods are provided here.

    public LabelText() {  // No-argument constructor
        this("(LabelText)", 12);
    }

    public LabelText(String label) {
        this(label, 12);
    }

    public LabelText(String label, int numChars) {
        this(label, numChars, null);
    }

    public LabelText(String label, int numChars, JComponent extra) {
        super();
        setLayout(new BoxLayout(this, BoxLayout.X_AXIS));
        theLabel = new JLabel(label);
        add(theLabel);
        theTextField = new JTextField(numChars);
        add(theTextField);
        if (extra != null) {  // Optional third component
            add(extra);
        }
    }

    public int getLabelAlignment() {
        return theLabel.getHorizontalAlignment();
    }

    public void setLabelAlignment(int align) {
        theLabel.setHorizontalAlignment(align);
    }

    public String getText() {
        return theTextField.getText();
    }

    public void setText(String text) {
        theTextField.setText(text);
    }

    public String getLabel() {
        return theLabel.getText();
    }

    public void setLabel(String text) {
        theLabel.setText(text);
    }
}
```
:p What is the purpose of `LabelText` class?
??x
The `LabelText` class serves as a JavaBean that combines a label and a one-line text field into a single component, simplifying GUI applications. It uses the set/get design pattern for its properties and implements necessary methods to be a JavaBean.

For example, it includes:
```java
public String getText() {  // Getter
    return theTextField.getText();
}

public void setText(String text) {  // Setter
    theTextField.setText(text);
}
```
x??

---

#### Setting Font for Components
Background context: In Java Swing, setting a custom font for components such as labels and text fields requires handling potential issues where the superclass's `setFont()` method might be called prematurely. This can happen during component creation or when applying a look and feel that calls `installColorsAndFont`.
:p How does the provided code handle setting the font for components?
??x
The code checks if the label and text field are not null before calling `setFont()`. If they are null, it means these components haven't been created yet, so no attempt is made to set their fonts. This ensures that the font can only be set once the components have been properly initialized.
```java
public void setFont(Font f) {
    if (theLabel == null) 
        theLabel.setFont(f);
    if (theTextField == null) 
        theTextField.setFont(f);
}
```
x??

---
#### Adding ActionListener for Textfield
Background context: An `ActionListener` can be added to a text field component in Java Swing. This listener will receive notifications when an action event occurs, such as pressing Enter or clicking a button.
:p How does the provided code add an `ActionListener` to a text field?
??x
The provided code uses the `addActionListener()` method of the `JTextField` class to attach the specified `ActionListener` object. This ensures that the listener will be notified whenever an action event occurs in the text field.
```java
public void addActionListener (ActionListener l) {
    theTextField.addActionListener(l);
}
```
x??

---
#### Removing ActionListener from Textfield
Background context: Similarly, it's important to clean up resources by removing unnecessary `ActionListeners` when they are no longer needed. This prevents memory leaks and ensures that only relevant listeners remain attached.
:p How does the provided code remove an `ActionListener` from a text field?
??x
The code uses the `removeActionListener()` method of the `JTextField` class to detach the specified `ActionListener` object. This detaches the listener, preventing it from receiving further action events in the text field.
```java
public void removeActionListener (ActionListener l) {
    theTextField.removeActionListener(l);
}
```
x??

---
#### Creating a JAR File with jar Tool
Background context: Java’s standard tool for creating archives is the `jar` command. These archives can contain multiple files and are used to package classes, resources, or both into a single file.
:p How do you create a JAR archive using the `jar` tool?
??x
You can use the `jar` command with various options such as `-c` for creating an archive, `-t` for generating a manifest table of contents, and `-x` for extracting files from an existing archive. The most common usage is to create a JAR file.
```bash
jar cvf myarchive.jar path/to/your/classes/*.class
```
This command creates (`c`) a `myarchive.jar` file with the specified classes.

To include additional resources:
```bash
jar cvf myarchive.jar -C path/to/resources .
```
x??

---

---
#### JAR File Naming and Manifest File
JAR (Java Archive) files are used to package Java classes and resources into a single file for distribution. The naming convention often includes the package name, but this is optional. A manifest file (.mf) can be included within the JAR to store metadata about the archive.

:p How do you create a JAR file with all subdirectories included?
??x
You can use the `jar` command in the terminal with specific options to include all files and directories from the current directory. The syntax is as follows:
```sh
jar cvf output.jar .
```
Here, `cvf` stands for creating (`c`), verifying file headers (`v`), and specifying a manifest file or including one (if none provided). The dot (`.`) at the end indicates the current directory.

x??

---
#### Manifest File in JAR Archives
Manifest files are crucial in true JAR archives as they list the contents of the archive along with their attributes. These attributes can be required by applications and provide additional information such as names, values, or custom attributes.

:p What is a manifest file used for in JARs?
??x
A manifest file in JARs serves to specify metadata about the archive's contents. It can contain various attributes that might include details like classpath entries, main program specifications, and other application-specific information. Attributes are typically added using headers formatted as `name : value`.

Example of a manifest file content:
```
Main-Class: com.darwinsys.util.Main
Class-Path: lib/someLibrary.jar
MySillyAttribute: true
MySillynessLevel: high
```

x??

---
#### Running Programs Directly from JAR Files
To run programs directly from JAR files, you need a manifest file that includes an `Main-Class` attribute. This tells the JVM which class and method to start when running the JAR.

:p How do you set up a JAR file for direct execution of a main program?
??x
For direct execution of a Java program via a JAR file, ensure your manifest file contains the necessary information including `Main-Class`. Use the `-m` option with the `jar` command to include this manifest. Here’s an example:
```sh
jar -cvfm output.jar manifest.stub .
```
Here, `output.jar` is the name of the generated JAR file, and `manifest.stub` contains the metadata including the main class information.

x??

---
#### Using Maven for JAR Creation
Maven simplifies the process of creating JAR files by automatically packaging your source code into a JAR with appropriate configurations. It handles dependencies, manifest entries, and other build artifacts.

:p How does Maven facilitate the creation of a JAR file?
??x
Maven streamlines the creation of a JAR file through its `package` goal in the lifecycle. Simply running `mvn package` will compile your code, include necessary resources, and create a JAR file within the target directory. This process is automated and integrates seamlessly with other Maven plugins.

Example command:
```sh
mvn package
```
This command triggers a series of build phases including compiling source files, handling dependencies, and packaging everything into a JAR located in `target/`.

x??

---

