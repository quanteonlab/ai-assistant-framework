# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 73)

**Starting Chapter:** Problem. 17.2 Finding and Using Methods and Fields

---

---
#### Getting a Class Descriptor Using `ClassName.class` Keyword
Background context: In Java, reflection is used to inspect and manipulate classes at runtime. The `Class` class provides various methods for introspection. One common way to obtain a `Class` object for a given class name (or type) is through the use of the `.class` keyword.

:p How do you get a `Class` object using the `ClassName.class` keyword?
??x
The `ClassName.class` keyword can be used to retrieve a `Class` object directly from a class or array type at compile time. This method works for any type that is known at compile time, including primitive types.

```java
System.out.println("Object class: " + Object.class);
System.out.println("String class: " + String.class);
System.out.println("String[] class: " + String[].class);
System.out.println("Calendar class: " + Calendar.class);
System.out.println("Current class: " + ClassKeyword.class);
System.out.println("Class for int: " + int.class);
```
x??

---
#### Getting a Class Descriptor Using `getClass()` Method
Background context: In addition to using the `.class` keyword, you can also get a `Class` object by calling the `getClass()` method on an instance of a class. This method returns the `Class` object representing the runtime class of the object.

:p How do you get a `Class` object for an instance of a class?
??x
You can obtain a `Class` object for an instance of a class by calling the `getClass()` method on that instance.

```java
System.out.println("Sir Robin the Brave" .getClass());
System.out.println(Calendar.getInstance().getClass());
```
x??

---
#### Reflection Magic in JVM
Background context: When downloading bytecode files over the internet, browsers would use Java Applets. These applets were loaded into a running JVM to be executed on the user's desktop. The process of loading and executing these classes involves reflection, which allows runtime inspection and manipulation of classes.

:p How does a browser load an applet’s bytecode file into the running JVM?
??x
A browser would download the applet's bytecode file over the internet and then use Java's ClassLoader mechanism to load this class into the JVM. The ClassLoader is responsible for finding, loading, initializing, and unloading classes at runtime.

```java
// Example of a simple ClassLoader (not actual implementation)
public class SimpleClassLoader extends ClassLoader {
    @Override
    protected Class<?> findClass(String name) throws ClassNotFoundException {
        // Load bytecode from network or file system
        byte[] className = loadByteCodeFromNetworkOrFileSystem(name);
        return defineClass(name, className, 0, className.length);
    }
}
```
x??

---
#### JDK Tools and Reference Books
Background context: The chapter discusses the `javap` tool, which disassembles Java class files to show their internal structure. Additionally, a cross-reference tool is mentioned that can be used for creating detailed reference materials for Java APIs.

:p What are some of the tools discussed in this chapter related to the JDK?
??x
This chapter discusses two main tools:
1. `javap`: A disassembler tool that can show the internal structure of compiled class files.
2. A cross-reference tool: This can be used to create detailed references for Java APIs, potentially leading to the creation of your own reference materials.

```java
// Example usage of javap (not actual implementation)
public void useJavapTool() {
    try {
        String className = "SomeClass";
        // Command line invocation (hypothetical example)
        Process process = Runtime.getRuntime().exec("javap -c SomeClass");
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        String line;
        while ((line = reader.readLine()) != null) {
            System.out.println(line);
        }
    } catch (IOException e) {
        e.printStackTrace();
    }
}
```
x??

---

#### Getting Class Objects Using `Class.forName()`
To obtain a `Class` object for an arbitrary class, you can use `Class.forName(String className)`. This method is useful when the class name is provided as a string at runtime. The `Class` object obtained this way provides access to various reflective capabilities of the specified class.

:p How do you get a `Class` object using reflection?
??x
You can obtain a `Class` object for any class by calling `Class.forName(String className)`. This method is particularly useful when you have the class name as a string at runtime. The `Class` object returned has methods to access fields, methods, and constructors of the given class.

```java
public class Example {
    public static void main(String[] args) throws ClassNotFoundException {
        Class<?> clazz = Class.forName("java.lang.String");
        // Now you can use methods like getMethods() on this clazz object.
    }
}
```
x??

---

#### Listing Constructors, Methods, and Fields Using `Class` Methods
Using the `Class` object obtained via reflection, you can list its constructors, methods, and fields. The key methods are `getConstructors()`, `getMethods()`, and `getFields()`.

:p How do you list all constructors of a given class using reflection?
??x
You can list all constructors of a given class by calling the method `getConstructors()` on the `Class` object obtained through `Class.forName(className)`. This method returns an array of `Constructor` objects representing all public constructors of the specified class.

```java
public class ListMethods {
    public static void main(String[] argv) throws ClassNotFoundException {
        Class<?> clazz = Class.forName(argv[0]);
        Constructor<?>[] cons = clazz.getConstructors();
        printList("Constructors", cons);
    }

    static void printList(String s, Object[] o) {
        System.out.println("*** " + s + " ***");
        for (int i = 0; i < o.length; i++) {
            System.out.println(o[i].toString());
        }
    }
}
```
x??

---

#### Invoking Methods Using Reflection
To invoke a method using reflection, you first need to obtain the `Method` object representing the method. This can be done by calling `getMethod(String name, Class<?>... parameterTypes)` on the `Class` object.

:p How do you find and invoke a method using reflection?
??x
Finding and invoking a method involves several steps:

1. **Find the Method**: Use `getMethod(String name, Class<?>... parameterTypes)` to get the `Method` object representing the desired method.
2. **Invoke the Method**: Call `invoke(Object obj, Object... args)` on the `Method` object with an instance of the class and arguments corresponding to the method's parameters.

Here’s a simplified example:

```java
public class FindField {
    public static void main(String[] unused) throws NoSuchFieldException, IllegalAccessException {
        // Create target object.
        YearHolder o = new YearHolder();
        System.out.println("The value of 'currentYear' is: " + intFieldValue(o, "currentYear"));
    }

    int intFieldValue(Object o, String name) throws NoSuchFieldException, IllegalAccessException {
        Class<?> c = o.getClass();
        Field fld = c.getField(name);
        int value = fld.getInt(o);
        return value;
    }
}

class YearHolder {
    public int currentYear = Calendar.getInstance().get(Calendar.YEAR);
}
```

You can extend this to invoke methods as well:

```java
public class GetAndInvokeMethod {
    static class X {
        public void work(int i, String s) {
            System.out.printf("Called: i= %d, s= %s%n", i, s);
        }
    }

    public static void main(String[] argv) throws Exception {
        Class<?> clX = X.class;
        Class<?>[] argTypes = {int.class, String.class};
        Method worker = clX.getMethod("work", argTypes);
        Object[] theData = {42, "Chocolate Chips"};
        worker.invoke(new X(), theData);
    }
}
```
x??

---

#### Difference Between `getField()` and `getDeclaredField()`
`Class.getField(String name)` returns a public field with the specified name. On the other hand, `Class.getDeclaredField(String name)` returns the specified declared field, regardless of its accessibility.

:p How do you find a private or protected field using reflection?
??x
To find a private or protected field, use `getDeclaredField(String name)` instead of `getField(String name)`. This method allows you to access fields that are not accessible from outside the class due to their visibility modifiers.

```java
public class GetPrivateField {
    public static void main(String[] args) throws NoSuchFieldException, IllegalAccessException {
        Class<?> clazz = YearHolder.class;
        Field field = clazz.getDeclaredField("currentYear");
        // Make the field accessible.
        field.setAccessible(true);
        int value = field.getInt(new YearHolder());
        System.out.println("The current year is: " + value);
    }
}

class YearHolder {
    private int currentYear = Calendar.getInstance().get(Calendar.YEAR);
}
```
x??

---

#### Using `Constructor` Objects to Create Instances
You can use the `Constructor` object obtained via reflection to create instances of a class. The key method here is `newInstance(Object... initargs)`.

:p How do you use `Constructor` objects to instantiate a class?
??x
To use a `Constructor` object to create an instance of a class, call its `newInstance(Object... initargs)` method. This method takes the arguments that match the constructor's parameters and returns an instance of the class.

```java
public class InstantiateClass {
    public static void main(String[] args) throws Exception {
        Class<?> clazz = String.class;
        Constructor<?> cons = clazz.getConstructor();
        Object obj = cons.newInstance(); // Create a new string object.
        System.out.println(obj);
    }
}
```
x??

---

---
#### Accessing Private Fields and Methods via Reflection
Background context: In Java, using reflection allows you to inspect and modify the properties of classes at runtime. This includes accessing private fields and methods, which are not normally accessible due to encapsulation principles. The `java.lang.reflect.Field` and `java.lang.reflect.Method` classes provide a way to do so by invoking the `setAccessible(true)` method.

:p How can you access private fields or methods using reflection?
??x
To access private fields or methods using reflection, you need to use the `setAccessible(true)` method on the corresponding Field or Method object. This bypasses the access control checks that are enforced at runtime by the JVM.
```java
Field field = X.class.getDeclaredField("p");
field.setAccessible(true); // Allow access to the private field
int value = (int) field.get(new X()); // Get the value of the private field p
```
x??

---
#### Using MethodHandle for Reflection
Background context: Java introduced `MethodHandle` as a way to simplify and generalize method invocation through reflection. It was designed to provide more direct access to method implementations, potentially making the code faster than using the Reflection API.

:p Why might you use `MethodHandle` instead of the standard Reflection API?
??x
`MethodHandle` is intended to be used for more efficient and generalized method invocations compared to the traditional Reflection API. However, in practice, it has not shown significant performance benefits over the Reflection API, so its usage remains limited.

```java
// Example pseudo-code using MethodHandle (not actual syntax)
try {
    MethodHandles.Lookup lookup = MethodHandles.lookup();
    MethodHandle mh = lookup.findVirtual(X.class, "method", methodType(int.class));
    int result = (int) mh.invokeExact(new X(), 42);
} catch (Throwable e) {
    // Handle exceptions
}
```
x??

---
#### The `Integer.TYPE` Constant
Background context: In Java, each primitive type has a corresponding wrapper class. For example, the `Integer` class wraps the `int` primitive type. One of the constants in the `Integer` class is `TYPE`, which refers to the Class object for the `int` primitive.

:p How can you use `Integer.TYPE`?
??x
You can use `Integer.TYPE` to refer to the `Class` object representing the `int` primitive type, allowing you to interact with it through reflection or generics. For example:
```java
Class<?> intType = Integer.TYPE;
```
x??

---
#### The Reflection API and Security Manager
Background context: When using reflection in Java, especially for accessing private members of classes, you may need to bypass security restrictions imposed by the `SecurityManager`. This can be done by setting the accessibility of fields or methods to true.

:p How does the `SecurityManager` affect reflection?
??x
The `SecurityManager` enforces certain security checks that restrict access to private fields and methods. If you want to use reflection to access these members, you need to ensure that the `SecurityManager` allows such actions by setting the accessibility of the Field or Method to true.

```java
Field field = X.class.getDeclaredField("p");
field.setAccessible(true); // Bypass security restrictions for private field p
```
x??

---

