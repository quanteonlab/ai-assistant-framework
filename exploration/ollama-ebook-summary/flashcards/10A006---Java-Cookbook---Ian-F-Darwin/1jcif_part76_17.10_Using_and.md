# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 76)

**Starting Chapter:** 17.10 Using and Defining Annotations. Problem. Solution. Discussion

---

#### Annotations Overview
Annotations provide additional information beyond what source code conveys. They can be used both at compile-time and runtime.
:p What is an annotation, and what are its primary uses?
??x
An annotation is a type of metadata that can be added to your Java code using the `@` symbol followed by the annotation name. Annotations help in providing additional information about classes, methods, fields, etc., which can be used at compile-time or runtime. For example, they can be used for overriding checks, persistence mapping, and more.

Annotations can be applied to various elements such as:
- Classes
- Methods
- Fields
- Parameters
- Local variables (since Java 8)

Here's an example of how you might apply an annotation:

```java
@MyAnnotation
public class MyClass {
    // class body
}
```

?: Explain the syntax and usage of annotations in detail.
??x
Annotations are defined using the `@interface` keyword. Here is a simple example of defining your own annotation:

```java
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
@interface MyAnnotation {
    String value();
}
```

This defines a custom `MyAnnotation` that can be applied to methods. The `@Retention` and `@Target` meta-annotations specify where the annotation can be used.

For applying an existing annotation, such as `@Override`, you simply place it before the element you want to annotate:

```java
public class MyClass {
    @Override
    public boolean equals(Object obj) { // Correct version with @Override
        // implementation
    }
}
```

?: How does the `@Override` annotation work in practice?
??x
The `@Override` annotation is used to indicate that a method or constructor is intended to override a method or constructor declared in a superclass. If the overridden method does not exist, the compiler will generate an error.

Here’s how it works:

```java
public class MyClass {
    @Override // This annotation tells the compiler this method should override a method from a superclass.
    public boolean equals(Object obj) {
        // implementation
    }
}
```

If you try to override a method incorrectly, such as in Example 17-14:

```java
public class MyClass {
    public boolean equals(MyClass object2) { // Incorrect: should be 'Object'
        // implementation
    }
}
```

The `@Override` annotation would catch this error at compile-time.

?: What is a common benefit of using the `@Override` annotation?
??x
A key benefit of using the `@Override` annotation is that it ensures methods are correctly overridden from superclasses or interfaces. If you accidentally try to override a method with an incorrect signature, the compiler will catch this error and prevent runtime issues.

For example:

```java
public class MyClass {
    @Override // This helps in catching potential errors at compile-time.
    public boolean equals(Object obj) { // Correct version with @Override
        // implementation
    }
}
```

Without `@Override`, such an error might go unnoticed until the code is run and issues occur due to incorrect method signatures.

?: How do annotations assist during runtime?
??x
Annotations can be used for metadata that is processed at runtime. For instance, in Java Persistence API (JPA), you use annotations like `@Entity` and `@Id` to mark entity classes for database persistence.

Here’s an example of a JPA entity class:

```java
import javax.persistence.Entity;
import javax.persistence.Id;

@Entity // Marks this as a JPA entity
public class MyEntity {
    @Id // Marks the field as the primary key
    private Long id;

    // other fields and methods
}
```

At runtime, these annotations can be used by frameworks like Hibernate to manage object persistence.

?: What is an example of using `@Override` in a real-world scenario?
??x
An example of using `@Override` in a real-world scenario would be ensuring that the overridden method correctly matches the signature from the superclass or interface. For instance, if you override the `equals()` and `hashCode()` methods as part of the `Object` class's contract:

```java
public class MyClass {
    @Override // Ensures this method overrides Object.equals()
    public boolean equals(Object obj) {
        // implementation
    }

    @Override // Ensures this method overrides Object.hashCode()
    public int hashCode() {
        // implementation
    }
}
```

?: How can annotations help in maintaining code?
??x
Annotations can help in maintaining code by ensuring that changes to superclass methods or interfaces are propagated correctly. For example, if a method is removed from a superclass and subclasses still try to override it with the `@Override` annotation, the compiler will generate an error.

This helps in identifying dead code and ensuring consistency across your codebase:

```java
public class SubClass extends SuperClass {
    @Override // If this method no longer exists in SuperClass, this would cause a compile-time error.
    public void oldMethod() {
        // implementation
    }
}
```

By using `@Override`, you can avoid potential runtime issues and keep your codebase clean.

#### JPA Annotations Overview
Background context: JPA (Java Persistence API) is a Java specification that defines how to create, read, update, and delete persistent objects and their mappings to database tables. Annotations are used extensively in JPA for metadata description without XML configurations.

:p What are the key annotations used in the provided `Person` class?
??x
The key annotations used include:
- `@Entity`: Indicates that this class is a persistence entity.
- `@Id`: Marks the field as the primary identifier of an entity.
- `@GeneratedValue`: Specifies how to generate the primary key values for newly created objects.

Explanation: These annotations are crucial for JPA to understand which fields should be persisted, how they should be managed, and how their values should be generated or assigned.
```java
@Entity
public class Person {
    // Fields and methods here
}
@Id
@GeneratedValue(strategy = GenerationType.AUTO, generator="my_poid_gen")
private int id;
```
x??

---

#### Column Annotation Usage
Background context: The `@Column` annotation is used to control the column mapping of a field in a database table. It allows specifying the name of the column or other properties related to the database schema.

:p How does the `@Column` annotation work in the provided `Person` class?
??x
The `@Column` annotation works by allowing you to specify the name of the column that corresponds to a field in the database. In this case, it maps the `lastName` property to a "surname" column.

Explanation: The `name` attribute within `@Column` is used to override the default naming convention based on the Java bean property name.
```java
@Column(name="surname")
public String getLastName () { ... }
```
x??

---

#### Custom Annotations in Java
Background context: Annotations are a way of adding metadata or information about code elements. While JPA annotations provide built-in features, you can also define your own custom annotations to add more flexibility and semantics to your application.

:p How do you define a simple custom annotation in Java?
??x
To define a simple custom annotation in Java, use the `@interface` keyword followed by the name of the annotation. The example provided shows a basic definition without any elements or methods.

Explanation: Custom annotations can be used to mark classes, fields, methods, parameters, and local variables.
```java
package lang;

public @interface MyToyAnnotation { }
```
x??

---

#### Entity Class Example
Background context: An entity class in JPA is the main entry point for object-relational mapping. It represents a database table and its corresponding rows.

:p What are the key points of the `Person` entity class provided?
??x
The key points of the `Person` entity class include:
- The `@Entity` annotation, which marks the class as an entity.
- The `@Id` and `@GeneratedValue` annotations to manage the primary key.
- The `getFullName()` method that is marked with `@Transient`, indicating it should not be persisted.

Explanation: This class defines a person with fields for ID, first name, and last name. The constructor and methods provide a basic implementation of these properties.

```java
@Entity
public class Person {
    int id;
    protected String firstName;
    protected String lastName;

    public Person() { ... }
    public Person(String firstName, String lastName) { ... }

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO, generator="my_poid_gen")
    public int getId() { ... }

    // Other fields and methods here
}
```
x??

---

#### Using `@Transient` Annotation
Background context: The `@Transient` annotation is used to indicate that a field or method should not be stored in the database.

:p What does the `@Transient` annotation do in the provided example?
??x
The `@Transient` annotation prevents the `getFullName()` method from being persisted to the database. This means that while you can use this method in your application logic, its value will not be stored or retrieved from the database.

Explanation: This is useful when you need a computed field that should not affect the state of the entity.
```java
@Transient
public String getFullName () { ... }
```
x??

---

---
#### Annotation Interface Overview
Annotations are represented as interfaces that extend `java.lang.annotation.Annotation`. These annotations can have attributes and methods defined within them. The `@interface` keyword is used to define an annotation type.

:p What does an annotation interface look like in Java?
??x
An annotation interface typically extends the `java.lang.annotation.Annotation` class and may include method declarations representing attribute values.

```java
public @interface MyAnnotation {
    String value() default "";
}
```
x??

---
#### @Target Annotation
The `@Target` annotation is used to specify where an annotation type can be applied. It takes a single parameter that specifies the element types on which the annotation is valid.

:p What does the `@Target` annotation do?
??x
The `@Target` annotation defines the target elements for an annotation, such as methods, fields, classes, interfaces, etc. This helps the compiler to enforce where annotations can be applied.

```java
@Target(ElementType.TYPE)
public @interface MyAnnotation {
}
```
x??

---
#### Retention Policy
The `@Retention` annotation is used to define how long an annotation should remain valid throughout the life cycle of a program. It can have different values such as `SOURCE`, `CLASS`, or `RUNTIME`.

:p What does the `@Retention` annotation do?
??x
The `@Retention` annotation specifies the retention policy for an annotation, indicating when and where the annotation information is available during the compilation and runtime phases.

```java
@Retention(RetentionPolicy.RUNTIME)
public @interface MyAnnotation {
}
```
x??

---
#### Annotation with Attributes
Annotations can have attributes that are defined as methods. These attributes can be used within the annotated code to provide specific values or behaviors.

:p How do you define attributes in an annotation?
??x
Attributes in an annotation are defined as method declarations inside the annotation interface. The attribute values can be specified when the annotation is applied to a target element.

```java
public @interface AnnotationDemo {
    boolean fancy() default false;
    int order() default 42;
}
```
x??

---
#### Applying Annotations to Classes
Annotations can be used on various elements such as classes, interfaces, and methods. The `@Target` annotation specifies the applicable target element types.

:p How do you apply an annotation to a class?
??x
You apply an annotation to a class by prefixing it with the `@` symbol before the class definition. You can also specify attribute values if needed.

```java
@AnnotationDemo(fancy = true)
class FancyClassJustToShowAnnotation {
    // Class body
}
```
x??

---
#### Reflection API for Annotations
The Java Reflection API allows you to inspect annotations at runtime, providing detailed information about the classes, methods, and fields of your program.

:p How do you use reflection to get annotations?
??x
You can use the `getAnnotations()` method of a class or interface to retrieve all its annotations. Then, you can check if an annotation instance matches specific types and cast it accordingly.

```java
Class<?> c = FancyClassJustToShowAnnotation.class;
for (Annotation a : c.getAnnotations()) {
    if (a instanceof AnnotationDemo) {
        AnnotationDemo ad = (AnnotationDemo)a;
        System.out.println("\t" + a);
    }
}
```
x??

---

#### Finding Classes Annotated with a Specific Annotation
Background context: This concept involves using Java Reflection to discover classes annotated with a specific annotation. Annotations can be applied at various levels (class, method, field, parameter) and are used for metadata purposes such as indicating plug-in-like functionality.

:p What is the purpose of `findAnnotatedClasses` in the `PluginsViaAnnotations` class?
??x
The purpose of `findAnnotatedClasses` is to find all classes within a specified package that have a given annotation. This method uses Java Reflection to iterate through the list of class names, load each class, and check if it has the desired annotation.

Code example:
```java
public static List<Class<?>> findAnnotatedClasses(String packageName,
                                                 Class<? extends Annotation> annotationClass) throws Exception {
    List<Class<?>> ret = new ArrayList<>();
    String[] clazzNames = ClassesInPackage.getPackageContent(packageName);
    for (String clazzName : clazzNames) {
        if (clazzName.endsWith(".class")) {
            continue;
        }
        clazzName = clazzName.replace('/', '.').replace(".class", "");
        Class<?> c = null;
        try {
            c = Class.forName(clazzName);
        } catch (ClassNotFoundException ex) {
            System.err.println("Weird: class " + clazzName + 
                               " reported in package but gave CNFE: " + ex);
            continue;
        }
        if (c.isAnnotationPresent(annotationClass) && !ret.contains(c)) {
            ret.add(c);
        }
    }
    return ret;
}
```

x??

---
#### Method-Level Annotations
Background context: This concept extends the previous one by handling method annotations, which are often used for lifecycle methods in frameworks like Java EE.

:p How does the flow change when dealing with both class and method-level annotations?
??x
When dealing with both class and method-level annotations, you first check if a class is annotated. If it is, then you proceed to inspect its methods to see if any of them are annotated with specific method-specific annotations.

Code example:
```java
public static void processClassesWithAnnotations(String packageName,
                                                 Class<? extends Annotation> classAnnotationClass,
                                                 Class<? extends Annotation> methodAnnotationClass) throws Exception {
    List<Class<?>> classes = findAnnotatedClasses(packageName, classAnnotationClass);
    for (Class<?> c : classes) {
        Method[] methods = c.getDeclaredMethods();
        for (Method m : methods) {
            if (m.isAnnotationPresent(methodAnnotationClass)) {
                // Perform some action on the method
                System.out.println("Method " + m.getName() + " is annotated with " + methodAnnotationClass);
            }
        }
    }
}
```

x??

---
#### Handling Class-Level Annotations
Background context: This concept focuses on identifying and processing classes that are annotated at the class level, which can be used for defining plugins or add-in functionality.

:p How would you implement a mechanism to discover classes with a specific class-level annotation?
??x
To implement this, you need to use Java Reflection to scan through all the classes in a given package and check if they have the specified class-level annotation. If found, these classes can be stored for later processing or execution.

Code example:
```java
public static List<Class<?>> findAnnotatedClasses(String packageName,
                                                 Class<? extends Annotation> annotationClass) throws Exception {
    List<Class<?>> ret = new ArrayList<>();
    String[] clazzNames = ClassesInPackage.getPackageContent(packageName);
    for (String clazzName : clazzNames) {
        if (clazzName.endsWith(".class")) {
            continue;
        }
        clazzName = clazzName.replace('/', '.').replace(".class", "");
        Class<?> c = null;
        try {
            c = Class.forName(clazzName);
        } catch (ClassNotFoundException ex) {
            System.err.println("Weird: class " + clazzName +
                               " reported in package but gave CNFE: " + ex);
            continue;
        }
        if (c.isAnnotationPresent(annotationClass)) {
            ret.add(c);
        }
    }
    return ret;
}
```

x??

---
#### Using Annotations for Plug-in Functionality
Background context: This concept illustrates how to use annotations to identify and process classes that should be treated as plugins or add-ins. This can be useful in frameworks where certain classes need to be dynamically discovered and executed.

:p How do you differentiate between a class-level annotation and a method-level annotation?
??x
To differentiate between class-level and method-level annotations, you first check if the class itself is annotated with the desired class-level annotation. If it is, then you proceed to inspect its methods for any method-specific annotations. This dual-checking ensures that both classes and their contained methods can be processed according to the defined metadata.

Code example:
```java
public static void processClassesWithAnnotations(String packageName,
                                                 Class<? extends Annotation> classAnnotationClass,
                                                 Class<? extends Annotation> methodAnnotationClass) throws Exception {
    List<Class<?>> classes = findAnnotatedClasses(packageName, classAnnotationClass);
    for (Class<?> c : classes) {
        if (c.isAnnotationPresent(classAnnotationClass)) { // Check class-level annotation
            System.out.println("Class " + c.getName() + 
                               " is annotated with " + classAnnotationClass);
            
            Method[] methods = c.getDeclaredMethods();
            for (Method m : methods) {
                if (m.isAnnotationPresent(methodAnnotationClass)) { // Check method-level annotation
                    System.out.println("\t" + m.getName() + " is annotated with " + methodAnnotationClass);
                }
            }
        } else {
            System.out.println("\tSomebody else's annotation: " + classAnnotationClass.getAnnotation(classAnnotationClass));
        }
    }
}
```

x??

---

---
#### Finding Classes with Annotated Methods
Background context: This concept involves searching through a package to find classes that have methods annotated with a specific annotation. The provided code snippet illustrates how to implement this functionality.

:p How does the method `findClassesWithAnnotatedMethods` work?
??x
The `findClassesWithAnnotatedMethods` method searches for classes in a given package that contain at least one method annotated with a specified annotation class. It uses reflection to load and inspect each class within the package, checking methods for the presence of the required annotation.

```java
public static List<Class<?>> findClassesWithAnnotatedMethods(String packageName,
                                                              Class<? extends Annotation> methodAnnotationClass)
throws Exception {
    List<Class<?>> ret = new ArrayList<>();
    String[] clazzNames = ClassesInPackage.getPackageContent(packageName);
    for (String clazzName : clazzNames) {
        if (clazzName.endsWith(".class")) {
            continue;
        }
        clazzName = clazzName.replace('/', '.').replace(".class", "");
        Class<?> c = null;
        try {
            c = Class.forName(clazzName);
        } catch (ClassNotFoundException ex) {
            System.err.println("Weird: class " + clazzName + 
                               " reported in package but gave CNFE: " + ex);
            continue;
        }
        for (Method m : c.getDeclaredMethods()) {
            if (m.isAnnotationPresent(methodAnnotationClass) && !ret.contains(c)) {
                ret.add(c);
            }
        }
    }
    return ret;
}
```

x??
---
#### Cross-Referencing Java Classes
Background context: This concept involves creating a cross-reference tool that can list the fields and methods of classes in a JAR file. The provided code snippet illustrates how to use reflection to achieve this.

:p What is the purpose of the `CrossRef` class?
??x
The `CrossRef` class serves as a cross-referencing tool for Java classes. It extends an `APIFormatter` class and uses reflection to load and list the fields and methods of each class found in JAR files or specified through command-line arguments.

```java
public class CrossRef extends APIFormatter {
    public static void main(String[] argv) throws IOException {
        CrossRef xref = new CrossRef();
        xref.doArgs(argv);
    }

    protected void doClass(Class<?> c) {
        startClass(c);
        try {
            Field[] fields = c.getDeclaredFields();
            Arrays.sort(fields, (o1, o2) -> o1.getName().compareTo(o2.getName()));
            for (Field field : fields) {
                if (!Modifier.isPrivate(field.getModifiers())) {
                    putField(field, c);
                }
            }

            Method methods[] = c.getDeclaredMethods();
            Arrays.sort(methods, (o1, o2) -> o1.getName().compareTo(o2.getName()));
            for (Method method : methods) {
                if (!Modifier.isPrivate(method.getModifiers())) {
                    putMethod(method, c);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        endClass();
    }

    protected void putField(Field fld, Class<?> c) {
        println(fld.getName() + " field " + c.getName());
    }

    protected void putMethod(Method method, Class<?> c) {
        String methName = method.getName();
        println(methName + " method " + c.getName());
    }
}
```

x??
---

