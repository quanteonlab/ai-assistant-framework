# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 44)

**Starting Chapter:** 10.6 Scanning Input with the Scanner Class. Problem. Solution. Discussion

---

#### Scanner Class Overview
Background context: The `Scanner` class is a part of Java's standard library and allows for flexible token-based input scanning. It can read various types such as integers, doubles, strings, etc., based on specified delimiters or regular expressions.

If applicable, add code examples with explanations.
:p What is the `Scanner` class used for?
??x
The `Scanner` class is primarily used to parse and extract tokens from an input source. It can read different data types like integers, doubles, strings, etc., by using specific methods such as `nextInt()`, `nextDouble()`, or `next()`.

Example code:
```java
String sampleDate = "25 Dec 1988";
try (Scanner sDate = new Scanner(sampleDate)) {
    int dayOfMonth = sDate.nextInt();
    String month = sDate.next();
    int year = sDate.nextInt();
    System.out.printf("%d-%s %02d\n", year, month, dayOfMonth);
}
```
x??

---

#### Scanner Method Usage
Background context: The `Scanner` class provides various methods to read input tokens of different types. These include specific data type methods like `nextInt()`, `nextDouble()` and more generic methods for strings or regular expressions.

:p How can you use the `hasNext()` method in a `Scanner`?
??x
The `hasNext()` method checks if there is another token to be read from the input source. It returns true if the next token matches the specified type or pattern, allowing the program to decide whether to proceed with reading the next token.

Example code:
```java
Scanner sDate = new Scanner("25 Dec 1988");
if (sDate.hasNextInt()) {
    int dayOfMonth = sDate.nextInt();
    // Proceed with processing the day of month
}
```
x??

---

#### Stack Implementation in Calculator
Background context: The provided code snippet demonstrates a simple calculator using a stack to manage operands. The `Stack<Double>` is used to store numbers, and arithmetic operations are performed based on the tokens read from the input.

:p What does the `pop()` method do in this implementation?
??x
The `pop()` method removes and returns the top element of the stack. In this context, it is used to retrieve the last number that was pushed onto the stack when performing calculations or checking for operands.

Example code:
```java
Stack<Double> s = new Stack<>();
// Example usage
double value = s.pop();
```
x??

---

#### Clearing the Stack
Background context: The `clearStack()` method is used to remove all elements from the stack. This is useful when you need to reset the state of the calculator or prepare for a new set of operations.

:p How does the `clearStack()` method work?
??x
The `clearStack()` method clears all elements from the stack, effectively emptying it and resetting its state. In this context, it allows the calculator to start afresh without retaining any previous calculations.

Example code:
```java
public void clearStack() {
    s.removeAllElements();
}
```
x??

---

#### Using hasNextDouble()
Background context: The `hasNextDouble()` method checks if there is a double value token available in the input. It returns true if the next token can be converted to a double, allowing for conditional logic based on whether such a token exists.

:p How does `hasNextDouble()` work?
??x
The `hasNextDouble()` method checks if the next token from the scanner can be parsed into a double value. If it can, the method returns true; otherwise, it returns false.

Example code:
```java
Scanner scan = new Scanner("25 34.67");
if (scan.hasNextDouble()) {
    double value = scan.nextDouble();
    // Proceed with processing the double value
}
```
x??

---

#### Reading from a File Using Scanner
Background context: The `SimpleCalcScanner` class demonstrates reading input from a file using the `Scanner` class. It constructs a `Scanner` from a `FileReader` to read tokens from a file, allowing for operations like arithmetic calculations on the input data.

:p How does the `SimpleCalcScanner` constructor work?
??x
The `SimpleCalcScanner` constructor can be initialized with a file name or an open `Reader`. It constructs a `Scanner` to read tokens from the specified source. The `doCalc()` method uses the scanner to process arithmetic operations based on the input.

Example code:
```java
public SimpleCalcScanner(String fileName) throws IOException {
    this(new FileReader(fileName));
}

// Example usage in doCalc()
public void doCalc() throws IOException {
    while (scan.hasNext()) {
        if (scan.hasNextDouble()) {
            push(scan.nextDouble());
        } else {
            String token = scan.next();
            switch(token) {
                case "+":
                    // Perform addition
                    break;
                default:
                    // Handle other tokens
            }
        }
    }
}
```
x??

---

---
#### Push and Pop Operations on a Stack
Background context: The provided Java code snippet demonstrates how to perform push, pop, and peek operations on a stack for evaluating arithmetic expressions. These operations are fundamental in implementing a simple expression evaluator using stacks.

:p What do the `push` and `pop` methods do in the given code?
??x
The `push` method adds an element (in this case, a double value) to the top of the stack. The `pop` method removes and returns the top element from the stack. These operations are crucial for implementing an expression evaluator where operands and intermediate results are stored temporarily.

```java
void push(double val) {
    s.push(Double.valueOf(val));
}

double pop() {
    return ((Double)s.pop()).doubleValue();
}
```
x??

---
#### Peek Operation on a Stack
Background context: The `peek` method in the provided code retrieves the top element of the stack without removing it. This is useful for evaluating operations where you need to see the current top value but do not want to alter the stack.

:p What does the `peek` method return?
??x
The `peek` method returns the top element of the stack, allowing you to inspect this value without removing it from the stack.

```java
double peek() {
    return ((Double)s.peek()).doubleValue();
}
```
x??

---
#### Clearing a Stack
Background context: The `clearStack` method is used to clear all elements from the stack. This can be useful in scenarios where multiple expressions need to be evaluated, and you want to start with an empty stack.

:p How does the `clearStack` method work?
??x
The `clearStack` method clears all elements from the stack by removing them entirely. In this implementation, it uses `s.removeAllElements()` which removes all elements from the collection represented by the stack.

```java
void clearStack () {
    s.removeAllElements();
}
```
x??

---
#### Handling Arithmetic Operations in a Stack-Based Expression Evaluator
Background context: The code snippet provided handles basic arithmetic operations such as addition, subtraction, multiplication, and division within an expression evaluator using a stack. It demonstrates how operators are processed based on their precedence.

:p How is the minus (`-`) operator handled in the given code?
??x
The `-` operator is handled by popping two operands from the stack, subtracting the second operand (top of the stack) from the first operand, and then pushing the result back onto the stack. This ensures that operations are performed correctly based on their order.

```java
case "-":
    // Found - operator, perform it (order matters).
    tmp = pop();
    push(pop() - tmp);
    break;
```
x??

---
#### Evaluating Arithmetic Expressions Using a Stack
Background context: The code snippet provides an implementation for evaluating arithmetic expressions using stacks. It includes handling of operators and operands to compute the result of the expression.

:p What does the `out.println(peek());` line do in the given switch-case structure?
??x
The `out.println(peek());` line prints the final result stored at the top of the stack, which is the evaluated value of the arithmetic expression. This line ensures that the computed result is displayed to the user.

```java
case "=":
    out.println(peek());
    break;
```
x??

---

#### Parser Generators Overview
Parser generators are tools that help create parsers for programming languages, scripts, and other input structures. They use a combination of lexical and grammatical specifications to recognize complex patterns within input data.

These tools can handle more advanced scanning tasks than simple classes like `StreamTokenizer` or `Scanner`, as they support the definition of tokens and rules for their sequence in a formal grammar.

For Java developers, there are several third-party solutions available that simplify the process of creating parsers:
- ANTLR
- JavaCC
- JParsec
- JFlex + CUP (like original lex & yacc)
- Parboiled

Each tool has its own strengths and can be used for various applications ranging from simple calculators to full language parsing.

:p What are parser generators, and why might a Java developer need them?
??x
Parser generators are specialized tools designed to create parsers capable of recognizing complex patterns within input data based on defined lexical and grammatical rules. They are useful for Java developers when dealing with more advanced scanning tasks that cannot be handled by simpler classes like `StreamTokenizer` or `Scanner`.

For example, they can help in creating a parser for a custom language or protocol where the tokens and their sequence must follow specific patterns.

```java
// Pseudocode to illustrate token definition in ANTLR
grammar MyGrammar;
myRule: ID ':' name;
name: [A-Za-z0-9]+;

ID : ('a'..'z'|'A'..'Z') ('a'..'z'|'A'..'Z'|'0'..'9'|'_')* ;
```
x??

---

#### Java Input Options
Java provides several options for creating or opening files. These include `StandardCopyOption`, `LinkOption`, `FileAttribute`, and `OpenOption`. Each option serves a specific purpose in how the file is accessed.

For instance, you can create a file with specific permissions using `PosixFilePermissions` or open it in read-only mode using `StandardOpenOption`.

Here's an example of creating a file with certain POSIX permissions:

:p How do Java developers control file permissions and attributes?
??x
Java developers use the `PosixFilePermission` class to control who can access files on disk, based on POSIX permissions. These permissions are defined for three actors: owner (user), group, and other.

Example code to set file permissions:
```java
Set<PosixFilePermission> perms = PosixFilePermissions.fromString("rwxr-xr--");
Path filePath = Paths.get("/tmp/xx");

// Apply the permissions when creating a file
Files.createFile(filePath, PosixFilePermissions.asFileAttribute(perms));

// Print the actual permissions to verify
System.out.println(Files.probeTypes(filePath).get().toString());
```

In this example, we set `OWNER_READ`, `OWNER_WRITE`, and `OWNER_EXECUTE` for the owner, read-only access for the group, and read-only access for others. The file is then created with these permissions.

x??

---

#### JFlex + CUP Example
JFlex generates lexical analyzers (lexers), while CUP generates parsers (yacc-like). Together, they can be used to create a complete parser for complex languages or protocols.

Here's an example of how you might use them together:

:p How do JFlex and CUP work together?
??x
JFlex is used to generate lexical analyzers, which recognize tokens in the input stream. CUP, on the other hand, generates parsers based on a BNF-like grammar definition.

Together, they can process complex inputs by first breaking down the text into tokens (lexing) and then interpreting these tokens according to predefined rules (parsing).

Here is an example of how you might set them up:

1. **JFlex**:
   - Create a JFlex file (`Lexer.flex`):
     ```flex
     /* Lexer.flex */
     %% // start of lexical rules
     [a-zA-Z_][a-zA-Z0-9]*    { return newToken(ID); }
     [0-9]+                   { return newToken(NUMBER); }
     .                        { return newToken(OTHER); }
     %%
     ```

2. **CUP**:
   - Create a CUP grammar file (`Parser.cup`):
     ```cup
     /* Parser.cup */
     %{
       public class MyParser {
         // Your parser implementation goes here.
       }
     %}
     
     ID : [a-zA-Z_][a-zA-Z0-9]* ;
     NUMBER : [0-9]+ ;
     
     program : ID '=' NUMBER { System.out.println("Found: " + $1 + "=" +$3); } ;
     ```

Using a build tool like Ant or Maven, you can generate the lexer and parser from these files:
```xml
<target name="generate">
    <java fork="true" classname="org.jflex.Main">
        <arg value="Lexer.flex"/>
        <classpath>
            <pathelement location="jflex.jar"/>
        </classpath>
    </java>

    <java fork="true" classname="cup.CUPParser">
        <arg value="-parser" />
        <arg value="MyParser" />
        <arg value="-classdir" />
        <arg value="target/classes" />
        <arg value="-destdir" />
        <arg value="target/classes" />
        <arg value="Parser.cup"/>
    </java>
</target>
```

x??

---

