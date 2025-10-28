# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 43)

**Starting Chapter:** 10.4 Printing with Formatter and printf

---

#### Console Class for Reading from a Program's Controlling Terminal
Background context explaining that the `Console` class is used to read directly from a program’s controlling terminal, which is typically the user’s terminal window or command prompt. The standard input can be changed by piping or redirection on most operating systems.
If you need to bypass any indirections and read from wherever the user is sitting, the `Console` class is often your best choice.

:p How do you obtain an instance of the `Console` class?
??x
You cannot instantiate `Console` yourself; instead, you must get an instance from the `System.console()` method.
```java
String name = System.console().readLine("What is your name? ");
```
x??

---

#### Checking for a Null Console Instance
Background context explaining that the `System.console()` method can return null if there isn't a connected console. This is particularly relevant in IDEs like Eclipse where a controlling terminal might not be set up when running as a Java application.

:p How should production-quality code handle a possible null return from `System.console()`?
??x
Production-quality code should always check for `null` before using the `Console`. If it fails, use a logger or just plain `System.out`.
```java
if (console != null) {
    String name = console.readLine("What is your name? ");
} else {
    System.err.println("No console available.");
}
```
x??

---

#### Reading a Password Without Echoing
Background context explaining that the `Console` class supports reading passwords without echoing, which is useful for preventing shoulder surfing. This feature has been standard in command-line applications for decades.

:p How can you read a password from the user using the `Console` class?
??x
You can use the `readPassword()` method of the `Console` class with a prompt argument.
```java
char[] password = console.readPassword("Password: ");
```
x??

---

#### Converting Byte Array to String for Passwords
Background context explaining that the `readPassword()` method returns an array of bytes, which can be used directly in some encryption and security APIs. It is often recommended to convert this byte array into a string.

:p How do you convert the byte array returned by `readPassword()` to a string?
??x
You can convert the byte array returned by `readPassword()` to a string using the `new String(byteArray)` constructor.
```java
String passwordStr = new String(password);
```
x??

---

#### Overwriting the Byte Array for Security
Background context explaining that it is generally advised to overwrite the byte array after use to prevent security leaks when other code can access the stack, although this might be less critical if a string has already been constructed.

:p What is recommended after using the `readPassword()` method?
??x
It is generally advised to overwrite the byte array with zeros or random data to prevent security leaks.
```java
for (int i = 0; i < password.length; i++) {
    password[i] = 0;
}
```
x??

---

#### Formatter and printf Overview
Formatter is a powerful tool for formatting output in Java, similar to C's printf. It provides detailed control over how values are displayed, making it versatile for various printing tasks. The `java.util.Formatter` class is used for complex formatting needs, while `String.format()` or `PrintWriter.printf()`/`PrintStream.printf()` offer simpler methods.
:p What is the main purpose of using Java's Formatter?
??x
Formatter primarily serves to format output with precise control over how values are displayed. It allows developers to specify exactly how numbers, strings, dates, and other data types should be printed, ensuring consistent and readable output. This is particularly useful for creating well-structured reports or log files.
??x

---

#### Using Formatter Explicitly
To use `Formatter` explicitly, you create a new instance of the class and then call its `format()` method to format your desired string.
:p How do you create a simple formatted print statement using Formatter?
??x
You can create a `Formatter` object and then use its `format()` method. For example:
```java
Formatter fmtr = new Formatter();
fmtr.format("The year is %04d - The value of PI is %.2f", 1956, Math.PI);
```
This will format the output with a four-digit width for the year and two decimal places for PI. Finally, you should call `close()` on the `Formatter` object to free resources.
??x

---

#### Using String.format()
For simpler formatting needs, `String.format()` can be used directly in your code. It creates a string according to the format specification.
:p How do you use `String.format()` for simple printing tasks?
??x
You can use `String.format()` to create a formatted string without explicitly creating a `Formatter` object. For example:
```java
String result = String.format("The year is %04d - The value of PI is %.2f", 1956, Math.PI);
```
This method directly returns the formatted string.
??x

---

#### Using PrintWriter.printf() or PrintStream.printf()
For output streams like `PrintWriter` and `PrintStream`, you can use `printf()` methods which internally utilize a `Formatter`.
:p How do you format output using `PrintWriter.printf()`?
??x
You can format output directly in the stream by calling `printf()`. For example:
```java
System.out.printf("The year is %04d - The value of PI is %.2f", 1956, Math.PI);
```
This will print the formatted string to standard output.
??x

---

#### Format Codes for Numbers
Format codes in `Formatter` use a syntax like `%[argument_number]$[width][.precision]type`. For example, `%04d` formats an integer with a minimum width of 4 characters, and `%f` formats floating-point values.
:p What is the format code used to print integers with at least four digits?
??x
The format code for printing integers with at least four digits is `%04d`. This specifies that the integer should be printed in decimal format and padded with leading zeros if necessary to ensure a minimum width of 4 characters.
??x

---

#### Format Codes for Floating-Point Numbers
Floating-point numbers can be formatted using codes like `%f` for general floating point, `%e` for scientific notation, or `%g` which chooses between `f` and `e` based on the value. You can also specify precision with a period followed by the number of decimal places.
:p How do you format a floating-point number to two decimal places?
??x
To format a floating-point number to two decimal places, you use the code `%4.2f`. This specifies that the number should be printed in general floating point format and have exactly two digits after the decimal point.
??x

---

#### Formatting Dates with t Codes
For date formatting, codes start with `t` followed by a specific letter like `Y`, `m`, or `d`. For example, `%1$tY` formats the year of an argument as at least four digits.
:p How do you format a date to show only the year?
??x
To format a date to display only the year with at least four digits, use the code `%tY`. This specifies that the fourth digit (and more) should be shown for the year part of the date.
??x

---

#### Example Usage in Code
The `FormatterDemo` class demonstrates how to create and use a `Formatter`, `String.format()`, and `System.out.printf()` for printing formatted output.
:p What does the `FormatterDemo` class do?
??x
The `FormatterDemo` class showcases various methods of using formatting techniques. It includes examples of:
1. Creating and using a `Formatter` object to format strings.
2. Using `String.format()` for simpler string formatting tasks.
3. Utilizing `System.out.printf()` to print formatted output directly.
The class prints the year 1956 and the value of PI with different formats, demonstrating the flexibility of Java's formatting capabilities.
??x

#### Reading Characters One at a Time
Background context explaining the concept. The `read()` method of the `Reader` class is defined to return an `int` so that it can use `-1` (EOF) to indicate the end of the file. This allows handling of characters as integers, even when casting them back to `char`.
If applicable, add code examples with explanations.
:p How do you read a file one character at a time in Java?
??x
You can read a file one character at a time using the `read()` method of the `Reader` class. The method returns an `int`, which allows handling the end-of-file condition (`-1`) and characters as integers.
```java
public class ReadCharsOneAtATime {
    void doFile(Reader is) throws IOException {
        int c;
        while ((c = is.read()) != -1) { // Use '!=' for comparison, not '.='
            System.out.print((char) c);  // Cast to char and print it
        }
    }
}
```
x??

---

#### Using StringTokenizer for File Scanning
Background context explaining the concept. The `StringTokenizer` class is used to split lines of text into tokens based on a specified delimiter, making it useful for parsing fixed formats or structured data.
If applicable, add code examples with explanations.
:p How can you use `StringTokenizer` to parse user@host.domain format from each line in a file?
??x
You can use `StringTokenizer` along with `BufferedReader.readLine()` to process lines of text. For the `user@host.domain` format, you tokenize the line using the `@` character as the delimiter and extract the username and host parts.
```java
public class ScanStringTok {
    static void process(String fileName) throws IOException {
        String s = null;
        try (BufferedReader is = new BufferedReader(new FileReader(fileName))) {
            while ((s = is.readLine()) != null) {
                StringTokenizer st = new StringTokenizer(s, "@", true);
                String user = (String) st.nextElement();  // Get the first token
                st.nextElement();                         // Consume the '@' token
                String host = (String) st.nextElement();  // Get the second token
                System.out.println("User name: " + user + "; host part: " + host);
            }
        } catch (NoSuchElementException ix) {
            System.err.println("Malformed input " + s);
        } catch (IOException e) {
            System.err.println(e);
        }
    }
}
```
x??

---

#### Using StreamTokenizer for File Scanning
Background context explaining the concept. The `StreamTokenizer` class provides a more flexible way to read and categorize tokens from an input stream, suitable for more complex parsing scenarios.
If applicable, add code examples with explanations.
:p How can you use `StreamTokenizer` to implement a simple calculator?
??x
You can use `StreamTokenizer` to tokenize input for a simple immediate-mode stack-based calculator. Tokens are read and processed based on their type (number, operator, etc.). Here's an example of how it works:
```java
public class SimpleCalcStreamTok {
    protected StreamTokenizer tf;
    // Other fields...

    public static void main(String[] av) throws IOException {
        if (av.length == 0)
            new SimpleCalcStreamTok(new InputStreamReader(System.in)).doCalc();
        else
            for (int i = 0; i < av.length; i++)
                new SimpleCalcStreamTok(av[i]).doCalc();
    }

    public void setOutput(PrintWriter out) {
        this.out = out;
    }

    protected void doCalc() throws IOException {
        int iType;
        double tmp;
        while ((iType = tf.nextToken()) != StreamTokenizer.TT_EOF) {
            switch (iType) {
                case StreamTokenizer.TT_NUMBER:
                    push(tf.nval);
                    break;
                // Other cases for operators, etc.
                default:
                    out.println("What's this? iType = " + iType);
            }
        }
    }

    void push(double val) {
        s.push(Double.valueOf(val));
    }
}
```
x??

---

Each of the flashcards covers a key concept from the provided text, explaining its context and providing relevant code examples.

