# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 21)

**Starting Chapter:** 4.11 Program Full Grep

---

#### Regular Expressions and Pattern Matching
Background context explaining regular expressions, their syntax, and how they are used for pattern matching. Highlight the use of `Pattern` and `Matcher` classes in Java.

:p What is a regular expression (regex) and what does it do?
??x
A regular expression is a sequence of characters that defines a search pattern within strings. In this context, we use regex patterns to match specific log entries from a web server's access log. The `Pattern` class compiles the regex into a reusable form, and the `Matcher` class provides methods to perform matching operations.

```java
// Example regex for matching HTTP request logs
String patternStr = "^\\d+\\.\\d+\\.\\d+\\.\\d+ - - \$$\\w+/\\d{2}/\\d{4}:\\d{2}:\\d{2}:\\d{2} [+-]\\d{4}\$$ \"GET /\\S+ HTTP/1.0\" \\d{3} \\d+";
Pattern pattern = Pattern.compile(patternStr);
```
x??

---

#### String Tokenizer vs Regular Expressions
Background context comparing `StringTokenizer` and regular expressions, highlighting the differences in flexibility and performance.

:p How does `StringTokenizer` compare to using regular expressions for parsing log files?
??x
`StringTokenizer` is a simpler method of splitting strings based on delimiters. It uses predefined delimiters such as spaces or commas, making it less flexible compared to regular expressions. Regular expressions provide much more flexibility and power, allowing complex patterns to be matched efficiently.

Example code using `StringTokenizer`:

```java
String line = "123.45.67.89 - - [27/Oct/2000:09:27:09 -0400] \"GET /java/javaResources.html HTTP/1.0\" 200 10450";
StringTokenizer tokenizer = new StringTokenizer(line, " -[/:]");
// The logic would be more complex and less flexible compared to regex.
```

Example using regular expressions:

```java
String line = "123.45.67.89 - - [27/Oct/2000:09:27:09 -0400] \"GET /java/javaResources.html HTTP/1.0\" 200 10450";
Pattern pattern = Pattern.compile("\\d+\\.\\d+\\.\\d+\\.\\d+|\\w+/\\S+|\\d{3} \\d+");
Matcher matcher = pattern.matcher(line);
// More straightforward and flexible.
```
x??

---

#### Command Line Option Parsing
Background context on Unix `grep` command-line options, their usage, and the implementation in Java using a custom `GetOpt` class.

:p What is the purpose of implementing a `GetOpt` class for parsing command-line arguments?
??x
The purpose of implementing a `GetOpt` class is to parse command-line options in a manner similar to Unix `grep`. This allows users to specify various flags and options, such as counting matches (`-c`), ignoring case sensitivity (`-i`), or listing file names only (`-l`). In Java, this can be achieved using the `GetOpt` class from the provided source.

Example usage in Java:

```java
String[] args = {"-c", "-i", "pattern.txt", "input.log"};
GetOpt.getopts(args);
```
x??

---

#### JGrep Program Overview
Background context on creating a full-fledged grep-like program, `JGrep`, with command-line options and file pattern matching.

:p What is the main goal of implementing `JGrep`?
??x
The main goal of implementing `JGrep` is to create a versatile line-matching program similar to Unix `grep`. It will support various command-line options such as counting matches (`-c`), ignoring case sensitivity (`-i`), and listing file names only (`-l`). The program will read patterns from files or the command line, search through text files, and print matching lines.

Example of how JGrep might be invoked:

```sh
java regex/JGrep -ci pattern.txt input.log
```
x??

---

#### Example 4-12: Full Grep Program Implementation
Background context on the example code for `JGrep`, which includes parsing command-line arguments, reading patterns, and matching lines.

:p What does the code in Example 4-12 do?
??x
The code in Example 4-12 implements a full-fledged grep-like program called `JGrep`. It parses command-line options using a custom `GetOpt` implementation, reads patterns from files or the command line, and searches for matches in text files. The program supports options like counting matches (`-c`), ignoring case sensitivity (`-i`), and listing file names only (`-l`). Matching lines are printed according to the specified options.

```java
public class JGrep {
    public static void main(String[] args) {
        // Implementation of command-line option parsing using GetOpt
        // Reading patterns and files, then searching for matches.
    }
}
```
x??

---

#### JGrep Main Method Overview
Background context: The `main` method of the `JGrep` class is responsible for parsing command-line arguments and setting up a `GetOpt` object to process these arguments. It then processes input files or stdin based on the provided options.

:p What does the main method do in the JGrep class?
??x
The main method initializes the pattern to search for and sets various flags based on command-line arguments. It then processes either standard input or a list of file names, depending on whether there are any file names provided after parsing the arguments.
```java
public static void main(String[] args) {
    if (args.length < 1) { // Check for insufficient arguments
        System.err.println(USAGE);
        System.exit(1); // Exit with error code 1
    }
    String patt = null;
    GetOpt go = new GetOpt("cf: hilnrRsv"); // Initialize the GetOpt object
    char c;
    while ((c = go.getopt(args)) != 0) { // Process command-line options
        switch(c) {
            case 'c': countOnly = true; break;
            case 'f': /* External file contains the pattern */
                try (BufferedReader b = new BufferedReader(new FileReader(go.optarg()))) {
                    patt = b.readLine();
                } catch (IOException e) {
                    System.err.println("Can't read pattern file " + go.optarg());
                    System.exit(1);
                }
                break;
            case 'h': dontPrintFileName = true; break;
            // Other cases for setting flags
        }
    }
    int ix = go.getOptInd(); // Get the index of the first argument not processed
    if (patt == null) patt = args[ix++]; // Set pattern from command-line arguments or stdin
    JGrep prog = null; // Create an instance of JGrep with the pattern
    try {
        prog = new JGrep(patt);
    } catch (PatternSyntaxException ex) { // Handle regex syntax errors
        System.err.println("RE Syntax error in " + patt);
        return;
    }
    if (args.length == ix) { // No files provided, process stdin
        dontPrintFileName = true; // Don't print filenames if reading from stdin
        if (recursive) {
            System.err.println("Warning: recursive search of stdin.");
        }
        prog.process(new InputStreamReader(System.in), null);
    } else { // Process file arguments
        if (!dontPrintFileName && ix == args.length - 1) dontPrintFileName = true;
        if (recursive) dontPrintFileName = false; // Handle directories recursively
        for (int i=ix; i<args.length; i++) {
            try {
                prog.process(new File(args[i])); // Process each file argument
            } catch(Exception e) {
                System.err.println(e); // Handle exceptions during file processing
            }
        }
    }
}
```
x??

---
#### GetOpt Class Usage
Background context: The `GetOpt` class is used to parse command-line options in the `main` method of the `JGrep` class. It allows setting and getting command-line arguments and their values.

:p How does the `GetOpt` object handle command-line arguments?
??x
The `GetOpt` object processes the command-line arguments by checking each one against a set of predefined options (e.g., "cf: hilnrRsv"). If an option is recognized, it sets corresponding flags or retrieves associated values. The method `getopt(args)` returns the next option code and updates internal state.
```java
GetOpt go = new GetOpt("cf: hilnrRsv"); // Initialize with options to recognize
char c;
while ((c = go.getopt(args)) != 0) { // Process each argument
    switch(c) {
        case 'c': countOnly = true; break;
        case 'f': /* External file contains the pattern */
            try (BufferedReader b = new BufferedReader(new FileReader(go.optarg()))) {
                patt = b.readLine();
            } catch (IOException e) {
                System.err.println("Can't read pattern file " + go.optarg());
                System.exit(1);
            }
            break;
        case 'h': dontPrintFileName = true; break;
        // Other cases for setting flags
    }
}
```
x??

---
#### JGrep Flags Initialization
Background context: The `JGrep` class initializes several static boolean fields to represent various command-line options. These fields are used to control the behavior of the program, such as whether to count lines or process directories recursively.

:p What are the flags initialized in the JGrep main method?
??x
The main method initializes several static flags representing different behaviors:
- `countOnly`: Set to true if the user specifies the `-c` option.
- `ignoreCase`: Set to true if the user specifies the `-i` option.
- `dontPrintFileName`: Set based on whether filenames should be printed or suppressed.
- `listOnly`: Set to true if the user specifies the `-l` option.
- `numbered`: Set to true if the user specifies the `-n` option.
- `recursive`: Set to true if the user specifies the `-r` or `-R` options.
- `silent`: Set to true if the user specifies the `-s` option.
- `inVert`: Set to true if the user specifies the `-v` option.

The flags are used to customize the behavior of the `JGrep` class when processing files and displaying results.
```java
// Flags in JGrep main method
protected static boolean countOnly = false;
protected static boolean ignoreCase = false;
protected static boolean dontPrintFileName = false;
protected static boolean listOnly = false;
protected static boolean numbered = false;
protected static boolean recursive = false;
protected static boolean silent = false;
protected static boolean inVert = false;
```
x??

---
#### Pattern Matching in JGrep
Background context: The `JGrep` class uses a regular expression pattern to match lines in files or stdin. It initializes the `Pattern` and `Matcher` objects based on the input pattern.

:p How does JGrep initialize its pattern for matching?
??x
The main method constructs a `Pattern` object from the provided regex pattern. This pattern is then used to create a `Matcher` object, which can be used to search through lines of text.
```java
JGrep prog = null;
try {
    prog = new JGrep(patt); // Create an instance with the regex pattern
} catch (PatternSyntaxException ex) {
    System.err.println("RE Syntax error in " + patt);
    return; // Exit if there's a syntax error
}
```
x??

---
#### Processing Files and Stdin in JGrep
Background context: The `JGrep` class processes either standard input or file arguments based on the presence of filenames. It handles both scenarios by calling appropriate methods to process lines according to the specified options.

:p How does JGrep handle file processing?
??x
If there are no files provided, JGrep processes standard input using an `InputStreamReader`. If filenames are provided, it processes each file individually or recursively based on the `-r` and `-R` flags. It also handles exceptions during file reading.
```java
if (args.length == ix) { // No files provided, process stdin
    dontPrintFileName = true; // Don't print filenames if reading from stdin
    if (recursive) {
        System.err.println("Warning: recursive search of stdin.");
    }
    prog.process(new InputStreamReader(System.in), null);
} else { // Process file arguments
    if (!dontPrintFileName && ix == args.length - 1) dontPrintFileName = true;
    if (recursive) dontPrintFileName = false; // Handle directories recursively
    for (int i=ix; i<args.length; i++) {
        try {
            prog.process(new File(args[i])); // Process each file argument
        } catch(Exception e) {
            System.err.println(e); // Handle exceptions during file processing
        }
    }
}
```
x??

---

#### Constructor for JGrep Class
Background context: The constructor initializes a `JGrep` object with a provided regex pattern. It handles debugging and compiles the regex pattern based on case sensitivity options.

:p What does the constructor of the `JGrep` class do?
??x
The constructor initializes an instance of the `JGrep` class with a given regular expression (`patt`). If `debug` is enabled, it prints out the provided pattern. It then compiles this regex into a pattern object and sets up a matcher to work on empty strings initially.

```java
public JGrep(String patt) throws PatternSyntaxException {
    if (debug) { // Debug mode check
        System.err.printf("JGrep.JGrep( percents) percentn", patt); // Print debug information
    }
    int caseMode = ignoreCase ? Pattern.UNICODE_CASE | Pattern.CASE_INSENSITIVE : 0; // Set case sensitivity options
    pattern = Pattern.compile(patt, caseMode); // Compile the regex pattern
    matcher = pattern.matcher(""); // Initialize a matcher object for empty strings
}
x??

---

#### Process Method with File as Argument
Background context: The `process` method handles processing of files and directories based on their type. It throws exceptions if the file does not exist or cannot be read, processes single files by reading them line-by-line, and recursively processes directories.

:p What is the role of the `process(File file)` method in the JGrep class?
??x
The `process` method handles the processing of a given `File`. It checks if the file exists and can be read. If not, it throws a `FileNotFoundException`. Depending on whether the file is a directory or a regular file, it processes them accordingly. For directories, it either prints an error message and returns, or recursively processes all files within that directory.

```java
public void process(File file) throws FileNotFoundException {
    if (!file.exists() || !file.canRead()) { // Check existence and readability
        throw new FileNotFoundException("Can't read file " + file.getAbsolutePath());
    }
    if (file.isFile()) { // Process single file
        process(new BufferedReader(new FileReader(file)), file.getAbsolutePath());
        return;
    } 
    if (file.isDirectory()) {
        if (!recursive) { // Print error message and exit for non-recursive mode with directories
            System.err.println("ERROR: -r not specified but directory given " + file.getAbsolutePath());
            return;
        }
        for (File nf : file.listFiles()) { // Recursively process each file in the directory
            process(nf);
        }
        return;
    }
    System.err.println("WEIRDNESS: neither file nor directory: " + file.getAbsolutePath()); // Handle unexpected cases
}
x??

---

#### Process Method with Reader and File Name as Arguments
Background context: The `process` method takes a reader object for reading lines from the input file and prints matched lines or other output based on the provided options. It counts matches, prints filenames, or outputs non-matching lines.

:p What is the role of the `process(Reader ifile, String fileName)` method in the JGrep class?
??x
The `process` method reads each line from an input file using a `BufferedReader`. It uses a matcher to find patterns within each line. Depending on whether `-l`, `-c`, or `-v` options are set, it prints filenames, counts matches, prints matching lines with or without filenames, or outputs non-matching lines.

```java
public void process(Reader ifile, String fileName) {
    String inputLine;
    int matches = 0;
    try (BufferedReader reader = new BufferedReader(ifile)) { // Read each line from the file
        while ((inputLine = reader.readLine()) != null) {
            matcher.reset(inputLine); // Reset the matcher for the current line
            if (matcher.find()) {
                if (listOnly) { // Print filename and exit after first match (-l option)
                    System.out.println(fileName);
                    return;
                }
                if (countOnly) { // Count matches only (-c option)
                    matches++;
                } else {
                    if (!dontPrintFileName) { // Print the filename before each line
                        System.out.print(fileName + ": ");
                    }
                    System.out.println(inputLine); // Print matching lines
                }
            } else if (invert) { // Output non-matching lines (-v option)
                System.out.println(inputLine);
            }
        }
        if (countOnly) // Print total matches at the end for -c option
            System.out.println(matches + " matches in " + fileName);
    } catch (IOException e) {
        System.err.println(e); // Handle any I/O exceptions that occur
    }
}
x??

---

#### Lavarand and Pseudorandom Number Generators (PRNG)
Background context: The provided text discusses the now-defunct Lavarand, which used digitized video of lava lamps to generate "hardware-based" randomness. It also explains that most conventional random number generators are pseudorandom, meaning they are not truly random but are good enough for most purposes. True randomness comes from specialized hardware like analog sources of Brownian noise.

:p What is Lavarand and how did it work?
??x
Lavarand was a project that used digitized video of lava lamps to provide "hardware-based" randomness. The system recorded the movement of lava in lava lamps, which were then converted into digital signals to produce random numbers. This method aimed to generate true randomness but has since become obsolete.

No code examples needed for this concept.
x??

---

#### Pseudorandom Number Generators (PRNG) and Their Use
Background context: The text explains that conventional random number generators are pseudorandom, meaning they simulate randomness rather than providing truly random numbers. True randomness comes from specialized hardware like analog sources of Brownian noise.

:p What is the difference between true randomness and pseudorandomness?
??x
True randomness arises from an unpredictable source such as physical processes (e.g., thermal noise in circuits). Pseudorandom number generators (PRNGs) simulate randomness using algorithms, which means their output can be predictable if the initial seed or algorithm is known.

No code examples needed for this concept.
x??

---

#### Java's `java.lang.Math` Class
Background context: The text mentions that `java.lang.Math` provides an entire math library including trigonometry, conversions, rounding, and other mathematical functions. It is part of the standard Java API and can be used to perform various mathematical operations.

:p What does the `java.lang.Math` class contain?
??x
The `java.lang.Math` class contains a wide range of mathematical functions such as trigonometric methods (sine, cosine), conversions between different units (degrees to radians), rounding methods, and more. For example, it includes methods like `sin()`, `cos()`, `toRadians()`, `round()`.

Example code:
```java
public class MathClassExample {
    public static void main(String[] args) {
        double angleInDegrees = 45;
        // Convert degrees to radians and then calculate sine
        double sinValue = Math.sin(Math.toRadians(angleInDegrees));
        System.out.println("The sine of " + angleInDegrees + " degrees is: " + sinValue);
    }
}
```
x??

---

#### Java's `java.math` Package for Big Numbers
Background context: The text mentions that the `java.math` package provides support for big numbers, which are larger than standard built-in long integers. This package can be useful when dealing with very large or very small numerical values.

:p What is the purpose of the `java.math` package?
??x
The `java.math` package in Java provides classes like `BigInteger` and `BigDecimal`, allowing you to handle numbers that exceed the limits of standard numeric types such as `long`. This is useful for financial calculations, cryptographic operations, and other scenarios where precision or large values are required.

Example code:
```java
import java.math.BigInteger;

public class BigNumbers {
    public static void main(String[] args) {
        BigInteger bigInt1 = new BigInteger("12345678901234567890");
        BigInteger bigInt2 = new BigInteger("98765432109876543210");
        // Perform addition
        BigInteger sum = bigInt1.add(bigInt2);
        System.out.println("The sum is: " + sum);
    }
}
```
x??

---

#### Java's Handling of Numeric Data and Reliability
Background context: The text emphasizes that Java ensures program reliability through mechanisms like exception handling (using try-catch blocks) and type casting. This approach helps in writing more robust and portable code.

:p What are the common ways to notice Java’s commitment to reliability?
??x
Java ensures reliability by requiring programmers to handle exceptions using try-catch blocks, which can help prevent abrupt program termination due to runtime errors. Additionally, Java uses casting when storing values that might not fit into a variable's type, ensuring that operations do not fail unexpectedly.

Example code:
```java
public class ExceptionHandling {
    public static void main(String[] args) {
        try {
            int[] numbers = {1, 2, 3};
            System.out.println(numbers[5]); // This will throw an ArrayIndexOutOfBoundsException
        } catch (ArrayIndexOutOfBoundsException e) {
            System.out.println("Caught an exception: " + e.getMessage());
        }
    }
}
```
x??

---

#### Validating and Converting String to Double
Background context explaining how to validate whether a string represents a valid number and convert it to a double. The `Double.parseDouble` method is used, but if the input is invalid, a `NumberFormatException` is thrown.

:p How do you check if a given string can be parsed into a double using Java?
??x
To check if a given string can be parsed into a double in Java, you use the `Double.parseDouble` method within a try-catch block. If the parsing fails (i.e., the input is not a valid number), an exception (`NumberFormatException`) is thrown and caught to handle invalid inputs.

```java
public static void main(String[] argv) {
    String aNumber = argv[0];  // assume it's given as command-line argument
    double result;
    try {
        result = Double.parseDouble(aNumber);
        System.out.println("Number is " + result);
    } catch (NumberFormatException exc) {
        System.out.println("Invalid number " + aNumber);
        return;
    }
}
```

x??

---

#### Regular Expression for Number Validation
Background context explaining how to use regular expressions to validate whether a string contains a valid number, and differentiating between integers and floating-point numbers.

:p How can you use regular expressions to determine if a given string represents a valid number?
??x
You can use a regular expression to check if a string contains a valid number. The regex `[+-]*\d*\.?\d*[dDeEfF]*` is used, which allows for optional signs (`[+-]*`), digits before and after the decimal point (`\d*\.?\d*`), and scientific notation indicators (`[dDeEfF]*`).

```java
public static Number process(String s) {
    if (s.matches("[+-]*\\d*\\.?\\d*[dDeEfF]*")) {
        try {
            double dValue = Double.parseDouble(s);
            System.out.println("It's a double: " + dValue);
            return Double.valueOf(dValue);
        } catch (NumberFormatException e) {
            System.out.println("Invalid double: " + s);
            return Double.NaN;
        }
    } else { // did not contain ., d, e, or f, so try as int
        // Handle integer conversion logic here
    }
}
```

x??

---

#### Distinguishing Between Integers and Floating-Point Numbers
Background context explaining how to differentiate between integers and floating-point numbers by checking for specific characters in the input string.

:p How can you determine if a given number is an integer or a floating-point number using Java?
??x
To distinguish between integers and floating-point numbers, you need to check for the presence of certain characters such as `.`, `d`, `e`, or `f`. If any of these characters are present in the input string, it can be treated as a double; otherwise, convert it as an integer.

```java
public static Number process(String s) {
    if (s.matches("[+-]*\\d*\\.?\\d*[dDeEfF]*")) {  // checks for ., d, e, or f
        try {
            double dValue = Double.parseDouble(s);
            System.out.println("It's a double: " + dValue);
            return Double.valueOf(dValue);
        } catch (NumberFormatException e) {
            System.out.println("Invalid double: " + s);
            return Double.NaN;
        }
    } else { // did not contain ., d, e, or f
        try {
            int intValue = Integer.parseInt(s);  // convert as integer if no decimal point
            System.out.println("It's an integer: " + intValue);
            return Integer.valueOf(intValue);
        } catch (NumberFormatException e) {
            System.out.println("Invalid number: " + s);
            return Double.NaN;
        }
    }
}
```

x??

---

