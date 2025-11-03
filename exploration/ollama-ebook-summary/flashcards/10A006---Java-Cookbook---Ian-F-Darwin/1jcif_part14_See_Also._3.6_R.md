# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 14)

**Starting Chapter:** See Also. 3.6 Reversing a String by Word or by Character. Problem. Solution. 3.7 Expanding and Compressing Tabs

---

#### Reversing a String by Character Using StringBuilder
Background context: The `StringBuilder` class in Java provides an efficient way to reverse strings character by character. This method is straightforward and utilizes the built-in capabilities of `StringBuilder`.

:p How can you use `StringBuilder` to reverse a string?
??x
To reverse a string using `StringBuilder`, you can create an instance of `StringBuilder` with the original string, then call the `reverse()` method on it.

```java
String sh = "FCGDAEB";
System.out.println(sh + " -> " + new StringBuilder(sh).reverse());
```
This code snippet creates a string `"FCGDAEB"` and reverses its characters using `StringBuilder`.

x??

---

#### Reversing a String by Word Using Stack and StringTokenizer
Background context: To reverse a string word by word, you can use the `StringTokenizer` class to tokenize the input string into words. Each tokenized word is then pushed onto a stack. Finally, processing the stack in Last-In-First-Out (LIFO) order will result in the reversed order of words.

:p How does reversing a string by word work using `Stack` and `StringTokenizer`?
??x
Reversing a string by word involves tokenizing the input into individual words and pushing each word onto a stack. Since stacks follow LIFO, popping the elements from the stack will naturally reverse the order of the words.

```java
String s = "Father Charles Goes Down And Ends Battle";
Stack<String> myStack = new Stack<>();
StringTokenizer st = new StringTokenizer(s);
while (st.hasMoreTokens()) {
    myStack.push(st.nextToken());
}
// Process the stack in LIFO order to get reversed words
StringBuilder reversedString = new StringBuilder();
while (!myStack.isEmpty()) {
    reversedString.append(myStack.pop()).append(" ");
}
System.out.println(reversedString.toString().trim());
```
This code snippet first tokenizes the input string into words using `StringTokenizer` and pushes each word onto a stack. It then pops elements from the stack, appending them to a new `StringBuilder`, which results in the reversed order of words.

x??

---

#### EnTab Class for Expanding Tabs
Background context: The `EnTab` class is designed to convert spaces into tab characters in a file, allowing for more efficient use of space. This process is known as expanding tabs and is often used when saving files to disk where space usage is important.

:p What is the main purpose of the EnTab class?
??x
The main purpose of the `EnTab` class is to convert spaces into tab characters in a file, ensuring that text files are more compact without sacrificing readability. This is particularly useful for disk storage optimization.
x??

---
#### Constructor for EnTab Class
Background context: The `EnTab` class provides constructors with different parameters to allow users to specify the number of spaces each tab should replace.

:p What does the constructor in the `EnTab` class do?
??x
The constructor in the `EnTab` class initializes a new instance of the `EnTab` object and sets the tab spacing based on the number of spaces each tab should replace. If no specific value is provided, it defaults to 8 spaces.
```java
public EnTab(int n) {
    tabs = new Tabs(n);
}
public EnTab() {
    tabs = new Tabs();
}
```
x??

---
#### Entab Method for Process File Line by Line
Background context: The `entab` method processes a file line by line, replacing spaces with tab characters based on the specified tab spacing.

:p What does the `entab` method do?
??x
The `entab` method reads a file line by line and replaces spaces with tab characters. It works in conjunction with the `Tabs` class to determine where tab stops occur.
```java
public void entab(BufferedReader  is, PrintWriter  out) throws IOException {
    // Main loop: process entire file one line at a time.
    is.lines().forEach(line -> {
        out.println(entabLine (line));
    });
}
```
x??

---
#### EntabLine Method to Process Line by Line
Background context: The `entabLine` method processes each character in a string, replacing spaces with tab characters when they occur at tab stops.

:p What does the `entabLine` method do?
??x
The `entabLine` method processes each character in a line and replaces spaces with tab characters if they fall on tab stops. It ensures that trailing spaces are preserved.
```java
public String entabLine (String line) {
    int N = line.length(), outCol = 0;
    StringBuilder sb = new StringBuilder();
    char ch;
    int consumedSpaces = 0;
    for (int inCol = 0; inCol < N; inCol++) { // Cannot use foreach here
        ch = line.charAt(inCol);
        if (ch == ' ') {
            logger.info("Got space at " + inCol);
            if (tabs.isTabStop(inCol)) {
                logger.info("Got a Tab Stop " + inCol);
                sb.append('\t');
                outCol += consumedSpaces;
                consumedSpaces = 0;
            } else {
                consumedSpaces++;
            }
            continue;
        }
        while (inCol - 1 > outCol) { // Padding spaces
            logger.info("Padding space at " + inCol);
            sb.append(' ');
            outCol++;
        }
        sb.append(ch);
        outCol++;
    }
    for (int i = 0; i < consumedSpaces; i++) {
        logger.info("Padding space at end # " + i);
        sb.append(' ');
    }
    return sb.toString();
}
```
x??

---
#### DeTab Class for Compressing Tabs
Background context: The `DeTab` class is designed to convert tab characters back into spaces, reversing the process of expanding tabs.

:p What does the `DeTab` class do?
??x
The `DeTab` class converts tab characters in a file back into spaces. It works by expanding each tab character and converting it into one or more space characters.
```java
public void detab(BufferedReader  is, PrintWriter  out) throws IOException {
    is.lines().forEach(line -> {
        out.println(detabLine (line));
    });
}
```
x??

---
#### Detab Method to Process Line by Line
Background context: The `detab` method processes each line of text, converting tab characters into spaces.

:p What does the `detabLine` method do?
??x
The `detabLine` method converts tab characters in a line back into spaces. It ensures that each tab character is expanded with at least one space.
```java
public String detabLine (String line) {
    char c;
    int col = 0;
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < line.length(); i++) {
        if ((c = line.charAt(i)) == '\t') { // Tab character
            sb.append(c);
            ++col;
            continue;
        }
        do { // Expand tab to spaces
            sb.append(' ');
        } while (.ts.isTabStop(++col));
    }
    return sb.toString();
}
```
x??

---
#### Tabs Class for Tab Logic Handling
Background context: The `Tabs` class handles the logic for determining where tab stops occur and provides methods to set and check these positions.

:p What is the purpose of the `Tabs` class?
??x
The `Tabs` class provides a mechanism to determine where tab stops are located. It allows setting the number of spaces each tab should replace and checks if a given column is a tab stop.
```java
public boolean isTabStop (int col) {
    if (col <= 0)
        return false;
}
```
x??

---

#### Converting Strings to Upper and Lower Case
Background context: The `String` class in Java provides methods to convert strings to uppercase or lowercase. This is useful when you need to standardize string representations for comparison or processing.

If applicable, add code examples with explanations:
```java
public class StringCaseConversion {
    public static void main(String[] args) {
        String name = "Java Cookbook";
        System.out.println("Normal:\t" + name);
        System.out.println("Upper:\t" + name.toUpperCase());
        System.out.println("Lower:\t" + name.toLowerCase());
    }
}
```
:p How do you convert a string to uppercase in Java?
??x
You can convert a string to uppercase using the `toUpperCase()` method provided by the `String` class. This method returns a new string with all characters converted to their uppercase equivalents.

```java
String upperCaseName = name.toUpperCase();
```

This code creates a new string where all alphabetic characters are transformed into uppercase, while non-alphabetic characters remain unchanged.
x??

---

#### Comparing Strings Without Case Sensitivity
Background context: Sometimes, you need to compare strings regardless of their case. Java provides the `equalsIgnoreCase()` method for this purpose.

If applicable, add code examples with explanations:
```java
public class StringCaseComparison {
    public static void main(String[] args) {
        String name = "Java Cookbook";
        String javaName = "java cookBook"; // If it were Java identifiers :-)
        if (name.equals(javaName)) {
            System.err.println("equals() correctly reports false");
        } else {
            System.err.println("equals() incorrectly reports true");
        }
        if (name.equalsIgnoreCase(javaName)) {
            System.err.println("equalsIgnoreCase() correctly reports true");
        } else {
            System.err.println("equalsIgnoreCase() incorrectly reports false");
        }
    }
}
```
:p How do you compare two strings without considering case sensitivity in Java?
??x
You can use the `equalsIgnoreCase()` method to compare two strings ignoring their case. This method returns `true` if both strings are equal when compared disregarding case, and `false` otherwise.

```java
if (name.equalsIgnoreCase(javaName)) {
    System.err.println("equalsIgnoreCase() correctly reports true");
} else {
    System.err.println("equalsIgnoreCase() incorrectly reports false");
}
```

This code snippet checks if the two strings are equal regardless of their case. If they are considered equal, it prints that `equalsIgnoreCase()` correctly reports `true`.
x??

---

#### Using Locale for Case Conversion
Background context: The `toUpperCase()` and `toLowerCase()` methods in Java can also take a `Locale` argument to specify different rules based on the locale.

If applicable, add code examples with explanations:
```java
public class StringCaseWithLocale {
    public static void main(String[] args) {
        String name = "Java Cookbook";
        System.out.println("Upper (default):" + name.toUpperCase());
        Locale frenchLoc = new Locale("fr", "FR");
        System.out.println("Upper (French): " + name.toUpperCase(frenchLoc));
    }
}
```
:p How can you use the `Locale` parameter with case conversion methods in Java?
??x
You can use the `toUpperCase()` and `toLowerCase()` methods by passing a `Locale` object as an argument to specify different rules based on the locale. This is particularly useful for internationalization purposes.

```java
Locale frenchLoc = new Locale("fr", "FR");
System.out.println("Upper (French): " + name.toUpperCase(frenchLoc));
```

This code converts the string to uppercase according to French language rules, which can be different from the default behavior.
x??

---

#### String Case in Java API Internationalization and Localization
Background context: The Java API provides extensive internationalization and localization features. This includes methods like `toUpperCase()` and `toLowerCase()` that support locale-specific conversions.

If applicable, add code examples with explanations:
```java
public class StringCaseInternationalization {
    public static void main(String[] args) {
        String name = "Java Cookbook";
        System.out.println("Upper (default):" + name.toUpperCase());
        Locale germanLoc = new Locale("de", "DE");
        System.out.println("Upper (German): " + name.toUpperCase(germanLoc));
    }
}
```
:p What are the internationalization features in Java’s `String` class related to case conversion?
??x
Java's `String` class supports locale-specific conversions for case transformation. This is part of its comprehensive internationalization and localization capabilities.

```java
Locale germanLoc = new Locale("de", "DE");
System.out.println("Upper (German): " + name.toUpperCase(germanLoc));
```

This code snippet demonstrates converting the string to uppercase according to German language rules, showcasing how locale-specific transformations can be performed.
x??

---

