# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 16)

**Starting Chapter:** 3.12 Using a Particular Locale. Problem. Solution. 3.13 Creating a Resource Bundle

---

#### Getting Available Locales
Background context: The `Locale.getAvailableLocales()` method provides a list of all available locales that can be used for formatting and other operations. This is useful when you need to know which locale options are supported by your system.

:p How do you get the list of available locales in Java?
??x
You can use the `Locale.getAvailableLocales()` method to retrieve an array of `Locale` objects representing all available locales.
```java
Locale[] availableLocales = Locale.getAvailableLocales();
```
x??

---

#### Using a Particular Locale
Background context: Sometimes, you might want to use a locale other than the default one for operations such as formatting dates or numbers. This is particularly useful when you need to cater to specific language and regional preferences.

:p How do you obtain a `Locale` object in Java?
??x
You can obtain a `Locale` object either by using predefined variables provided by the `Locale` class or by constructing your own `Locale` object with a language code and country code.
```java
// Using predefined Locale
Locale locale1 = Locale.FRANCE;

// Constructing a Locale
Locale locale2 = new Locale("en", "UK"); // English, UK version
```
x??

---

#### Formatting Dates and Numbers
Background context: Classes like `DateTimeFormatter` and `NumberFormat` offer overloads that allow you to format dates and numbers according to the rules of a specific locale. This is useful for displaying data in a way that makes sense to users from different regions.

:p How can you use a particular `Locale` with `DateFormat`?
??x
You can use a `Locale` object when creating an instance of `DateFormat`. The following example demonstrates how to create a medium-length date formatter for France and the UK:
```java
Locale frLocale = Locale.FRANCE;
Locale ukLocale = new Locale("en", "UK");

// Creating formatters
DateFormat frDateFormatter = DateFormat.getDateInstance(DateFormat.MEDIUM, frLocale);
DateFormat ukDateFormatter = DateFormat.getDateInstance(DateFormat.MEDIUM, ukLocale);

// Using the formatters to format a date (example)
LocalDateTime now = LocalDateTime.now();
String frenchDate = frDateFormatter.format(now);
String ukDate = ukDateFormatter.format(now);
```
x??

---

#### Application of UseLocales
Background context: The `UseLocales` class demonstrates how to use different locales based on user settings or command-line arguments. This is useful for applications that need to adapt to the language and regional preferences of their users.

:p What does the `UseLocales` class do?
??x
The `UseLocales` class provides a way to format dates using different locales based on the user's OS settings, command-line arguments (`-Duser.lang=` or `-Duser.region=`), or predefined values. It demonstrates how to use `Locale` objects with formatting classes.
```java
package i18n;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.FormatStyle;
import java.util.Locale;

public class UseLocales {
    public static void main(String[] args) {
        // Example of using locales to format dates
        Locale frLocale = Locale.FRANCE;
        Locale ukLocale = new Locale("en", "UK");

        DateFormat frDateFormatter = DateFormat.getDateInstance(DateFormat.MEDIUM, frLocale);
        DateFormat ukDateFormatter = DateFormat.getDateInstance(DateFormat.MEDIUM, ukLocale);

        LocalDateTime now = LocalDateTime.now();
        String frenchDate = frDateFormatter.format(now);
        String ukDate = ukDateFormatter.format(now);

        System.out.println("French Date: " + frenchDate);
        System.out.println("UK Date: " + ukDate);
    }
}
```
x??

---

#### Locale and DateTime Formatting
Background context: This section discusses how to use Java's `Locale` and `DateTimeFormatter` for date formatting based on different locales. The example demonstrates creating formatters specific to French (France) and English (UK) locales, then prints the current date in each of these formats.

:p What is a `Locale` object used for in this context?
??x
A `Locale` object is used to define a language and country combination, which affects how text is displayed. In the provided example, it's used to customize date formatting according to French (France) and English (UK) preferences.
x??

---
#### DateTimeFormatter with Locale
Background context: The `DateTimeFormatter.ofLocalizedDateTime()` method is used to create localized date-time formatters based on a specific format style (`FormatStyle.MEDIUM` in this case). These formatters can be further customized by specifying the desired locale.

:p How does one create a `DateTimeFormatter` that formats dates according to a specified locale?
??x
To create a `DateTimeFormatter` that formats dates according to a specified locale, you first use the `ofLocalizedDateTime()` method with the desired format style. Then, you can call `.localizedBy(locale)` on this formatter to apply the specified locale.

```java
DateTimeFormatter defaultDateFormatter = DateTimeFormatter.ofLocalizedDateTime(FormatStyle.MEDIUM);
DateTimeFormatter frDateFormatter = defaultDateFormatter.localizedBy(Locale.FRANCE);
DateTimeFormatter ukDateFormatter = defaultDateFormatter.localizedBy(new Locale("en", "UK"));
```

x??

---
#### LocalDateTime and Current Time
Background context: The `LocalDateTime.now()` method returns the current date-time in the system's time zone. This example uses this method to get the current date and time, which is then formatted using different formatters based on various locales.

:p How does one obtain the current date and time in Java?
??x
To obtain the current date and time in Java, you use `LocalDateTime.now()`. This method returns a `LocalDateTime` object representing the current date-time without any timezone information (i.e., it is not aware of changes to the system's clock or daylight saving time).

```java
LocalDateTime now = LocalDateTime.now();
```

x??

---
#### Resource Bundle Overview
Background context: A resource bundle in Java is a collection of names and values. For internationalization, you can use `ResourceBundle` subclasses or create plain text properties files that can be loaded using `ResourceBundle.getBundle()`. This approach allows for user customization and adaptation to local variations.

:p What are the advantages of using Properties files over a `ResourceBundle` subclass?
??x
The main advantage of using Properties files over a `ResourceBundle` subclass is simplicity and ease of use. Text-based properties files can be edited directly by users, allowing for customization of text in applications according to local preferences or dialects not covered by default.

For example, if the application supports multiple languages but does not cover all regional variations, users can modify the Properties file to adjust the wording as needed.

x??

---

#### Resource Bundle Naming Convention
Background context: When working with internationalization (i18n) and localization (l10n) in Java, resource bundles are used to store language-specific strings. It is important to ensure that your resource bundle file names do not conflict with your class names.
:p What should you avoid when naming resource bundle files?
??x
You should avoid using the same name as any of your Java classes for your resource bundle files. The `ResourceBundle` constructs a class dynamically with the same name as the resource file, which can lead to naming conflicts and errors in your application.
x??

---

#### Creating Properties Files
Background context: To support multiple languages in your Java applications, you need to create properties files that contain localized strings. These files are typically named after the language (e.g., `en.properties` for English).
:p How do you create a default properties file for menu items?
??x
You can create a properties file like this:
```
# Default Menu properties
file.label=File Menu
file.new.label=New File
file.new.key=N
file.save.label=Save
file.new.key=S
```
This defines the strings used in your application's menus. Note that each key-value pair represents a localized string, where the key is typically an identifier for the string.
x??

---

#### Formatting Text with Fmt Program
Background context: The `Fmt` program is a simple text formatter designed to output text from a file, formatting it neatly within a specified column width (72 in this case). This was typical of older computing platforms before advanced word processors and desktop publishing tools.
:p What does the `Fmt` program do?
??x
The `Fmt` program reads words from a file, formats them such that they fit within the specified column width (72 characters), and outputs the formatted text. It discards any line breaks present in the original input and uses `println()` to start new lines when necessary.
x??

---

#### Fmt Program Code Analysis
Background context: The `Fmt` program is a simple example of how you can read from files, format text, and write it out using Java's standard I/O classes. It handles different input sources (standard input, files, streams) through various constructors.
:p What are the constructors in the `Fmt` class?
??x
The `Fmt` class has several constructors to handle different input scenarios:
```java
public Fmt(BufferedReader  inFile, PrintWriter  outFile);
public Fmt(PrintWriter  out);
public Fmt(BufferedReader  file) throws IOException;
public Fmt(String fname) throws IOException;
public Fmt(InputStream  file) throws IOException;
```
Each constructor sets up the necessary `BufferedReader` and `PrintWriter` instances to read from and write to different sources.
x??

---

#### Formatting Lines in Fmt Program
Background context: The core functionality of the `Fmt` program is implemented in the `format()` method, which processes lines from a file or stream, ensuring they fit within the specified column width.
:p How does the `format()` method work?
??x
The `format()` method works by:
1. Reading lines from an input source (`Stream<String>`).
2. Using a `StringBuilder` to construct formatted lines that do not exceed the column width (72 characters in this case).
3. Outputting each line when it reaches the end of the current line or encounters the end of the file.
Here is the logic in detail:
```java
public static void format(Stream<String> s, PrintWriter  out) {
    StringBuilder  outBuf = new StringBuilder();
    for (String line : s) {
        if (outBuf.length() + line.length() + 1 > COLWIDTH) { // Check if adding this line exceeds the column width
            out.println(outBuf.toString()); // Output the current buffer and clear it
            outBuf.setLength(0); // Reset the buffer for the next line
        }
        outBuf.append(line).append(" "); // Append the line to the buffer with a space
    }
    if (outBuf.length() > 0) { // Output any remaining text in the buffer
        out.println(outBuf.toString());
    }
}
```
x??

---

