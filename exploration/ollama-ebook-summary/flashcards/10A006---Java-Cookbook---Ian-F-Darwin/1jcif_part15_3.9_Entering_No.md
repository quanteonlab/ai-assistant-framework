# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 15)

**Starting Chapter:** 3.9 Entering Nonprintable Characters. Problem. Solution. 3.10 Trimming Blanks from the End of a String

---

#### Using Java String Escapes
Background context: In Java, string literals can contain nonprintable characters using specific escape sequences. These sequences are denoted by a backslash (\) followed by one or more characters that represent different types of control characters and special symbols.

If applicable, add code examples with explanations:
```java
public class StringEscapes {
    public static void main(String[] argv) {
        // Example usage of string escapes
        System.out.println("An alarm entered in Octal: \007");
        System.out.println("A tab key: \t(what comes after)");
        System.out.println("A newline:  (what comes after)");
        System.out.println("A Unicode character: \u0207");
        System.out.println("A backslash character: \\");
    }
}
```

:p What are Java string escapes used for?
??x
Java string escapes are used to include nonprintable characters, special symbols, and Unicode characters within a string literal.
x??

---
#### Tab Character in String Escapes
Background context: The tab character can be included in strings by using the escape sequence \t.

:p How do you represent a tab character in a Java string?
??x
To represent a tab character in a Java string, use the escape sequence \t.
x??

---
#### Linefeed in String Escapes
Background context: The linefeed (Unix newline) can be included in strings by using the escape sequence \n. Note that for platform-specific newlines, you should use System.getProperty("line.separator").

:p How do you represent a linefeed in a Java string?
??x
To represent a linefeed in a Java string, use the escape sequence \n.
x??

---
#### Carriage Return in String Escapes
Background context: The carriage return can be included in strings by using the escape sequence \r.

:p How do you represent a carriage return in a Java string?
??x
To represent a carriage return in a Java string, use the escape sequence \r.
x??

---
#### Form Feed in String Escapes
Background context: The form feed character can be included in strings by using the escape sequence \f.

:p How do you represent a form feed in a Java string?
??x
To represent a form feed in a Java string, use the escape sequence \f.
x??

---
#### Backspace in String Escapes
Background context: The backspace character can be included in strings by using the escape sequence \b.

:p How do you represent a backspace in a Java string?
??x
To represent a backspace in a Java string, use the escape sequence \b.
x??

---
#### Single Quote in String Escapes
Background context: Single quotes (') can be included in strings by using the escape sequence \'.

:p How do you include a single quote within a Java string literal?
??x
To include a single quote within a Java string literal, use the escape sequence \'.
x??

---
#### Double Quote in String Escapes
Background context: Double quotes ("") are used to delimit strings. To include a double quote inside a string, you can use the escape sequence \".

:p How do you include a double quote within a Java string?
??x
To include a double quote within a Java string, use the escape sequence \".
x??

---
#### Unicode Character in String Escapes
Background context: Unicode characters can be included in strings by using the escape sequence \u followed by four hexadecimal digits. The Unicode character is then displayed when the string is printed.

:p How do you include a Unicode character within a Java string?
??x
To include a Unicode character within a Java string, use the escape sequence \u followed by four hexadecimal digits.
x??

---
#### Octal Character in String Escapes
Background context: The octal character can be included in strings using the escape sequence +\+NNN where NNN are the three digits of the octal number.

:p How do you include an octal character within a Java string?
??x
To include an octal character within a Java string, use the escape sequence +\+NNN, where NNN are the three digits of the octal number.
x??

---
#### Backslash Character in String Escapes
Background context: The backslash character (\) itself can be included in strings by using the double escape sequence \\.

:p How do you include a backslash within a Java string?
??x
To include a backslash within a Java string, use the double escape sequence \\.
x??

---

#### String Trimming and Comparison
Background context: In Java, strings often contain leading or trailing spaces which can affect various operations. The `trim()` method is useful for removing these spaces. This concept is particularly important when dealing with user input where whitespace might be inconsistently entered.

:p What does the `trim()` method do in a string?
??x
The `trim()` method removes any leading and trailing whitespace characters from a string. This means it will remove spaces, tabs, and newlines at the start and end of the string but leaves the rest of the string unchanged.
```java
String str = "   Hello World!    ";
str = str.trim(); // str now equals "Hello World!"
```
x??

---
#### Using `trim()` in Code
Background context: In the provided code, `trim()` is used to remove leading and trailing spaces from lines of Java source code before comparing them with special marks.

:p How does `trim()` help in processing Java source code?
??x
`trim()` helps by ensuring that any extraneous spaces at the beginning or end of a line are removed, making it easier to accurately compare strings without being affected by whitespace. This is crucial for tasks like identifying specific comments or markers within the code.
```java
String inputLine = "    //+";
inputLine = inputLine.trim();
// Now inputLine equals "//+" which can be compared directly with START_MARK
```
x??

---
#### String Comparison Logic
Background context: The provided Java class `GetMark` includes logic for comparing lines of Java source code to specific marks, using the `trim()` method to ensure accurate comparisons.

:p How does the `trim()` and `equals` methods work together in the `GetMark` class?
??x
The `trim()` method is used to remove any leading or trailing whitespace from a line before comparing it with another string using the `equals` method. This ensures that only the actual content of the line is compared, regardless of any extra spaces.

For example:
```java
String inputLine = "    //+";
inputLine = inputLine.trim();
boolean matchesStartMark = inputLine.equals("//+");
```
This ensures a precise match without being affected by extraneous whitespace.
x??

---
#### Handling `strip()` Methods
Background context: While not directly used in the provided code, it's important to know that Java also provides other methods like `strip()`, `stripLeading()`, and `stripTrailing()` which can be useful for different scenarios involving whitespace manipulation.

:p What are some of the other string trimming methods available in Java?
??x
Java offers several string trimming methods:
- `strip()`: Removes all leading and trailing white space.
- `stripLeading()`: Removes only leading white space.
- `stripTrailing()`: Removes only trailing white space.
These can be useful depending on the specific whitespace manipulation needs.

For example, to remove just leading spaces:
```java
String str = "   Hello World!    ";
str = str.stripLeading(); // str now equals "Hello World!    "
```
x??

---
#### Conditional Logic with `trim()`
Background context: The provided code demonstrates how to use the `trim()` method in a conditional statement to check for specific marks and process lines of Java source code accordingly.

:p How is the `trim()` method used within the `process` method?
??x
The `trim()` method is used within the `process` method to remove any leading or trailing spaces from each line before comparing it with predefined start and end marks. This ensures that the comparison logic works correctly regardless of how many spaces are entered by the user.

For example:
```java
String inputLine = "    //+";
inputLine = inputLine.trim();
if (inputLine.equals("//+")) {
    // Handle start mark
}
```
This approach guarantees accurate processing and comparison.
x??

---

#### ResourceBundle and Localization
Background context: In Java, `ResourceBundle` is used for internationalization (I18N) and localization (L10N). It allows fetching localized resources such as strings based on the user's locale. This mechanism ensures that your application can adapt to different languages and cultures without changing the underlying code.

:p What is ResourceBundle used for in Java applications?
??x
ResourceBundle is used for internationalization and localization, allowing developers to fetch text strings specific to a given language or culture by using resource bundles.
x??

---
#### Creating I18N Strings with ResourceBundle
Background context: To create internationalized strings, you first need a `ResourceBundle` which contains name-value pairs. Each name corresponds to the ID of a string that needs translation, and each value is the localized text.

:p How do you get a ResourceBundle for your application?
??x
You get a ResourceBundle by calling `ResourceBundle.getBundle("bundleName")`. This method searches for files named according to the locale settings.
x??

---
#### Using I18N Strings in Code
Background context: Once you have a `ResourceBundle`, you can use it to fetch localized strings instead of hardcoding them into your application.

:p How do you fetch a string from a ResourceBundle?
??x
You fetch a string by calling `rb.getString("resourceKey")` where `rb` is an instance of `ResourceBundle` and "resourceKey" is the ID assigned to the resource. If the key is not found, it returns a default value.
x??

---
#### Creating I18N UI Components
Background context: To create internationalized user interface components like buttons or labels, you can use a utility class that abstracts the ResourceBundle fetching process.

:p How do you create an I18N JButton in Java?
??x
You can create an I18N JButton by using a convenience method from an `I18NUtil` class. Here's an example:
```java
JButton exitButton = I18NUtil.getButton("exit.label", "Exit");
```
This method fetches the localized string for "exit.label" and uses it to create the button.
x??

---
#### Locale Configuration in JSF
Background context: In JavaServer Faces (JSF), you can configure locales using `faces-config.xml`. This allows your application to support multiple languages and adapt messages based on user settings.

:p How do you configure locales for a JSF application?
??x
You configure locales by setting them in the `faces-config.xml` file. Here’s an example configuration:
```xml
<application>
    <locale-config>
        <default-locale>en</default-locale>
        <supported-locale>es</supported-locale>
        <supported-locale>fr</supported-locale>
    </locale-config>
    <resource-bundle>
        <base-name>resources</base-name>
        <var>msg</var>
    </resource-bundle>
</application>
```
This sets the default locale to English and supports Spanish and French. Strings are then accessed using the `msg` variable in facelets files.
x??

---
#### Loading ResourceBundle Files
Background context: The name of the ResourceBundle file is typically composed of the base name, an underscore, the language code, another underscore (if a country or variant is specified), and ".properties".

:p What is the format for naming ResourceBundle files?
??x
The naming format for ResourceBundle files follows this pattern:
```
bundleBaseName_language_country.properties
```
For example, `Menus_en_US.properties` represents a resource bundle named "Menues" in English (United States).
x??

---
#### Setting Locale at Runtime
Background context: You can set the locale programmatically by using system properties or environment variables. This is useful for testing different languages during development.

:p How do you set the locale at runtime in Java?
??x
You set the locale at runtime by using the `-D` option on the command line:
```
java -Duser.language=en i18n.Browser
```
This runs the Java program named "Browser" in the English locale.
x??

---
#### Different Locale Examples
Background context: Here are examples of how different locales map to file names and environment settings.

:p What are some examples of property filenames for different locales?
??x
For various locales, here are some examples:
- Default locale: `Menus.properties`
- Swedish: `Menus_sv.properties`
- Spanish: `Menus_es.properties`
- French-Canadian: `Menus_fr_CA.properties`

These files contain localized strings for the respective languages and regions.
x??

---

