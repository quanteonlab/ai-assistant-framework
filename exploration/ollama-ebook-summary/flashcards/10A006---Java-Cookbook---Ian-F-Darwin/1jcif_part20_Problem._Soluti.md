# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 20)

**Starting Chapter:** Problem. Solution. 4.9 Matching Newlines in Text

---

#### Matching Accented, or Composite, Characters
Composite characters can be entered in various forms. For example, consider the letter 'e' with an acute accent (é). This character might appear as a single Unicode character (\u00e9) or as two separate characters ('e' followed by \u0301).
:p How do you match accented characters regardless of their form?
??x
To match accented characters regardless of their form, you can use the `CANON_EQ` flag when compiling a regular expression pattern. This flag enables canonical matching, treating different decompositions of the same character as equivalent.
```java
Pattern.compile(pattern, Pattern.CANON_EQ);
```
x??

---

#### Case Insensitive Matching
Case-insensitive matching allows for matches without regard to case. For instance, you can match both 'A' and 'a'.
:p How do you enable case-insensitive matching in a regular expression?
??x
To enable case-insensitive matching, use the `CASE_INSENSITIVE` flag when compiling your regular expression pattern.
```java
Pattern.compile(pattern, Pattern.CASE_INSENSITIVE);
```
x??

---

#### Comments and Whitespace Ignoring
Whitespace and comments (from '#' to end-of-line) can be ignored in patterns by using the `COMMENTS` flag. This is useful for readability in complex regexes.
:p How do you ignore whitespace and comments in a regular expression pattern?
??x
To ignore whitespace and comments, use the `COMMENTS` flag when compiling your regular expression pattern.
```java
Pattern.compile(pattern, Pattern.COMMENTS);
```
x??

---

#### Dotall Mode
In dotall mode, the dot (.) matches any character including newline characters. Normally, it only matches non-newline characters.
:p How do you enable dot to match all characters including newlines?
??x
To enable the dot to match any character, including newlines, use the `DOTALL` flag when compiling your regular expression pattern.
```java
Pattern.compile(pattern, Pattern.DOTALL);
```
x??

---

#### Multiline Mode
Multiline mode changes how the start (^) and end ($) of string anchors work. In multiline mode, they match both at the beginning and end of the entire string as well as after each newline within the string.
:p How do you enable multiline matching in a regular expression?
??x
To enable multiline matching, use the `MULTILINE` flag when compiling your regular expression pattern.
```java
Pattern.compile(pattern, Pattern.MULTILINE);
```
x??

---

#### Unicode Case Folding
Unicode-aware case folding makes case-insensitive matching work properly with non-ASCII characters. It ensures that uppercase and lowercase versions of a character are treated as equivalent according to the Unicode standard.
:p How do you enable Unicode-aware case folding in a regular expression?
??x
To enable Unicode-aware case folding, use the `UNICODE_CASE` flag when compiling your regular expression pattern.
```java
Pattern.compile(pattern, Pattern.UNICODE_CASE);
```
x??

---

#### UNIX Lines Mode
In multiline mode with this flag set, only the newline sequence `\n` is valid. This affects how anchors (`^` and `$`) work in patterns.
:p How do you restrict newlines to only be \n in a regular expression?
??x
To restrict newlines to only be `\n`, use the `UNIX_LINES` flag when compiling your regular expression pattern.
```java
Pattern.compile(pattern, Pattern.UNIX_LINES);
```
x??

---

#### Matching Combining Accents Using CANON_EQ
Background context: The `CANON_EQ` flag in Java's `Pattern.compile()` method is used to match characters based on their canonical equivalence. This means that it considers characters with the same visual representation but different internal representations as equivalent.

In this example, we are using `CANON_EQ` to see if combining accents (like an acute accent) are treated the same way as precomposed characters (like é).

:p What is the purpose of the `CANON_EQ` flag in Java's regular expression matching?
??x
The purpose of the `CANON_EQ` flag is to match characters based on their canonical equivalence, meaning that it considers visually identical characters with different internal representations as equivalent. This is useful for matching accented characters where multiple Unicode sequences can represent the same character.

For example:
- The character "é" can be represented in two ways: as a precomposed character (U+00E9) or as a base character 'e' followed by a combining acute accent (U+0301).

Here is how the code snippet works to match these characters using `CANON_EQ`:
```java
String pattStr = "\u00e9gal"; // égal - the pattern we are matching against
String[] input = {
    "\u00e9gal", // égal - this one had better match :-)
    "e\u0301gal", // e + Combining acute accent
    "e\u02cagal", // e + Modifier letter acute accent
    "e'gal", // e + single quote
    "e\u00b4gal" // e + Latin-1 'acute'
};

Pattern pattern = Pattern.compile(pattStr, Pattern.CANON_EQ);
for (int i = 0; i < input.length; i++) {
    if (pattern.matcher(input[i]).matches()) {
        System.out.println(
            pattStr + " matches input " + input[i]
        );
    } else {
        System.out.println(
            pattStr + " does not match input " + input[i]
        );
    }
}
```
x??

---

#### Matching Newlines in Text
Background context: In Java's regular expression API, the newline character (`\n`) by default has no special significance. To match newlines as beginning or end of a line, you can use the `MULTILINE` flag.

The `MULTILINE` flag changes the behavior of regex metacharacters like `^` and `$`, making them match at the start or end of each line instead of just the entire string.

:p How do you match newlines in Java's regular expressions?
??x
To match newlines in Java's regular expressions, you can use either `\n` or `\r` (or both) within your regex pattern. Additionally, using the `MULTILINE` flag will change how `^` and `$` behave.

Here is an example of matching newlines with a simple string:
```java
String input = "I dream of engines more engines, all day long";

String[] patt = {
    "engines.more engines",
    "ines more",
    "engines$"
};

for (int i = 0; i < patt.length; i++) {
    System.out.println("PATTERN " + patt[i]);
    boolean found;
    Pattern p1l = Pattern.compile(patt[i]);
    found = p1l.matcher(input).find();
    System.out.println("DEFAULT match " + found);
}
```

With the `MULTILINE` flag, you can change the behavior of `^` and `$` to match the beginning or end of each line:
```java
Pattern.compile(patt[0], Pattern.MULTILINE); // engines.more engines will now match only at the start of a line
Pattern.compile(patt[2], Pattern.MULTILINE); // engines$ will now match at the end of a line in each segment
```

This allows you to handle multiline text more effectively.
x??

---

#### Understanding Line Terminators and Multiline Matching
Background context: In Unix-based systems, tools like `sed` and `grep` traditionally treat newlines as delimiters for lines. However, Java's regex API treats newline characters as just another character by default.

The `MULTILINE` flag in the `Pattern.compile()` method allows you to change this behavior so that `^` matches at the start of each line (or immediately after a newline) and `$` matches at the end of each line (or immediately before a newline).

:p How does the `MULTILINE` flag affect the behavior of regex metacharacters like ^ and $?
??x
The `MULTILINE` flag in Java's `Pattern.compile()` method changes how the regex metacharacters `^` and `$` behave. By default, these characters match only at the start or end of the entire input string.

With the `MULTILINE` flag:
- `^` matches at the beginning of each line (or immediately after a newline).
- `$` matches at the end of each line (or immediately before a newline).

For example, consider this pattern and its usage:

```java
String input = "I dream\nof engines more\engines, all day long";

Pattern.compile("engines", Pattern.MULTILINE); // This will match 'engines' in both lines

Matcher matcher = Pattern.compile("engines", Pattern.MULTILINE).matcher(input);
if (matcher.find()) {
    System.out.println("Match found: " + matcher.group());
}
```

In this case, the pattern `engines` matches at the start of each line. Without the `MULTILINE` flag, it would only match the first occurrence as a whole string.

Additionally, you can use `\n`, `\r`, or `\r\n` directly in your patterns to explicitly match newline characters.
x??

---

#### Regular Expression Patterns and Multiline Matching
Background context: In Java, regular expression patterns can be compiled with different flags to modify their behavior. The `Pattern.DOTALL` and `Pattern.MULTILINE` flags are used to adjust how certain characters like `.` (dot) and the start/end of line anchors (`^`, `$`) behave within a pattern.

:p How does the `Pattern.DOTALL` flag affect regular expression patterns in Java?
??x
The `Pattern.DOTALL` flag modifies the behavior of the dot character (`.`) so that it matches any character, including newline characters. This is useful when you want to match across multiple lines without having to explicitly include newline characters.

```java
Pattern pml = Pattern.compile(patt[i], Pattern.DOTALL | Pattern.MULTILINE);
```
x??

---

#### Multiline Matching in Regular Expressions
Background context: The `Pattern.MULTILINE` flag changes the behavior of anchors (`^`, `$`) so that they match at the beginning and end of each line, not just the entire string. This is particularly useful when you need to perform operations on multi-line strings.

:p How does the `Pattern.MULTILINE` flag affect regular expression patterns in Java?
??x
The `Pattern.MULTILINE` flag modifies the behavior of anchors such as `^` and `$`, making them match the start and end of each line, respectively. This is useful when you are working with multi-line strings and need to ensure that your pattern matches correctly within each line.

```java
Pattern pml = Pattern.compile(patt[i], Pattern.DOTALL | Pattern.MULTILINE);
```
x??

---

#### Apache Logfile Parsing Example
Background context: The provided code snippet demonstrates how to parse an Apache logfile entry using a complex regular expression. This example showcases the use of various regex features, such as nongreedy quantifiers and capturing groups.

:p What is the role of the regular expression in parsing Apache logfiles?
??x
The regular expression plays a crucial role in extracting specific fields from Apache logfile entries. It helps in breaking down the log entry into meaningful components like IP address, user name, date/time, request, response code, bytes sent, referer URL, and user-agent string.

```java
final static String logEntryPattern = "^([\\d.]+) (\\S+) (\\S+) \$$([\\w:/]+\\s[+-]\\d{4})\$$ \"(.+?)\" (\\d{3}) (\\d+) \"([^\\"]+)\" \"([^\\"]+)\"";
```
x??

---

#### Nongreedy Quantifiers in Regular Expressions
Background context: In the provided regex, the `.+?` quantifier is used to match a quoted string non-greedily. This ensures that it matches as little as possible while still matching the entire quoted string.

:p What does the `.+?` quantifier do in regular expressions?
??x
The `.+?` quantifier is a nongreedy version of `.+`, which matches one or more occurrences of any character but stops as soon as it finds the next match, ensuring that it matches the minimum amount necessary.

```java
\"(.+?)\"
```
x??

---

#### Extracting Fields from Logfile Entries
Background context: The code snippet demonstrates how to extract various fields such as IP address, request, referrer URL, and browser version using the regular expression provided. It highlights the use of capturing groups and conditional checks.

:p How does the code handle extracting different fields from log entries?
??x
The code uses a `Pattern` and `Matcher` object to parse the log entry line. It extracts various fields such as IP address, user name, date/time, request, response code, bytes sent, referer URL, and user-agent string using capturing groups in the regular expression.

```java
if (matcher.matches() || LogParseInfo.MIN_FIELDS > matcher.groupCount()) {
    System.err.println("Bad log entry (or problem with regex):");
    System.err.println(logEntryLine);
    return;
}
```
x??

---

