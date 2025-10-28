# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 63)

**Starting Chapter:** See Also

---

#### JSON Parsing Using org.json API
Background context: The `org.json` package is used for parsing and creating JSON objects in Java. It's a simple library that provides basic functionality without the complexity of higher-level libraries like Jackson. This makes it suitable for simpler use cases but not as powerful or flexible.

:p What is the purpose of using `org.json` API for JSON parsing?
??x
The purpose of using `org.json` API is to parse and work with JSON data in a straightforward manner, making it easier to handle basic JSON structures without the need for complex configurations. This library is particularly useful when you want to quickly read and manipulate simple JSON objects.

It can be used in Android development as well because it has no external dependencies and doesn't require any additional setup.
x??

---
#### Reading JSON Data with org.json
Background context: The `org.json` API allows for basic parsing of JSON data into Java objects, making it easier to work with JSON content. This involves creating a `JSONObject`, which represents the root object in the JSON structure.

:p How does one read and print specific fields from a JSON file using `org.json`?
??x
To read and print specific fields from a JSON file using `org.json`, you first need to parse the input stream into a `JSONObject`. Then, you can retrieve values by their keys. Here's an example:

```java
import org.json.*;

public class SoftwareParseOrgJson {
    final static String FILE_NAME = "/json/softwareinfo.json";

    public static void main(String[] args) throws Exception {
        InputStream jsonInput = SoftwareParseOrgJson.class.getResourceAsStream(FILE_NAME);
        if (jsonInput == null) {
            throw new NullPointerException("can't find " + FILE_NAME);
        }
        
        // Create a JSONObject from the input stream
        JSONObject obj = new JSONObject(new JSONTokener(jsonInput));

        // Print individual fields
        System.out.println("Software Name: " + obj.getString("name"));
        System.out.println("Version: " + obj.getString("version"));
        System.out.println("Description: " + obj.getString("description"));
        System.out.println("Class: " + obj.getString("className"));

        // Retrieve and print the array of contributors
        JSONArray contribs = obj.getJSONArray("contributors");
        for (int i = 0; i < contribs.length(); i++) {
            System.out.println("Contributor Name: " + contribs.get(i));
        }
    }
}
```
x??

---
#### Handling JSON Arrays with org.json
Background context: When working with JSON data that includes arrays, `org.json` provides a way to parse and iterate over the array elements. The `JSONArray` class is used to handle these arrays.

:p How does one retrieve an array of contributors from a JSON object using `org.json`?
??x
To retrieve an array of contributors from a JSON object using `org.json`, you first need to get the `JSONArray` instance corresponding to the "contributors" key. Then, you can iterate over the elements in the array.

Here is how it's done:

```java
// Assuming obj is a JSONObject representing the parsed JSON data
JSONArray contribs = obj.getJSONArray("contributors");
for (int i = 0; i < contribs.length(); i++) {
    System.out.println("Contributor Name: " + contribs.get(i));
}
```

This code snippet demonstrates how to get the `JSONArray` from a specific key in a JSON object and iterate over its elements.
x??

---
#### Handling Non-Iterable JSONArray
Background context: The `org.json.JSONArray` class does not implement the `Iterable` interface, which means you cannot use Java's enhanced for loop (`for-each`) to directly iterate over its elements. Instead, you need to use traditional indexing or other methods provided by the `JSONArray`.

:p Why can't a `JSONArray` be used with a forEach loop in `org.json`?
??x
A `JSONArray` from `org.json` cannot be used with a `forEach` loop because it does not implement the `Iterable` interface. This means that you need to use traditional methods like indexing to iterate over its elements.

Here is an example of how to manually iterate through the array:

```java
// Assuming contribs is a JSONArray from "contributors" key in JSON object
for (int i = 0; i < contribs.length(); i++) {
    System.out.println("Contributor Name: " + contribs.get(i));
}
```

This code uses an index-based approach to iterate through the elements of the `JSONArray`.
x??

---

#### JSON Parsing and Writing Using org.json Library
The `org.json` library provides a simple API for handling JSON data. It includes classes like `JSONObject` and `JSONArray`, which can be used to create, manipulate, and convert JSON objects and arrays into strings.
:p How does the `org.json` library facilitate JSON operations?
??x
The `org.json` library simplifies working with JSON by offering methods such as `put()`, `toString()`, and others. For instance, a `JSONObject` can be created and populated using method chaining:
```java
JSONObject jsonObject = new JSONObject();
jsonObject.put("Name", "robinParse").put("Version", "1.2.3").put("Class", "RobinParse");
```
The `toString()` method converts the JSON object into a string representation.
x??

---

#### Writing JSON Using org.json Library
`JSONObject` and `JSONArray` classes in the `org.json` library provide methods to create, manipulate, and convert JSON data. The `toString()` method can be used to get a correctly formatted JSON string from a `JSONObject`.
:p How does one use `JSONObject` to write JSON data?
??x
To write JSON data using `JSONObject`, you instantiate it and use the `put()` method for adding key-value pairs, followed by calling `toString()` to get the JSON string.
```java
public class WriteOrgJson {
    public static void main(String[] args) {
        JSONObject jsonObject = new JSONObject();
        jsonObject.put("Name", "robinParse")
                  .put("Version", "1.2.3")
                  .put("Class", "RobinParse");
        String printable = jsonObject.toString();
        System.out.println(printable);
    }
}
```
This outputs: `{"Name":"robinParse","Class":"RobinParse","Version":"1.2.3"}`
x??

---

#### JSON-B Introduction
JSON-B (JSON Binding) is a new Java standard designed for reading and writing JSON data. It simplifies the process of converting between JSON strings and Java objects, using annotations to customize mappings.
:p What does JSON-B offer in terms of functionality?
??x
JSON-B provides a standardized way to read from and write to JSON strings by converting them to and from Java POJOs (Plain Old Java Objects). This is done without the need for extensive annotations on classes, but customizations are possible via the API.
```java
public class ReadWriteJsonB {
    public static void main(String[] args) throws IOException {
        Jsonb jsonb = JsonbBuilder.create();
        
        String jsonInput = "{\"id\":0,\"firstName\":\"Robin\",\"lastName\":\"Williams\"}";
        Person rw = jsonb.fromJson(jsonInput, Person.class);
        System.out.println(rw);
        
        String result = jsonb.toJson(rw);
        System.out.println(result);
    }
}
```
This code reads a JSON string and converts it to a `Person` object, then writes the object back to JSON.
x??

---

#### Customizing JSON-B Output
JSON-B can be customized using annotations. For example, redundant properties like `fullName`, which is derived from concatenating first name and last name, can be excluded from JSON output by annotating the getter method with `@JsonbTransient`.
:p How does customizing JSON output work in JSON-B?
??x
Customization in JSON-B involves adding specific annotations to fields or methods. To exclude a redundant property like `fullName`, annotate its getter method as follows:
```java
public class Person {
    // other properties and getters/setters

    @JsonbTransient
    public String getFullName() {
        return firstName + " " + lastName;
    }
}
```
This annotation tells JSON-B to ignore the `getFullName()` property when converting objects to JSON.
x??

---

---

#### JSON Pointer Syntax and Usage
JSON Pointer is a standard for identifying elements within a JSON document, inspired by XPath but simpler due to the structure of JSON. It uses a string starting with "/" followed by names or indices to identify elements.
:p What is the syntax used for JSON Pointers?
??x
The syntax for JSON Pointers starts with "/", followed by the name or index of the element within the JSON document. For example, "/firstName" identifies the "firstName" element, and "/roles/1" identifies the second item in the "roles" array.
```java
// Example of creating a JSON Pointer
JsonPointer jsonPointer = Json.createPointer("/roles/1");
```
x??

---

#### Extracting Elements from a JSON Document Using JSON Pointer
To extract specific elements from a JSON document, you can use `Json.createPointer` to create a pointer and then call `getValue` on the pointer object. This method retrieves the value at the specified path in the JSON structure.
:p How do you use JSON Pointers to retrieve values from a JSON document?
??x
To use JSON Pointers, first, you need to create a pointer using `Json.createPointer`. Then, call `getValue` on this pointer and pass the root of your JSON structure. This method returns the value at the specified path in the JSON document.
```java
// Example code snippet
String jsonPerson = "{\"firstName\":\"Robin\",\"lastName\":\"Williams\", \"age\": 63, \"id\":0, \"roles\":[\"Mork\", \"Mrs. Doubtfire\", \"Patch Adams\"]}";
JsonReader rdr = Json.createReader(new StringReader(jsonPerson));
JsonStructure jsonStr = rdr.read();
rdr.close();

// Creating and using JSON Pointer to get "firstName"
JsonPointer jsonPointer = Json.createPointer("/firstName");
JsonString jsonString = (JsonString)jsonPointer.getValue(jsonStr);
String firstName = jsonString.getString();
```
x??

---

#### Checking for Element Existence with containsValue
When using JSON Pointers, you might need to check if a particular element exists before retrieving its value. The method `containsValue` can be used to verify the existence of an element at the specified path.
:p How do you check if a specific element exists in a JSON document?
??x
To check if a specific element exists in a JSON document, use the `containsValue` method on the `JsonPointer`. This method returns true if the pointer matches any value within the JSON structure, indicating that the element is present.
```java
// Example code snippet to check for existence of "roles" array
JsonPointer jsonPointer = Json.createPointer("/roles");
boolean exists = jsonPointer.containsValue(jsonStr);
if (exists) {
    // Element exists; proceed with further processing
}
```
x??

---

#### Handling Different Data Types in JSON Pointers
When using JSON Pointers, the data type of the retrieved value can vary depending on the structure of your JSON document. For example, numbers might be represented as `JsonNumber`, and strings as `JsonString`.
:p What are some different data types you might encounter when using JSON Pointers?
??x
You might encounter various data types when using JSON Pointers, such as `JsonNumber` for numerical values and `JsonString` for string values. For instance, if the age is 63 in an example JSON document, `getValue("/age")` would return a `JsonNumber`. However, if you change it to 63.5, it might be represented as `JsonDecimalNumber`.
```java
// Example code snippet handling different data types
JsonPointer jsonPointer = Json.createPointer("/age");
JsonValue value = jsonPointer.getValue(jsonStr);
if (value instanceof JsonNumber) {
    JsonNumber num = (JsonNumber)value;
    System.out.println(num + "; a " + num.getClass().getName());
}
```
x??

---

