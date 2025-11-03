# Flashcards: Pro-ASPNET-Core-7_processed (Part 61)

**Starting Chapter:** 5.6 Using object and collection initializers

---

---
#### Using String Interpolation in C#
Background context: C# supports string interpolation, which allows for more readable and concise string creation by embedding expressions directly within strings. This feature can improve performance compared to concatenation methods.

:p How does string interpolation work in C#?
??x
String interpolation works by prefixing a string with the `$` character, allowing you to embed expressions inside curly braces `{}` that will be evaluated at runtime and inserted into the resulting string. This approach is particularly useful for creating formatted strings that include values from variables or constants.

For example:
```csharp
string name = "Alice";
int age = 25;
string greeting = $"Hello, my name is {name} and I am {age} years old.";
```
In this case, `greeting` will be `"Hello, my name is Alice and I am 25 years old."`.

x??
---
#### Using Object Initializers in C#
Background context: Object initializers are a syntactic sugar feature that allows you to initialize properties of an object during the object's creation. This makes the code more concise and easier to read compared to setting each property after creating the object.

:p What is the difference between using an object initializer versus setting properties separately?
??x
Using an object initializer simplifies the process by allowing you to specify the values of multiple properties in a single line when creating an object. For example, instead of:
```csharp
Product kayak = new Product();
kayak.Name = "Kayak";
kayak.Price = 275M;
```
You can use an object initializer like this:
```csharp
Product kayak = new Product { Name = "Kayak", Price = 275M };
```
This approach is more concise and readable, making the code easier to maintain.

x??
---
#### Using Collection Initializers in C#
Background context: Collection initializers are a syntactic sugar feature that allows you to create and initialize collections (such as arrays or lists) with their contents specified inline. This makes it more convenient to define collections within method calls without needing separate statements for initialization.

:p How do collection initializers work?
??x
Collection initializers allow the creation of an array or list along with its elements in a single statement. For example, instead of:
```csharp
string[] names = new string[3];
names[0] = "Bob";
names[1] = "Joe";
names[2] = "Alice";
```
You can use a collection initializer like this:
```csharp
string[] names = new string[] { "Bob", "Joe", "Alice" };
```
This approach is more concise and allows you to define the collection inline within a method call.

x??
---

#### Index Initializer Syntax for Dictionaries

In recent versions of C#, there is a more natural way to initialize collections like dictionaries, using index notation that aligns with how values are retrieved or modified. This approach reduces verbosity and makes code easier to read.

:p How does the latest version of C# allow initializing dictionaries in a more concise manner?
??x
The latest versions of C# support an index initializer syntax for dictionaries, making initialization more readable by using square brackets `[]` instead of multiple lines with `{}`. For example:

```csharp
Dictionary<string, Product> products = new Dictionary<string, Product>
{
    ["Kayak"] = new Product { Name = "Kayak", Price = 275M },
    ["Lifejacket"] = new Product { Name = "Lifejacket", Price = 48.95M }
};
```

This syntax is more natural because it directly uses the index notation, which aligns with how dictionary keys are accessed and modified.

x??

---

#### Target-Typed New Expressions for Dictionary Initialization

Target-typed new expressions simplify the initialization of collections by allowing you to omit the explicit type when using a collection initializer. This can make your code more concise and easier to understand.

:p How does target-typed new expression work in dictionary initialization?
??x
In C#, you can use `new()` as an implicit type for dictionary initializers, making the code less verbose. For example:

```csharp
Dictionary<string, Product> products = new()
{
    ["Kayak"] = new Product { Name = "Kayak", Price = 275M },
    ["Lifejacket"] = new Product { Name = "Lifejacket", Price = 48.95M }
};
```

Here, `new()` is used instead of explicitly specifying the type `Dictionary<string, Product>`. This works because the compiler can infer the dictionary's type based on the elements being added.

x??

---

#### Restarting ASP.NET Core Application

To see changes in your application, you need to restart the ASP.NET Core server. Changes made to code or configuration will not take effect until the application is restarted.

:p How do you restart an ASP.NET Core application?
??x
You can restart an ASP.NET Core application by stopping and then starting it again through the development server interface or by using commands like `dotnet run` in your terminal. For example:

1. Stop the current running instance.
2. Start a new instance of the application.

Alternatively, if you are using Visual Studio, you can simply restart the debugging session from within the IDE.

x??

---

#### Using HTTP Requests to Test Changes

To verify that changes have been applied correctly in your ASP.NET Core application, you need to make HTTP requests to the appropriate endpoints. This is typically done by navigating to the URL or using a tool like Postman.

:p How do you test an ASP.NET Core application change?
??x
Testing changes in an ASP.NET Core application involves making HTTP requests to specific URLs that correspond to your application's routes. For instance, if you have an `Index` action method that returns "Bob Joe Alice" when accessed at `http://localhost:5000`, you would navigate to this URL in a web browser or use a tool like Postman.

For example:

- Navigate to `http://localhost:5000` in your web browser.
- Use a command-line tool such as `curl`:
  ```bash
  curl http://localhost:5000
  ```

x??

---

#### Product and Dictionary Setup

In the provided code, products are being set up using both traditional C# dictionary initialization and more modern index initializer syntax. This setup is common in web applications to pass data from controllers to views.

:p What is the purpose of setting up product dictionaries in a controller?
??x
The purpose of setting up product dictionaries in a controller is to manage and organize product information, which can then be passed to views for display or further processing. In this case, products are defined with names and prices, and these dictionaries might be used to populate dropdowns, lists, or other UI elements.

Example:
```csharp
public class HomeController : Controller {
    public ViewResult Index() {
        Dictionary<string, Product> products = new Dictionary<string, Product>
        {
            ["Kayak"] = new Product { Name = "Kayak", Price = 275M },
            ["Lifejacket"] = new Product { Name = "Lifejacket", Price = 48.95M }
        };
        return View("Index", products.Keys);
    }
}
```

x??

---
#### Pattern Matching in C#
Pattern matching is a powerful feature introduced in recent versions of C# that allows for more readable and concise code compared to traditional switch statements. It uses keywords like `is` and `when` to test object types or specific conditions, simplifying complex conditional logic.

:p What is the `is` keyword used for in pattern matching?
??x
The `is` keyword in C# is used to perform a type check on an object. If the value stored in the variable matches the specified type, it assigns the value to a new variable, allowing subsequent code to use that variable without needing explicit type conversions.

```csharp
if (data[i] is decimal d) {
    total += d;
}
```
x??

---
#### Using `when` Keyword with Pattern Matching
The `when` keyword in C# can be used within pattern matching cases to provide additional conditions. This makes the switch statement more selective, allowing for complex logic within the same case.

:p How does the `when` keyword work in a switch statement?
??x
The `when` keyword in a switch statement allows you to add an additional condition to a case that uses pattern matching. It narrows down which cases will be executed based on the specified conditions, making the code more specific and readable.

```csharp
switch (data[i]) {
    case decimal decimalValue when decimalValue > 0:
        total += decimalValue;
        break;
}
```
x??

---
#### Extension Methods in C#
Extension methods are a feature in C# that allow you to add new methods to existing classes without modifying the original class. They are particularly useful for adding functionality to third-party libraries or classes where direct modification is not possible.

:p What is an extension method and how does it work?
??x
An extension method in C# is a static method defined on a static class that can be called as if it were an instance method of the type being extended. The `this` keyword before the first parameter marks the method as an extension method, allowing you to use methods of a specific type without needing to create instances of that type.

```csharp
public static class MyExtensionMethods {
    public static decimal TotalPrices(this ShoppingCart cartParam) {
        decimal total = 0;
        if (cartParam.Products?.Any() != false) { // Check if Products is not null and has elements
            foreach (Product? prod in cartParam.Products) {
                total += prod?.Price ?? 0; // Safe navigation operator to handle nullable values
            }
        }
        return total;
    }
}
```
x??

---
#### Applying Extension Methods
You can apply extension methods to existing instances of a class without changing the original class definition. This is particularly useful for adding new functionality in scenarios where you do not have access to the source code.

:p How do you use an extension method in your C# application?
??x
To use an extension method, simply call it as if it were a regular instance method on an object of the type being extended. Here’s how you can apply the `TotalPrices` method to a `ShoppingCart` instance:

```csharp
HomeController homeController = new HomeController();
decimal totalValue = homeController.Index().Model[0].Substring(6); // Assuming Index returns a string with "Total: $304.95"
Console.WriteLine($"Total value is {totalValue}");
```
x??

---

