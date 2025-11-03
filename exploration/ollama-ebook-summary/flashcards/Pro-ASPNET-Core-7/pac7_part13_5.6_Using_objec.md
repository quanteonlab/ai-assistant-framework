# Flashcards: Pro-ASPNET-Core-7_processed (Part 13)

**Starting Chapter:** 5.6 Using object and collection initializers

---

#### Suppressing Warnings using #pragma Directive
Background context: The C# compiler may issue warnings that can sometimes be ignored or suppressed for specific parts of the code. This is useful when working with nullable reference types and avoiding false positives from the compiler.

:p How does one suppress a specific warning in C#?
??x
To suppress a specific warning, you use the `#pragma` directive followed by the warning number. In this case, to suppress the CS8602 warning (which is related to possible null reference errors), you would add `#pragma warning disable CS8602`. This can be done before the code where the warning occurs.

```csharp
#pragma warning disable CS8602
// Code that might trigger a null reference warning
```
x??

---

#### Using String Interpolation in C#
Background context: C# supports string interpolation, which allows you to embed expressions inside string literals. This makes it easier to create formatted strings by directly using variables and constants within the string template.

:p How is string interpolation used in C#?
??x
String interpolation in C# uses the `$` symbol before a string literal to denote that the string can contain embedded expressions, which are evaluated at runtime and inserted into the string. The syntax involves placing variable or constant names inside `{}` within the string.

```csharp
Product?[] products = Product.GetProducts();
string result = $"Name: {products[0]?.Name}, Price: {products[0]?.Price}";
```
x??

---

#### Example of String Interpolation in HomeController.cs
Background context: The example shows how to use string interpolation within the `Index` method of the `HomeController`. It demonstrates formatting a string that includes product information.

:p What does the following code snippet do?
```csharp
Product?[] products = Product.GetProducts();
return View(new string[] {
    $"Name: {products[0]?.Name}, Price: { products[0]?.Price }"
});
```
??x
The code uses string interpolation to format a string that includes the name and price of the first product from an array. If `products[0]` is null, the question mark after it (`?.`) ensures that no null reference exception occurs by returning null for properties.

```csharp
Product?[] products = Product.GetProducts(); // Assume this returns some products
string[] viewModels = new string[] {
    $"Name: {products[0]?.Name}, Price: {products[0]?.Price}" // Interpolated string with null safety
};
return View(viewModels);
```
x??

---

#### Object Initializers in C#
Background context: Object initializers allow you to create and initialize an object using a more concise syntax. This is particularly useful when creating objects and setting their properties at the same time, reducing the number of lines required.

:p How does one use object initialization in C#?
??x
Object initialization allows you to initialize an object and set its property values in a single line without calling the constructor explicitly. You can specify property names and their corresponding values using curly braces `{}`.

```csharp
Product kayak = new Product {
    Name = "Kayak",
    Price = 275M
};
```
x??

---

#### Collection Initializers in C#
Background context: Collection initializers allow you to create a collection and populate it with elements in a single statement, making the code more readable and concise. This is useful for creating arrays or lists without needing separate statements for each element.

:p How does one use a collection initializer in C#?
??x
A collection initializer allows you to specify the contents of an array or list when initializing it. Instead of separately setting each element, you can provide all elements within curly braces `{}`.

```csharp
string[] names = new string[] {
    "Bob",
    "Joe",
    "Alice"
};
```
x??

---

#### Example of Using Collection Initializers in HomeController.cs
Background context: The example demonstrates using a collection initializer to create and populate an array with initial values. This can be particularly useful for quickly setting up data for views.

:p What does the following code snippet do?
```csharp
return View("Index", new string[] {
    "Bob",
    "Joe",
    "Alice"
});
```
??x
The code initializes a string array with three names and passes it to a view. The `new string[] { ... }` syntax is used to define the array contents inline, making the setup more concise.

```csharp
string[] names = new string[] {
    "Bob",
    "Joe",
    "Alice"
};
return View("Index", names);
```
x??

---

#### Restarting ASP.NET Core and Requesting a URL

Background context: The provided text describes how to restart an ASP.NET Core application and make a request to `http://localhost:5000` to see specific output. This is done to demonstrate certain C# features in the context of web development.

:p What does restarting ASP.NET Core and requesting `http://localhost:5000` do?

??x
Restarting ASP.NET Core and requesting `http://localhost:5000` reloads the application and triggers a request that outputs specific data to the browser, demonstrating C# features such as dictionary initialization.

```csharp
// HomeController.cs in the Controllers folder
namespace LanguageFeatures.Controllers {
    public class HomeController : Controller {
        public ViewResult Index() {
            Dictionary<string, Product> products = new Dictionary<string, Product>() {
                ["Kayak"] = new Product { Name = "Kayak", Price = 275M },
                ["Lifejacket"] = new Product { Name = "Lifejacket", Price = 48.95M }
            };
            return View("Index", products.Keys);
        }
    }
}
```
x??

---

#### Index Initializer Syntax

Background context: The text explains how to initialize a dictionary using C# index initializer syntax, which is more natural and concise compared to the traditional approach.

:p How does the new index initializer syntax work in initializing dictionaries?

??x
The new index initializer syntax allows you to initialize collections such as dictionaries in a more readable way. Instead of using curly braces `{ }`, it uses square brackets `[ ]` for key-value pairs, making the initialization process simpler and more intuitive.

```csharp
// HomeController.cs in the Controllers folder
namespace LanguageFeatures.Controllers {
    public class HomeController : Controller {
        public ViewResult Index() {
            Dictionary<string, Product> products = new Dictionary<string, Product>() {
                ["Kayak"] = new Product { Name = "Kayak", Price = 275M },
                ["Lifejacket"] = new Product { Name = "Lifejacket", Price = 48.95M }
            };
            return View("Index", products.Keys);
        }
    }

    public class Product {
        public string Name { get; set; }
        public decimal Price { get; set; }
    }
}
```
x??

---

#### Target-Typed New Expressions

Background context: The text introduces a more concise way to initialize objects using target-typed new expressions, which can further simplify code by reducing redundancy.

:p What is the purpose of target-typed new expressions in C#?

??x
Target-typed new expressions provide a more concise syntax for creating instances of types. By replacing the type specification with `new()`, you can make your code cleaner and avoid redundant declarations.

```csharp
// HomeController.cs in the Controllers folder
namespace LanguageFeatures.Controllers {
    public class HomeController : Controller {
        public ViewResult Index() {
            Dictionary<string, Product> products = new() {
                ["Kayak"] = new Product { Name = "Kayak", Price = 275M },
                ["Lifejacket"] = new Product { Name = "Lifejacket", Price = 48.95M }
            };
            return View("Index", products.Keys);
        }

        public class Product {
            public string Name { get; set; }
            public decimal Price { get; set; }
        }
    }
}
```
x??

---

#### Pattern Matching in C#
Pattern matching is a feature introduced in C# that simplifies complex conditional logic by allowing developers to match values against patterns. This can include checking for specific types, ranges, and more within switch statements or if/else blocks.

The `is` keyword is used for type testing and can be used in the following way:
```csharp
if (data[i] is decimal d) { ... }
```
This checks if the value stored in `data[i]` is of type `decimal`. If it matches, the value is assigned to a new variable `d`, which can then be used directly without needing explicit type conversion.

:p How does the `is` keyword work for pattern matching?
??x
The `is` keyword performs a type check and assigns the value to a new variable if the type matches. For example:
```csharp
if (data[i] is decimal d) {
    total += d;
}
```
In this case, it checks if `data[i]` is of type `decimal`. If true, it assigns that value to the `d` variable and adds its value to `total`.

x??

---
#### Pattern Matching in Switch Statements
Pattern matching can also be used within switch statements using the `when` keyword. This allows for more complex conditions beyond just simple equality checks.

Example:
```csharp
switch (data[i]) {
    case decimal decimalValue: 
        total += decimalValue; 
        break;
    case int intValue when intValue > 50: 
        total += intValue; 
        break;
}
```
This code first attempts to match `data[i]` as a `decimal` and then checks if it is greater than 50, respectively.

:p What are the key differences between using an `if` statement with pattern matching versus a traditional switch statement?
??x
The main difference lies in readability and flexibility. Pattern matching allows for more descriptive and concise code compared to traditional switch statements, especially when dealing with multiple conditions. For instance:
```csharp
switch (data[i]) {
    case decimal decimalValue: 
        total += decimalValue; 
        break;
    case int intValue when intValue > 50: 
        total += intValue; 
        break;
}
```
This pattern matching approach is more expressive, making the logic clearer and potentially reducing boilerplate code.

x??

---
#### Extension Methods in C#
Extension methods allow you to add methods to existing classes without modifying their source code. They are particularly useful when working with third-party libraries where you don't have access to the original class definition.

Syntax for defining an extension method:
```csharp
public static class MyClassExtensions {
    public static void MyMethod(this MyClass obj) { ... }
}
```
The `this` keyword before a parameter in a static method indicates that it is an extension method. You can use these methods as if they were part of the original class.

:p How do you define and call an extension method in C#?
??x
To define an extension method, you declare it within a static class using `this` before the first parameter:
```csharp
public static class MyExtensions {
    public static void AddValue(this ShoppingCart cart) { ... }
}
```
You can then call this method as if it were part of the `ShoppingCart` class:
```csharp
cart.AddValue();
```

x??

---
#### Example of Applying Extension Methods
In the provided example, an extension method was added to the `ShoppingCart` class. The method calculates the total price of all products in the cart.

Code snippet:
```csharp
public static decimal TotalPrices(this ShoppingCart cartParam) {
    decimal total = 0;
    if (cartParam.Products != null) {
        foreach (Product? prod in cartParam.Products) {
            total += prod?.Price ?? 0;
        }
    }
    return total;
}
```

:p How does the extension method `TotalPrices` work?
??x
The `TotalPrices` method iterates through each product in the `Products` collection of a `ShoppingCart`. If a product is not null, it adds its price to the total. If any value might be null, it uses the null-conditional operator `?.` and the null-coalescing operator `??` to handle potential null values.

```csharp
public static decimal TotalPrices(this ShoppingCart cartParam) {
    decimal total = 0;
    if (cartParam.Products != null) { // Ensure Products is not null before iteration
        foreach (Product? prod in cartParam.Products) {
            total += prod?.Price ?? 0; // Add the product's price, or zero if it's null
        }
    }
    return total;
}
```

x??

---

