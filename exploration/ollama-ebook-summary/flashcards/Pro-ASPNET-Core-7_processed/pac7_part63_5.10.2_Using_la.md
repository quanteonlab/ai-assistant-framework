# Flashcards: Pro-ASPNET-Core-7_processed (Part 63)

**Starting Chapter:** 5.10.2 Using lambda expression methods and properties

---

---
#### Lambda Expressions for Delegates with Single Parameters
Background context: Lambda expressions can be used to express logic for delegates that have a single parameter. This is useful when you need to pass simple operations or functions as arguments.

:p How do you use lambda expressions with a single parameter in C#?
??x
You use the lambda syntax with one parameter, like `prod => EvaluateProduct(prod)`. Here, `prod` is the single parameter and `EvaluateProduct(prod)` represents the operation to be performed on that parameter.
```csharp
var result = products.Where(prod => EvaluateProduct(prod));
```
x??
---

---
#### Lambda Expressions for Delegates with Multiple Parameters
Background context: When a delegate requires multiple parameters, you must wrap them in parentheses. This is necessary because C# needs to distinguish between single-parameter and multi-parameter lambda expressions.

:p How do you use lambda expressions with multiple parameters in C#?
??x
For a method requiring multiple parameters, you wrap the parameters in parentheses, like `(prod, count) => prod.Price > 20 && count > 0`. This format makes it clear that both `prod` and `count` are part of the input to the lambda expression.
```csharp
var results = products.Where((prod, count) => prod.Price > 20 && count > 0);
```
x??
---

---
#### Lambda Expressions with Multiple Statements
Background context: If a lambda expression requires more than one statement, you can use curly braces `{}` to encapsulate multiple lines of code. The final result is returned using the `return` statement.

:p How do you use lambda expressions with multiple statements in C#?
??x
When you need logic that spans multiple statements, use braces and a return statement:
```csharp
var results = products.Where((prod, count) => {
    // ...multiple code statements...
    return result;
});
```
This approach allows complex operations to be performed within the lambda expression.
x??
---

---
#### Lambda Expressions as Constructors or Methods
Background context: Lambda expressions can replace single-statement methods and constructors. This simplifies your code by making it more concise.

:p How do you rewrite a method with a single statement into a lambda expression in C#?
??x
You can write the method using a lambda expression, omitting the `return` keyword:
```csharp
public ViewResult Index() => 
    View(Product.GetProducts().Select(p => p?.Name));
```
This version is more concise and still performs the same operation as the original.
x??
---

---
#### Lambda Expressions in Properties
Background context: Just like methods, properties can be implemented using lambda expressions. This is particularly useful for simple logic that needs to be encapsulated within a property.

:p How do you implement a property using a lambda expression in C#?
??x
You can define a property with the `=>` operator and a single statement:
```csharp
public bool NameBeginsWithS => 
    Name.Length > 0 && Name[0] == 'S';
```
This example checks if the product name begins with "S" in one line.
x??
---

#### Type Inference Using `var` Keyword
Type inference is a feature where C# infers the type of a variable at compile time based on its initialization. This means you do not need to explicitly specify the type, which simplifies code by reducing redundancy.

In the provided example, the `names` array is defined with strings without explicitly stating the type:
```csharp
var names = new[] { "Kayak", "Lifejacket", "Soccer ball" };
```
:p What does `var` keyword do in C#?
??x
The `var` keyword allows you to declare a variable without specifying its type. The compiler infers the type based on the initialization expression.
```csharp
// Example of using var
var number = 42; // inferred as int
```
x??

---

#### Anonymous Types in C#
Anonymous types are created by combining object initializers with `var` keyword. These objects have properties defined at runtime and do not require a class or struct definition.

In the example provided, an array of anonymous objects is created to represent product data:
```csharp
var products = new[] {
    new { Name = "Kayak", Price = 275M },
    new { Name = "Lifejacket", Price = 48.95M },
    new { Name = "Soccer ball", Price = 19.50M },
    new { Name = "Corner flag", Price = 34.95M }
};
```
:p What are anonymous types in C# used for?
??x
Anonymous types in C# are useful for creating simple, lightweight objects that can be used to transfer data between a controller and a view without needing to define a class or struct.

These types have properties defined at runtime, but the type itself is not known until compilation. The compiler generates an anonymous type with the appropriate properties.
```csharp
// Example of using anonymous types in C#
var product = new { Name = "Kayak", Price = 275M };
```
x??

---

#### Displaying Type Names of Anonymous Objects
Anonymous objects created by object initializers can be accessed through reflection to show their type. However, these types are not named and are typically represented as generic anonymous types.

In the example provided, `GetType().Name` is used to display the name of the generated type:
```csharp
var products = new[] {
    new { Name = "Kayak", Price = 275M },
    new { Name = "Lifejacket", Price = 48.95M },
    new { Name = "Soccer ball", Price = 19.50M },
    new { Name = "Corner flag", Price = 34.95M }
};
return View(products.Select(p => p.GetType().Name));
```
:p How can you display the type name of anonymous objects in C#?
??x
You can use `GetType().Name` to get the name of an anonymous object's generated type at runtime. However, these names are typically represented as generic types like `<f__AnonymousType0`2>`.

```csharp
// Example of displaying type name
var product = new { Name = "Kayak", Price = 275M };
Console.WriteLine(product.GetType().Name); // Output: <>f__AnonymousType0`2
```
x??

---

#### Default Implementations in C# Interfaces
Default implementations in interfaces allow adding methods or properties that have a predefined implementation. This feature enhances flexibility and maintainability by allowing updates to an interface without breaking existing implementations.

C# interfaces are typically used for defining contracts (methods and properties) that classes must implement, but they do not provide any default behavior. However, C# introduced the ability to define default implementations in interfaces starting from C# 8.0.

:p What is a key advantage of using default implementations in interfaces?
??x
Default implementations in interfaces allow you to add methods or properties with predefined logic directly within the interface definition. This means that any class implementing this interface can inherit these default behaviors, making it easier to update or expand an interface without breaking existing implementations.
x??

---
#### Implementing Default Interfaces in C#
To implement a default interface in your code, define the interface and include default methods or properties using the `=>` syntax.

:p How do you add a default implementation for a property in an interface?
??x
You can add a default implementation for a property in an interface by defining it with the `=>` syntax. For example:

```csharp
public interface IProductSelection {
    IEnumerable<Product>? Products { get; }
    IEnumerable<string>? Names => Products?.Select(p => p.Name);
}
```

Here, `Products` is a standard property definition, while `Names` has a default implementation that transforms the `Products` list into a list of names.
x??

---
#### Using Default Implementations in Classes
You can implement an interface with default methods and properties by simply implementing the interface. If you provide your own implementation for any method or property, it will override the default one.

:p How does the `ShoppingCart` class implement the `IProductSelection` interface?
??x
The `ShoppingCart` class implements the `IProductSelection` interface as follows:

```csharp
public class ShoppingCart : IProductSelection {
    private List<Product> products = new();
    
    public ShoppingCart(params Product[] prods) {
        products.AddRange(prods);
    }
    
    public IEnumerable<Product>? Products { get => products; }
}
```

In this example, the `ShoppingCart` class provides its own implementation for the `Products` property. The default implementation for `Names` is inherited from the interface.
x??

---
#### Applying Default Implementations in Controllers
Controllers can use interfaces to leverage default implementations provided by the implementing classes.

:p How does the `HomeController` utilize the `IProductSelection` interface?
??x
The `HomeController` utilizes the `IProductSelection` interface by creating an instance of `ShoppingCart` and accessing its default implementation for the `Names` property:

```csharp
public class HomeController : Controller {
    public ViewResult Index() {
        IProductSelection cart = new ShoppingCart(
            new Product { Name = "Kayak", Price = 275M },
            new Product { Name = "Lifejacket", Price = 48.95M },
            new Product { Name = "Soccer ball", Price = 19.50M },
            new Product { Name = "Corner flag", Price = 34.95M }
        );
        
        return View(cart.Names);
    }
}
```

Here, the `Names` property from `IProductSelection` is used directly in the view.
x??

---

