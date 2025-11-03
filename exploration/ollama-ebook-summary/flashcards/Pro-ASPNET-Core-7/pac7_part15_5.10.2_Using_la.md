# Flashcards: Pro-ASPNET-Core-7_processed (Part 15)

**Starting Chapter:** 5.10.2 Using lambda expression methods and properties

---

#### Lambda Expressions for Delegates
Background context: Lambda expressions provide a concise way to write methods that can be passed as arguments to other functions or used within collections. They are particularly useful when you need to define simple, inline logic without creating a full method.

:p What is a lambda expression and how does it relate to delegates?
??x
A lambda expression is an anonymous function that can capture variables from its containing scope and use them in the expression body. It can be used as a delegate or passed into methods like LINQ query expressions. Lambda expressions are useful for writing simple, inline logic without defining separate methods.
```csharp
// Example of using a lambda with LINQ
var products = Product.GetProducts().Where(p => p.Price > 20);
```
x??

---

#### Calling Methods within Lambda Expressions
Background context: You can call methods directly from lambda expressions. This is useful when you need to delegate the logic of your lambda expression to another method.

:p How do you call a method inside a lambda expression?
??x
To call a method inside a lambda expression, simply write the method name followed by its parameters in parentheses and include any necessary logic within the body. The result of the method is used as the value for the lambda expression.
```csharp
// Example of calling a method `EvaluateProduct` inside a lambda
var filteredProducts = Product.GetProducts().Where(p => EvaluateProduct(p));
```
x??

---

#### Multiple Parameters in Lambda Expressions
Background context: When using multiple parameters in a lambda expression, you must enclose the parameters within parentheses.

:p How do you handle multiple parameters in a lambda expression?
??x
When using multiple parameters in a lambda expression, enclose all parameters within parentheses. This helps to clearly define the input arguments for the lambda.
```csharp
// Example of a lambda with multiple parameters
var filteredProducts = Product.GetProducts().Where((prod, count) => prod.Price > 20 && count > 0);
```
x??

---

#### Multi-Statement Lambda Expressions
Background context: If you need to write more complex logic within your lambda expression that spans over multiple lines, use curly braces `{}` and a `return` statement.

:p How do you write multi-statement logic in a lambda expression?
??x
To include complex logic with multiple statements inside a lambda expression, use curly braces `{}`. Any logic written within the braces will be executed sequentially, and the result must be returned using a `return` statement.
```csharp
// Example of a multi-statement lambda expression
var filteredProducts = Product.GetProducts().Where((prod, count) => 
{
    if (count > 0 && prod.Price > 20)
    {
        return true;
    }
    return false;
});
```
x??

---

#### Lambda Expressions in Action Methods
Background context: In ASP.NET Core development, you often use single-statement action methods that select data and render views. These can be rewritten using lambda expressions for conciseness.

:p How can an action method use a lambda expression?
??x
An action method can use a lambda expression to simplify the logic of selecting data and rendering views. By omitting the `return` keyword and using the `=>` operator, you can write more concise code.
```csharp
// Example of an action method with a lambda expression
public ViewResult Index() => View(Product.GetProducts().Select(p => p?.Name));
```
x??

---

#### Lambda Expressions for Properties
Background context: Lambda expressions can also be used to define properties, providing a concise way to encapsulate logic.

:p How are lambda expressions used in property definitions?
??x
Lambda expressions can be used to implement properties. The syntax is similar to defining methods but uses the `=>` operator instead of `{ }`. This makes the implementation more readable and concise.
```csharp
// Example of a property using a lambda expression
public bool NameBeginsWithS => Name.Length > 0 && Name[0] == 'S';
```
x??

---

#### Type Inference Using `var` Keyword
Type inference allows you to declare variables without explicitly specifying their type, as demonstrated by the compiler deducing types from context. This is particularly useful for reducing redundancy and improving readability of code.
:p What does the `var` keyword allow in C#?
??x
The `var` keyword allows you to define a local variable or collection without explicitly specifying its type, relying on the compiler to infer the appropriate type based on the initializer or declaration context. This can make your code more concise and readable by focusing on what the variable represents rather than its concrete type.
```csharp
var names = new[] { "Kayak", "Lifejacket", "Soccer ball" };
```
x??

---

#### Example of Type Inference in HomeController.cs
In the provided example, `var` is used to define a variable `names`, which stores an array of strings. The type of `names` is inferred by the compiler based on its initialization with an array of string literals.
:p How does type inference work in the `Index()` method of `HomeController`?
??x
Type inference works such that when you use the `var` keyword, the variable's type is determined by the compiler. In this case, since the array is initialized with string literals, the inferred type for `names` is an array of strings (`string[]`). The following code demonstrates this:
```csharp
public ViewResult Index() {
    var names = new[] { "Kayak", "Lifejacket", "Soccer ball" };
    return View(names);
}
```
x??

---

#### Anonymous Types in C#
Anonymous types are a feature that allows you to define and use objects with properties without creating a custom class or struct. These objects can be used as simple view models for transferring data between controllers and views.
:p What is an anonymous type?
??x
An anonymous type is a type that is created automatically by the compiler when you initialize it using object initializers. This allows you to create objects with properties without defining a specific class or struct, making your code simpler and more concise. Here's how you define an anonymous type:
```csharp
var products = new[] {
    new { Name = "Kayak", Price = 275M },
    new { Name = "Lifejacket", Price = 48.95M },
    new { Name = "Soccer ball", Price = 19.50M },
    new { Name = "Corner flag", Price = 34.95M }
};
```
x??

---

#### Anonymous Types vs. Dynamic
Anonymous types are not the same as dynamic types in C#. While anonymous types provide strong typing for the properties you define, they do not allow property reassignment after initialization. They can be useful for data transfer but have limitations compared to dynamically typed variables.
:p How do anonymous types differ from dynamic types?
??x
Anonymous types and dynamic types both offer flexibility, but they serve different purposes and have distinct characteristics:
- **Anonymous Types**: These are created by the compiler when you use object initializers. They provide strong typing for properties defined in the initialization and cannot be changed after creation.
- **Dynamic Types**: These allow runtime type checking and property reassignment. You can assign a `dynamic` type to any variable, making it very flexible but potentially less safe than strongly-typed anonymous types.

An example of an anonymous type:
```csharp
var product = new { Name = "Kayak", Price = 275M };
```
x??

---

#### Example of Using Anonymous Types in View Models
The `Index()` method uses a collection of anonymous objects to transfer data from the controller to the view. The properties are defined within the object initializer, and these properties become part of the type inferred by the compiler.
:p How is an array of anonymous types used in this context?
??x
In the provided example, an array of anonymous types is created to serve as a simple view model for transferring data from the controller to the view. Each element in the `products` array has properties like `Name` and `Price`, which are defined using object initializers:
```csharp
var products = new[] {
    new { Name = "Kayak", Price = 275M },
    new { Name = "Lifejacket", Price = 48.95M },
    new { Name = "Soccer ball", Price = 19.50M },
    new { Name = "Corner flag", Price = 34.95M }
};
```
The properties `Name` and `Price` are part of the inferred type, which is unique to this instance but behaves like a normal class with these properties.
x??

---

#### Type Names of Anonymous Objects
When working with anonymous types, you can retrieve the type name using `GetType().Name`. However, be aware that the generated type names may not be user-friendly and are intended for internal use by the compiler.
:p How do you determine the type of an anonymous object in C#?
??x
You can determine the type of an anonymous object by calling the `GetType()` method on it and then using `.Name` to get a string representation of its type. The resulting name is generated by the compiler and may not be human-readable. Here's how you would do this:
```csharp
var products = new[] {
    new { Name = "Kayak", Price = 275M },
    new { Name = "Lifejacket", Price = 48.95M },
    new { Name = "Soccer ball", Price = 19.50M },
    new { Name = "Corner flag", Price = 34.95M }
};

return View(products.Select(p => p.GetType().Name));
```
The generated type names might look like `<>f__AnonymousType0` and can vary, as they are internal to the compiler.
x??

---

#### Default Implementations in Interfaces
Background context: C# provides a feature to define default implementations for properties and methods within interfaces. This allows developers to add new features to an interface without breaking existing implementations of that interface.

C/Java does not support this feature; only C# has introduced it as part of its language evolution.

:p What is the purpose of adding default implementations in C#'s interfaces?
??x
The purpose is to allow new features to be added to an interface without requiring all implementing classes to be updated. This can simplify maintenance and reduce the risk of breaking existing codebases when changes are made.

For example, if you need a property like `Names` for some implementations but not others, you can define it as a default implementation in the interface:

```csharp
public interface IProductSelection {
    IEnumerable<Product>? Products { get; }
    IEnumerable<string>? Names => Products?.Select(p => p.Name);
}
```
x??

---

#### Implementing Interfaces with Default Implementations
Background context: The `ShoppingCart` class is updated to implement the `IProductSelection` interface, which includes a default implementation for the `Names` property.

:p How does the `ShoppingCart` class use the default implementation provided in the `IProductSelection` interface?
??x
The `ShoppingCart` class uses the default implementation of the `Names` property defined in the `IProductSelection` interface. Even though the `Products` property is implemented, the `Names` property leverages the default implementation.

```csharp
public class ShoppingCart : IProductSelection {
    private List<Product> products = new();
    
    public ShoppingCart(params Product[] prods) {
        products.AddRange(prods);
    }
    
    public IEnumerable<Product>? Products { get => products; }
}
```
x??

---

#### Using Interfaces with Default Implementations in Controllers
Background context: The `HomeController` uses the `IProductSelection` interface, which now includes a default implementation for the `Names` property. This allows the controller to directly access and utilize the `Names` property.

:p How does the `HomeController` use the `Names` property defined in the `IProductSelection` interface?
??x
The `HomeController` uses the `Names` property by casting an instance of `ShoppingCart` to `IProductSelection` and then accessing the `Names` property directly. This works because the `ShoppingCart` class implements the `IProductSelection` interface, which includes a default implementation for `Names`.

```csharp
public ViewResult Index() {
    IProductSelection cart = new ShoppingCart(
        new Product { Name = "Kayak", Price = 275M },
        new Product { Name = "Lifejacket", Price = 48.95M },
        new Product { Name = "Soccer ball", Price = 19.50M },
        new Product { Name = "Corner flag", Price = 34.95M }
    );
    
    return View(cart.Names);
}
```
x??

---

