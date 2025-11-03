# Flashcards: ConcurrencyNetModern_processed (Part 5)

**Starting Chapter:** 2.1 Using function composition to solve complex problems

---

#### Function Composition Overview
Background context explaining function composition as a technique to combine simple functions into more complex ones. The main motivation is to build maintainable, reusable, and easy-to-understand code that can be used in concurrent applications.

:p What is function composition?
??x
Function composition is the process of combining two or more functions so that the output of one function becomes the input of another, thereby creating a new function. This technique helps in building modular and complex solutions from simple components.
x??

---

#### Function Composition in C#
Background context about how C# doesn't natively support function composition but can achieve it through custom methods.

:p How can you compose functions in C#?
??x
In C#, you can use a generic extension method to compose two or more functions. This involves defining an extension method that takes one function as input and returns another, effectively chaining the functions together.
```csharp
static Func<A, C> Compose<A, B, C>(this Func<A, B> f, Func<B, C> g) => (n) => g(f(n));
```
x??

---

#### Example of Function Composition in C#
Background context about using lambda expressions to define functions and then compose them.

:p How can you combine the `grindCoffee` and `brewCoffee` functions in C#?
??x
You can use a generic extension method called `Compose` to chain the execution of two functions. Here’s how it works:
```csharp
Func<CoffeeBeans, CoffeeGround> grindCoffee = coffeeBeans => new CoffeeGround(coffeeBeans);
Func<CoffeeGround, Espresso> brewCoffee = coffeeGround => new Espresso(coffeeGround);

static Func<A, C> Compose<A, B, C>(this Func<A, B> f, Func<B, C> g) => (n) => g(f(n));

Func<CoffeeBeans, Espresso> makeEspresso = grindCoffee.Compose(brewCoffee);
```
x??

---

#### Function Composition vs. Pipelining
Background context about the differences between function composition and pipelining.

:p What is the difference between function composition and pipelining?
??x
Function composition involves combining functions where the output of one function serves as input to another, resulting in a new function that can be executed later.
Pipelining executes each operation sequentially with the result of one feeding into the next immediately. Function composition returns a combined function without immediate execution, allowing it to be used later.

In summary:
- Pipelining: `input -> f1 -> output` and then `output -> f2 -> newOutput`
- Composition: `(input -> f1 -> output) . (output -> f2 -> finalOutput)` as a single function.
x??

---

#### Importance of Function Composition
Background context about why function composition is important for building maintainable, reusable, and clear code in concurrent applications.

:p Why is function composition important?
??x
Function composition is crucial because it simplifies complex problems by breaking them down into smaller, manageable functions. This makes the code easier to understand, maintain, and reuse. It also ensures that each part of the solution has a single responsibility, which is beneficial for concurrent programming where purity and modularity are essential.
x??

---

#### Use Case: Solving Problems through Function Composition
Background context about how function composition can be used to solve problems in a top-down manner.

:p How does function composition help in solving complex problems?
??x
Function composition helps by breaking down large, complex problems into smaller, simpler functions. Each of these small functions can then be composed together to create a solution that is both modular and easy to understand. This approach mirrors the natural process of deconstructing a problem into manageable parts before reassembling them to form a complete solution.
x??

---

#### Function Composition in Functional Programming

Function composition is a technique where functions are combined to form new, more complex functions. In functional programming languages like F#, it allows you to chain operations together in a declarative manner.

In the context of function composition, a higher-order function (HOF) takes one or more functions as input and returns a function as its result. This is useful for creating reusable and modular code.

:p What does function composition allow developers to do?
??x
Function composition allows developers to create new, more complex functions by chaining simpler ones together. It helps in breaking down problems into smaller, manageable functions that can be combined in various ways.
x??

---

#### Composing Functions with `Compose` Extension Methods

In C#, the `Compose` extension methods can be used to combine functions in a similar way to how they are done in functional languages like F#. This helps in creating reusable and modular code.

:p How does function composition work using `Compose` in C#?
??x
Function composition using `Compose` in C# works by taking two functions as input: one that transforms the input, and another that further processes the result. The outer function is applied to the inner function's output, effectively chaining them together.

For example:
```csharp
public static Func<B, C> Compose<A, B, C>(this Func<A, B> funcOne, Func<B, C> funcTwo)
{
    return a => funcTwo(funcOne(a));
}
```

Here, `funcOne` and `funcTwo` are functions that transform the input. The `Compose` method returns a new function that first applies `funcOne` to the input and then applies `funcTwo` to the result.

x??

---

#### Function Composition in F#

In F#, function composition is natively supported using the `>>` operator, which allows you to combine existing functions easily and read the code more naturally from left to right.

:p How does function composition work in F#?
??x
Function composition in F# uses the `>>` operator. When two functions are combined with `>>`, the first function is applied to an input value, and then the result of the first function is passed as input to the second function.

For example:
```fsharp
let add4 x = x + 4
let multiplyBy3 x = x * 3

// Using List.map with sequential composition
let list = [0..10]
let newList = List.map (fun x -> multiplyBy3(add4(x))) list

// Using function composition and `>>` operator
let newList = list |> List.map ((+) 4 >> (* 3))
```

Here, the `>>` operator is used to chain functions together. The result is a more readable and concise code that aligns with F# idioms.

x??

---

#### Closures in Functional Programming

Closures are special functions that carry an implicit binding to all nonlocal variables (free variables) referenced by them. They allow functions to access local state even when invoked outside their lexical scope, making it easier to manage and pass around data.

:p What is a closure?
??x
A closure is a higher-order function that has access to its lexical environment even after the outer function has finished executing. This means that a closure can reference variables from an enclosing scope and keep them in memory, allowing the function to maintain state across multiple invocations.

For example:
```csharp
string freeVariable = "I am a free variable";
Func<string, string> lambda = value => freeVariable + " " + value;
```

Here, `lambda` is a closure that captures the `freeVariable` from its enclosing scope. Each time `lambda` is called, it uses the captured state of `freeVariable`.

x??

---

#### Closures in C#

Closures have been available in .NET since version 2.0 and are particularly useful for managing state within functions.

:p What are closures used for in C#?
??x
Closures in C# are used to capture variables from their enclosing scope, making them accessible even after the outer function has finished executing. This is particularly useful when you need to maintain some state or context across multiple invocations of a function without having to pass around additional parameters.

For example:
```csharp
string freeVariable = "I am a free variable";
Func<string> lambda = () => freeVariable + " is still alive!";
```

Here, `lambda` is a closure that captures the `freeVariable`. Each time `lambda` is called, it returns the updated value of `freeVariable`, demonstrating how closures can manage state.

x??

---

#### Captured Variables and Closures in Asynchronous Programming

Background context: In functional programming, closures allow for capturing variables from their lexical scope even after those scopes have exited. This is particularly useful in asynchronous operations where you need to access or modify state that exists outside of an event handler.

C# provides lambda expressions which can capture local variables. However, the captured variable retains its initial value at the time of closure creation; any changes to the original variable afterwards do not affect the captured version.

:p How does a lambda expression with captured variables work in C# for asynchronous programming?
??x
A lambda expression captures the value of local variables at the time it is created. Even after these variables go out of scope or are changed, their initial values remain within the closure. For example:

```csharp
void UpdateImage(string url)
{
    System.Windows.Controls.Image image = img;
    var client = new WebClient();
    client.DownloadDataCompleted += (o, e) =>
    {
        if (image != null)
        {
            using (var ms = new MemoryStream(e.Result))
            {
                var imageConverter = new ImageSourceConverter();
                image.Source = (ImageSource)imageConverter.ConvertFrom(ms);
            }
        }
    };
    client.DownloadDataAsync(new Uri(url));
    image = null; // This does not affect the lambda's captured variable
}
```
In this example, `image` is a local variable that gets captured by the lambda. Even though it’s set to `null` after the call to `DownloadDataAsync`, the lambda still uses its original value.

x??

---

#### Null Objects and Closures

Background context: In functional programming languages like F#, null objects do not exist, thus avoiding potential bugs related to null values in closures. This is an important consideration when working with mutable state in a multithreading environment.

:p What happens if we try to update a UI element using a closure that captures its reference after the reference has been set to `null`?
??x
The closure still retains the initial value of the variable at the time it was created, even if the original variable is later set to `null`. This can lead to unexpected behavior. For example:

```csharp
void UpdateImage(string url)
{
    System.Windows.Controls.Image image = img;
    var client = new WebClient();
    client.DownloadDataCompleted += (o, e) =>
    {
        if (image != null)
        {
            using (var ms = new MemoryStream(e.Result))
            {
                var imageConverter = new ImageSourceConverter();
                image.Source = (ImageSource)imageConverter.ConvertFrom(ms);
            }
        }
    };
    client.DownloadDataAsync(new Uri(url));
    image = null; // This does not affect the lambda's captured variable
}
```
Here, setting `image` to `null` after starting the asynchronous download does not impact the closure, which still references the original value of `img`.

x??

---

#### Closures and Multithreading

Background context: In a multithreaded environment, closures can capture local variables, but if those variables are mutable, their values might change during execution. This can lead to race conditions or other unexpected behaviors.

:p How do closures in lambda expressions affect the behavior of code when running in multiple threads?
??x
Closures capture references to variables, not their current state at the time of closure creation. If a variable changes after being captured by a closure and that closure is executed later, it will use the latest value, not the original one. For example:

```csharp
for (int iteration = 1; iteration < 10; iteration++)
{
    Task.Factory.StartNew(() => Console.WriteLine("{0} - {1}", Thread.CurrentThread.ManagedThreadId, iteration));
}
```
Here, each task captures the `iteration` variable by reference. If `iteration` is modified before any of the tasks start executing, all tasks will print the final value of `iteration`, not the one at the time they were created.

x??

---

#### F# and Closures in Multithreading

Background context: In functional languages like F#, null objects do not exist, which simplifies state management. However, mutable state still needs careful handling to avoid issues with multithreading.

:p How does F# handle closures and variables in a multithreaded environment compared to C#?
??x
In F#, the concept of null objects doesn’t exist, so there’s no risk of using `null` references in closures. However, when dealing with mutable state, careful management is still necessary. For example:

```fsharp
let tasks = Array.zeroCreate<Task> 10

for index = 1 to 10 do
    tasks.[index - 1] <- Task.Factory.StartNew(fun () -> Console.WriteLine(index))
```
In this F# code, each task captures the `index` variable by value (not reference), ensuring that each task prints its own distinct number.

x??

---

