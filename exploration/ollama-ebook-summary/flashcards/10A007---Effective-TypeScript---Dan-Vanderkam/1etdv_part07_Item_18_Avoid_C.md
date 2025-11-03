# Flashcards: 10A007---Effective-TypeScript---Dan-Vanderkam_processed (Part 7)

**Starting Chapter:** Item 18 Avoid Cluttering Your Code with Inferable Types

---

#### Type Inference and Control Flow Analysis

TypeScript, like other modern programming languages, supports type inference to reduce redundancy. This allows developers to write fewer explicit type annotations, making their code more concise and readable.

In statically typed languages like TypeScript, types are inferred based on the context in which variables or expressions are used. Control flow analysis tracks how these types change as the program executes, ensuring type safety without cluttering the code with redundant type information.

:p How does TypeScript handle variable types differently compared to traditional statically typed languages?
??x
TypeScript infers the types of variables at specific locations within your code rather than requiring you to declare them explicitly everywhere. This is known as control flow analysis. For example, in a function like this:

```typescript
function add(a: number, b: number): number {
    return a + b;
}
```

The `number` type for the parameters and return value can be inferred by TypeScript. You don't need to declare them explicitly if you are sure of their types.

This allows you to write more concise code like:

```typescript
function add(a, b) {
    return a + b;
}
```

Here, TypeScript infers `a` and `b` as numbers based on the context in which they are used.
x??

---
#### Redundant Type Annotations

In TypeScript, many developers, especially beginners, tend to over-annotate their code with explicit type declarations. However, this can lead to cluttered and less readable code.

TypeScript has sophisticated type inference capabilities that allow it to automatically determine the types of variables based on context. This means you don't need to declare the type every time, making your code cleaner.

:p Why are explicit type annotations considered redundant in TypeScript?
??x
Explicit type annotations are redundant in TypeScript when the type can be inferred from the context. For example:

Instead of:
```typescript
let x: number = 12;
```

You can simply write:
```typescript
let x = 12;
```

The variable `x` is inferred to have a `number` type by TypeScript.

This approach reduces redundancy and improves code readability, as the editor can still show you the inferred types for clarity. However, it's important to note that in complex scenarios or when the type is not clear from context, explicit annotations may be necessary.
x??

---
#### Control Flow Analysis

Control flow analysis tracks how a variable’s type changes due to surrounding code. This helps TypeScript maintain type safety without requiring developers to write redundant type annotations.

For example, consider an object initialization with inferred types:

```typescript
const person = {
    name: 'Sojourner Truth',
    born: { 
        where: 'Swartekill, NY', 
        when: 'c.1797' 
    },
    died: { 
        where: 'Battle Creek, MI', 
        when: 'Nov. 26, 1883' 
    }
};
```

Here, the types for `name`, `where`, and `when` are inferred by TypeScript.

:p How does control flow analysis work in TypeScript?
??x
Control flow analysis works by tracking how a variable’s type changes as it is used throughout the code. For example:

```typescript
const person = {
    name: 'Sojourner Truth',
    born: { 
        where: 'Swartekill, NY', 
        when: 'c.1797' 
    },
    died: { 
        where: 'Battle Creek, MI', 
        when: 'Nov. 26, 1883' 
    }
};
```

In this example, `person.name` is inferred to be of type `string`, and the nested properties like `born.where` are also inferred to be of type `string`. The same applies to `died.where`.

TypeScript uses these inferences to ensure that operations on variables are type-safe without requiring explicit type annotations at every point.
x??

---

#### Type Inference for Object Literals and Arrays
TypeScript infers types based on object literals and array elements, which can add precision to variable types without explicit annotations. This is generally beneficial but should be balanced with clarity.

:p How does TypeScript infer the type of a function's return value in this context?
??x
TypeScript infers the return type based on the input types and operations performed within the function. For example:

```typescript
function square(nums: number[]) {
    return nums.map(x => x * x);
}
const squares = square([1, 2, 3, 4]); // inferred as const squares: number[]
```

x??

---

#### Precision in Type Inference
TypeScript may infer more precise types than explicitly defined ones. Explicit annotations can sometimes reduce precision and add unnecessary noise.

:p How does explicit annotation on a variable compare to inferring its type?
??x
Explicit annotations can be noisy and less precise compared to letting TypeScript infer the type. For example:

```typescript
const axis1: string = 'x'; // const axis1: string
const axis2 = 'y';         // const axis2: "y"
```

Here, inferring `axis2` as `"y"` is more precise than explicitly annotating it with `string`.

x??

---

#### Refactoring and Type Inference
Type inference allows for easier refactoring. Explicit type annotations can make refactoring harder by creating false errors.

:p How does type inference help in refactoring?
??x
Type inference simplifies refactoring because it automatically adapts to changes in the underlying types without requiring manual updates everywhere. For instance, changing `id` from `number` to `string` in a `Product` interface:

```typescript
interface Product {
    id: string; // Changed from number
    name: string;
    price: number;
}
```

Without explicit annotations, this change would be reflected automatically by TypeScript. With explicit annotations, you might get false errors.

x??

---

#### Destructuring Assignment and Type Inference
Destructuring assignment is a concise way to extract values and can work well with type inference, reducing the need for explicit type annotations in variable declarations.

:p How does destructuring enhance code conciseness and maintainability?
??x
Destructuring allows you to assign variables directly from objects or arrays without needing separate `const` statements. This improves readability and reduces boilerplate code:

```typescript
function logProduct(product: Product) {
    const {id, name, price} = product;
    console.log(id, name, price);
}
```

This version is more concise than explicitly annotating each variable:

```typescript
function logProduct(product: Product) {
    const {id, name, price}: {id: string; name: string; price: number} = product;
    console.log(id, name, price);
}
```

The first version leverages type inference and destructuring to avoid clutter.

x??

---

#### Function Parameter Types
Function parameters require explicit type annotations in TypeScript because the language does not infer types based on parameter usage. You must define the expected type for each function parameter explicitly.

:p Why do we need to annotate function parameters even if they are used within the function?
??x
In TypeScript, you need to provide explicit type annotations for function parameters because the type system infers variable types at their declaration points, not based on where they are used. For example:

```typescript
function logProduct(product: Product) {
    const id = product.id; // needs a type annotation here if `id` is not destructured
}
```

Here, without an explicit type for `id`, TypeScript cannot determine its type from the function's parameter declaration.

x??

#### Type Inference and Annotations
TypeScript infers types based on usage, but sometimes explicit annotations are beneficial for clarity or error reporting. This section discusses when to leave type annotations off versus when to include them.

:p When should you consider leaving type annotations off?
??x
You should consider leaving type annotations off in situations where the type can be inferred from context. For instance:
- Function parameters with default values.
- Callbacks where the library has typed declarations.
```typescript
function parseNumber(str: string, base = 10) {
    // The parameter 'base' will have its type inferred as number due to the default value of 10.
}

app.get('/health', (request, response) => { 
    // Here, TypeScript infers the types for request and response from the Express library's declarations.
});
```
x??

---

#### Excess Property Checking
Explicitly defining a type on an object literal can help catch errors by enabling excess property checking. This ensures that only defined properties are included in the object.

:p Why might you want to specify a type even when it can be inferred for an object literal?
??x
Specifying a type for an object literal enables excess property checking, which helps catch errors if any unexpected or undefined properties are added during its definition. This is particularly useful in larger codebases where errors could otherwise occur elsewhere.

```typescript
const elmo: Product = {
    name: 'Tickle Me Elmo',
    id: '048188 627152', // Explicit type annotation helps catch errors.
    price: 28.99,
};

// Without an annotation, a mistake in the object's definition might result in an error where it’s used:
const furby = {
    name: 'Furby',
    id: 630509430963,
    price: 35,
};
logProduct(furby); // Error reported here, not at the object's definition.
```
x??

---

#### Return Type Annotations
Even if a function return type can be inferred, annotating it is beneficial to avoid implementation errors leaking into function usage. This is especially crucial for exported functions in public APIs.

:p Why might you still want to annotate a function’s return type even when it can be inferred?
??x
Annotating the return type of a function helps catch implementation errors where the function does not conform to its declared return type, ensuring that any discrepancies are caught at the point of definition rather than at the call site.

```typescript
function getQuote(ticker: string): Promise<number> {
    const cache: {[ticker: string]: number} = {};
    
    if (ticker in cache) { 
        // Incorrectly returning a number instead of a promise.
        return cache[ticker]; 
    } 

    return fetch(`https://quotes.example.com/?q=${ticker}`)
        .then(response => response.json())
        .then(quote => {
            cache[ticker] = quote;
            return quote as number; // This should be `Promise.resolve(cache[ticker])`.
        });
}

// Incorrect implementation, but the error is not caught until function call:
getQuote('MSFT').then(considerBuying); 
// Error: Property 'then' does not exist on type 'number | Promise<number>'.
```

To avoid this issue and ensure the correct return type is reported at definition time:

```typescript
function getQuote(ticker: string): Promise<number> {
    const cache: {[ticker: string]: number} = {};
    
    if (ticker in cache) { 
        // Correct implementation with explicit annotation.
        return Promise.resolve(cache[ticker]); 
    } 

    return fetch(`https://quotes.example.com/?q=${ticker}`)
        .then(response => response.json())
        .then(quote => {
            cache[ticker] = quote;
            return quote as number; // Now the error is caught at definition time.
        });
}
```
x??

---

#### Importance of Annotating Return Types

Background context: When you annotate the return type, it helps prevent implementation errors from manifesting as runtime errors in user code. This is especially useful for functions that have multiple return statements like `getQuote`. TypeScript requires explicit type annotations to ensure all return statements return the same type.

If a function has multiple returns and a public API, providing an explicit return type annotation can prevent subtle bugs and make your intentions clearer.

:p Why is it important to annotate the return types in such functions?
??x
Annotating return types ensures that TypeScript checks for consistency across all possible paths of execution. Without annotations, TypeScript infers the most general return type, which might not align with your function's actual purpose. This can lead to errors only during runtime if not caught earlier.

For example:
```typescript
function getQuote(isPositive: boolean): string | null {
    if (isPositive) {
        return "Good news!";
    } else {
        // Implementation for negative quotes omitted.
        return null;
    }
}
```
Adding the explicit type annotation makes it clear that the function can return either a `string` or `null`.

```typescript
function getQuote(isPositive: boolean): string | null {
    if (isPositive) {
        return "Good news!";
    } else {
        // Implementation for negative quotes omitted.
        return null;
    }
}
```
x??

---

#### Use of Named Return Types

Background context: When using a named type, it's helpful to explicitly annotate the return type. This is because TypeScript infers the return type from the values being returned, which might not align with your intended use.

For instance:
```typescript
interface Vector2D {
    x: number;
    y: number;
}

function add(a: Vector2D, b: Vector2D): Vector2D {
    return { x: a.x + b.x, y: a.y + b.y };
}
```
Here, TypeScript infers the return type as `Vector2D`, but it might surprise users who expect to see a named type.

:p What is the advantage of explicitly annotating the return type when using a named type?
??x
Explicitly annotating the return type ensures that both you and your users understand the intended behavior. It makes the API clearer and easier to maintain, as the function's contract (its type signature) is explicit.

```typescript
interface Vector2D {
    x: number;
    y: number;
}

function add(a: Vector2D, b: Vector2D): Vector2D {  // Explicitly annotated return type
    return { x: a.x + b.x, y: a.y + b.y };
}
```
This makes it clear that the function returns an object with `x` and `y` properties, matching the `Vector2D` interface.

x??

---

#### Typescript ESLint Rule for Return Type Annotations

Background context: The `no-inferrable-types` rule in TypeScript ESLint ensures that you only use type annotations when they are necessary. This helps avoid redundant and misleading type information.

:p How does the `no-inferrable-types` rule in TypeScript ESLint work?
??x
The `no-inferrable-types` rule enforces that you explicitly annotate return types unless TypeScript can infer them correctly. This ensures that your code is clean and that type annotations are meaningful.

To use this rule, you would configure it in your `.eslintrc.json` file like so:

```json
{
    "rules": {
        "no-inferrable-types": ["error", { "ignoreUnions": true }]
    }
}
```

With this configuration, TypeScript ESLint will enforce that all return types must be explicitly annotated unless the type can be inferred correctly.

For example:
```typescript
function add(a: number, b: number): number {
    return a + b;  // Explicitly annotated return type (no warning)
}

// Without explicit annotation and with `ignoreUnions` set to true,
// this would produce a warning because the union can be inferred.
function processInput(input: string | null) {
    if (input) {
        return input.toUpperCase();  // No explicit annotation
    } else {
        return "Default";
    }
}
```

x??

---

#### Impact on Compiler Performance

Background context: For large codebases, reducing the amount of work TypeScript needs to do in inferring types can have a positive impact on compiler performance. Explicit type annotations reduce the complexity and improve readability.

:p How does annotating your return types affect compiler performance?
??x
Annotating return types can help improve compiler performance for large projects by reducing the complexity that the TypeScript compiler must handle during inference. When you annotate return types, you provide explicit information to the compiler, which can then optimize its work more effectively.

For instance:
```typescript
function calculateArea(radius: number): number {
    return Math.PI * radius ** 2;  // Explicitly annotated return type
}
```
In contrast, without annotations, TypeScript must infer that `calculateArea` returns a `number`, which might be less efficient for the compiler to determine.

For large codebases, this can have a noticeable impact on build times and overall performance. Therefore, it's beneficial to annotate types when necessary.

x??

---

#### Writing Type Signatures First

Background context: Writing the full type signature first helps you think more clearly about your function’s contract before implementing it. This is akin to test-driven development (TDD) in that you define what the function should do and how it should behave, ensuring that its design aligns with your intended use.

:p Why should you write the full type signature first when implementing a function?
??x
Writing the full type signature first helps ensure that your function's contract is clear from the start. This approach makes it easier to understand what inputs are expected and what outputs will be produced, leading to more robust and maintainable code.

For example:
```typescript
function addVectors(a: Vector2D, b: Vector2D): Vector2D {
    // Implementation details omitted.
}
```
This signature makes it clear that the function expects two `Vector2D` objects and returns a new one. Without this, you might implement it differently during development.

x??

---

#### Avoid Writing Type Annotations

Background context: TypeScript can often infer return types correctly in many cases, making explicit type annotations redundant and potentially misleading. By default, for functions with straightforward return types, the compiler infers the type automatically.

:p In what scenarios should you avoid writing type annotations?
??x
You should avoid writing type annotations when the compiler can infer the same type on its own. For simple return types like primitives or basic objects, explicit annotations add unnecessary complexity and might be misleading if they differ from the inferred type.

For example:
```typescript
function getLength(s: string): number {
    return s.length;
}
```
Here, TypeScript infers that `getLength` returns a `number`, so adding an explicit annotation is redundant:

```typescript
// Redundant and unnecessary in this case.
function getLength(s: string): number {  // Explicitly annotated
    return s.length;
}
```

x??

---

#### TypeScript Variable Types and Reuse
In JavaScript, variables can be reused to hold values of different types without issues. However, TypeScript enforces type safety, leading to errors when a variable is reassigned with a value of a different type.

:p How does TypeScript handle variable reuse for differently typed values?
??x
TypeScript infers the type based on the first assignment and does not allow reassignment with a different type, as it maintains strict type checking. This can lead to compiler errors if not handled properly.
```typescript
let productId: string = "12-34-56"; // Type is inferred as string initially
fetchProduct(productId); // No error

productId = 123456; // Error: Type 'number' is not assignable to type 'string'
fetchProductBySerialNumber(productId); // Error: Argument of type 'string' is not assignable to parameter of type 'number'
```
x??

---

#### Union Types in TypeScript
Union types allow a variable to hold values from multiple possible types. However, using union types can complicate the code and make it harder to reason about.

:p How do you define a union type in TypeScript?
??x
A union type is defined by separating the types with the `|` (pipe) symbol within angle brackets.
```typescript
let productId: string | number = "12-34-56"; // Union type definition

fetchProduct(productId); // Works for initial assignment as it's a string
productId = 123456; // Assignment is allowed, and the union narrows to number
fetchProductBySerialNumber(productId); // Now this works because productId is narrowed to number
```
x??

---

#### Prefer Separate Variables for Different Concepts
Using separate variables can improve code clarity and maintainability by avoiding type conflicts and making it easier for both humans and TypeScript to understand the logic.

:p Why should you prefer using different variables for different concepts?
??x
Using distinct variables for different concepts makes the code more readable and easier to understand. It avoids confusion related to variable reuse, especially in TypeScript where strict typing is enforced.
```typescript
const productId = "12-34-56"; // Clear variable name for ID
fetchProduct(productId); 

const serialNumber = 123456; // Clear variable name for serial number
fetchProductBySerialNumber(serialNumber);
```
x??

---

#### Const Over Let
Using `const` over `let` is generally preferred in TypeScript because it makes the code easier to reason about and harder to accidentally modify.

:p Why should you use `const` instead of `let`?
??x
Using `const` is better practice as it indicates that the value will not change, which can make the code more predictable. This also helps with type checking in TypeScript.
```typescript
// Use const for values that do not change
const productId = "12-34-56";
fetchProduct(productId);

let count = 0; // Use let when a value might change
count++;
```
x??

---

#### Shadowing Variables
Shadowing variables, where a variable with the same name is declared inside an inner scope, can lead to confusion and should be avoided.

:p What is shadowing in TypeScript?
??x
Shadowing occurs when you declare a variable with the same name as an outer variable within an inner scope. This creates two separate but unrelated variables.
```typescript
const productId = "12-34-56";
fetchProduct(productId);

{
    const productId = 123456; // Shadowed version, different from the outer one
    fetchProductBySerialNumber(productId); // Works with shadowed variable
}
```
x??

---

#### Widening in TypeScript
Background context explaining the concept. In TypeScript, when you initialize a variable with a constant but don't provide a type, the type checker needs to infer the most general possible type for that value. This process is known as widening.

:p What is widening in the context of TypeScript?
??x
Widening refers to the process where TypeScript infers the most general possible type for a variable based on its initial value when no explicit type annotation is provided. For example, if you assign a string literal to a variable, TypeScript will infer its type as `string`, but it can also include broader types like `any` or unions of types.
```typescript
let x = 'x';  // Inferred type: string
x = 'a';      // Valid assignment since the type is string
x = 'Four score and seven years ago...';  // Also valid because the type is still string

// Without a type annotation, TypeScript will choose the most general type:
const mixed = ['x', 1];  // Inferred type: (string | number)[] or [any, any] depending on context
```
x??

---

#### Vector3 Interface and getComponent Function
Background context explaining the concept. The example provided defines an interface `Vector3` for a 3D vector and a function `getComponent` to retrieve any of its components.

:p What is the purpose of the `Vector3` interface in this code snippet?
??x
The `Vector3` interface is used to define the structure of a 3D vector, specifying that it has properties `x`, `y`, and `z`, each of which should be of type `number`. This helps ensure that any object implementing or assigned as `Vector3` will have these specific numeric properties.
```typescript
interface Vector3 {
    x: number;
    y: number;
    z: number;
}
```
x??

---

#### Type Inference and getComponent Function Call Error
Background context explaining the concept. The example shows how TypeScript infers the type of a variable based on its initialization value, which can sometimes lead to errors if not careful.

:p Why does the call `getComponent(vec, x);` result in an error?
??x
The call `getComponent(vec, x);` results in an error because the type of `x` is inferred as `string`, but the function `getComponent` expects a more specific type for its second argument: `'x' | 'y' | 'z'`. This mismatch leads to a type error due to TypeScript's strict type checking.
```typescript
let x = 'x';  // Inferred type: string
let vec = {x: 10, y: 20, z: 30};  // Object literal with properties of number

// Error: Argument of type 'string' is not assignable to parameter of type '"x" | "y" | "z"'
getComponent(vec, x);
```
x??

---

#### Type Inference for `mixed` Variable
Background context explaining the concept. The example demonstrates how TypeScript infers the most general possible type when initializing a variable with an array containing mixed values.

:p What is the inferred type of the `mixed` variable in this code snippet?
??x
The inferred type of the `mixed` variable in this code snippet is `(string | number)[]`. This means that the array can contain elements of either `string` or `number`, but not both.
```typescript
const mixed = ['x', 1];  // Inferred type: (string | number)[]

// The inferred type allows for:
mixed.push('y');  // Valid since 'y' is a string
mixed.push(2);    // Also valid since 2 is a number
```
x??

---

#### Type Safety and Widening with `let` Variables
Background context explaining the concept. The example illustrates how TypeScript infers the type of variables declared with `let` based on their initial values, ensuring type safety but sometimes leading to widening issues.

:p How does TypeScript infer the type of a variable declared with `let`?
??x
TypeScript infers the type of a `let` variable based on its initial value. For primitive types like strings, numbers, and booleans, it will choose the most specific type possible. However, if the value is more complex or ambiguous, TypeScript may infer a broader type like `any` or a union type.

For example:
```typescript
let x = 'x';  // Inferred type: string

// If the value changes to something else, the type remains consistent:
x = 'a';      // Valid assignment since the inferred type is string
x = 'Four score and seven years ago...';  // Also valid because the type is still string

// However, if you assign a more complex structure:
let y = {x: 10};  // Inferred type: { x: number }
y = {x: 20, y: 30};  // Valid since it matches the inferred object type
```
x??

---

#### Const Keyword and Type Narrowing

Const is a keyword used to declare variables that cannot be reassigned. In TypeScript, using `const` helps to infer more precise types since these values cannot change after initialization.

In JavaScript/TypeScript:
```typescript
const x = 'x'; // The type of x is narrowed down to the string literal "x"
let vec = {x: 10, y: 20, z: 30};
getComponent(vec, x);  // This code now passes the type checker because x cannot be reassigned.
```
:p How does using `const` help in inferring more precise types?
??x
Using `const` helps TypeScript infer a narrower and more specific type for variables. Since the variable cannot change after initialization, the compiler can deduce that it will always hold a particular value or set of values, thus avoiding potential errors due to unexpected changes.

```typescript
// Example with const
const obj = { x: 1 };
obj.x = 3; // This is allowed as x was initialized and its type inferred to be number.
```
x??

---

#### Object Type Inference

TypeScript infers the most specific possible type for objects, but this can sometimes lead to overly generic types if properties are added or modified after initialization.

In JavaScript/TypeScript:
```typescript
const obj = { x: 1 };
obj.x = '3'; // Error: Type 'string' is not assignable to type 'number'.
```
:p How does TypeScript infer the type of an object, and what issues can arise from this?
??x
TypeScript infers the type of objects based on their properties. It tries to find a "best common type" that fits all known property assignments. However, if you add new properties or modify existing ones, it may not catch these changes due to its static nature.

```typescript
const obj = { x: 1 };
obj.x = '3'; // Error: Type 'string' is not assignable to type 'number'.
// This error occurs because TypeScript infers the initial type of `x` as number.
```
x??

---

#### Const Assertion for Narrowed Types

The `as const` assertion can be used to narrow down the type inferred by TypeScript, making it more precise. This is useful when you want to ensure that a value holds specific properties and values.

In JavaScript/TypeScript:
```typescript
const obj = { x: 1 } as const;
obj.x = '3'; // Error: Property 'x' does not have a setter.
```
:p How can `as const` be used to make TypeScript infer more precise types?
??x
The `as const` assertion in TypeScript allows you to narrow down the type of an object or array literal so that it reflects its exact properties and values at initialization. This ensures that any reassignment attempts are caught by the compiler, making your code safer.

```typescript
const obj = { x: 1 } as const;
obj.x = '3'; // Error: Property 'x' does not have a setter.
// This error occurs because `as const` makes `x` readonly and infers its exact type to be `{ readonly x: 1; }`.
```
x??

---

#### Mixed Types and Array Inference

For arrays, TypeScript may infer a tuple type if you initialize it with specific values. However, this can lead to issues when trying to add or modify elements later.

In JavaScript/TypeScript:
```typescript
const arr = [1, 2] as const;
arr[0] = 'a'; // Error: Type '"a"' is not assignable to type '1'.
```
:p How does TypeScript infer types for arrays and objects?
??x
For arrays, TypeScript can infer a tuple type when you initialize it with specific values. This ensures that the array has fixed elements of certain types.

```typescript
const arr = [1, 2] as const;
arr[0] = 'a'; // Error: Type '"a"' is not assignable to type '1'.
// The `as const` assertion makes each element's type exact and readonly.
```

For objects, TypeScript infers the best common type based on the properties you define. If more properties are added later, these changes might not be caught by the compiler.

```typescript
const obj = { x: 1 } as const;
obj.x = '3'; // Error: Property 'x' does not have a setter.
// This error occurs because `as const` ensures that `x` is readonly and exactly typed to `{ readonly x: 1; }`.
```
x??

---

#### Const Assertion for Tuple Inference
Background context: Sometimes, TypeScript infers array types when you expect tuple types. Using a `const` assertion can help infer tuple types instead of arrays while allowing elements to widen to their base type.

:p How does using a `const` assertion guide TypeScript to infer tuple types?
??x
A `const` assertion with an array helps TypeScript infer the exact elements, resulting in a tuple type rather than an array. For example:

```typescript
const arr2 = [1, 2, 3] as const; // ^? const arr2: readonly [1, 2, 3]
```

This tells TypeScript to treat each element as a literal value and the whole array as a tuple.
x??

---

#### Using `tuple` Function for Inference
Background context: The `tuple` function can guide TypeScript to infer tuple types. This function serves no runtime purpose but aids in type inference.

:p What is the `tuple` function used for?
??x
The `tuple` function guides TypeScript toward inferring the exact tuple type you want, rather than an array type. It's useful when you need precise control over element types and structure:

```typescript
function tuple<T extends unknown[]>(...elements: T) { return elements; }

const arr3 = tuple(1, 2, 3); // ^? const arr3: [number, number, number]
const mix = tuple(4, 'five', true); // ^? const mix: [number, string, boolean]
```

The `tuple` function here ensures that each element's type is not widened to a broader base type.
x??

---

#### Object.freeze for Readonly Types
Background context: The `Object.freeze` method in JavaScript can be used to enforce readonly properties at runtime. This method introduces readonly modifiers into inferred types.

:p How does `Object.freeze` affect the inferred type?
??x
Using `Object.freeze`, you can make an object's keys readonly, and this affects how TypeScript infers the type:

```typescript
const frozenArray = Object.freeze([1, 2, 3]); // ^? const frozenArray: readonly number[]
const frozenObj = Object.freeze({ x: 1, y: 2 }); // ^? const frozenObj: Readonly<{ x: 1; y: 2; }>
```

While `frozenObj` looks different, its type is exactly the same as if you had used a `const` assertion. However, unlike `const`, it enforces readonly properties at runtime.
x??

---

#### Satisfies Operator for Type Narrowing
Background context: The `satisfies` operator ensures that values meet specific requirements and guides TypeScript to infer precise types.

:p How does the `satisfies` operator help with type inference?
??x
The `satisfies` operator is used to narrow down a value to match a specified type. It helps prevent values from being widened beyond their base type:

```typescript
type Point = [number, number];
const capitals1 = { ny: [-73.7562, 42.6526], ca: [-121.4944, 38.5816] }; // ^? const capitals1: { ny: number[]; ca: number[]; }
const capitals2 = { 
    ny: [-73.7562, 42.6526], 
    ca: [-121.4944, 38.5816] 
} satisfies Record<string, Point>;
capitals2 // ^? const capitals2: { ny: [number, number]; ca: [number, number]; }
```

This ensures that the values are narrowed to `Point` type rather than being widened to `number[]`.
x??

---

