# Flashcards: 10A007---Effective-TypeScript---Dan-Vanderkam_processed (Part 6)

**Starting Chapter:** Item 13 Know the Differences Between type and interface

---

#### Similarities Between Type and Interface in TypeScript
Background context: In TypeScript, both `type` and `interface` are used to define named types. The two constructs share many similarities but have subtle differences that can affect code consistency and readability.

:p What are some similarities between `type` and `interface` in TypeScript?
??x
The two main similarities are:
1. **Excess Property Checking**: Both provide the same excess property checking, meaning if you define a type or interface with additional properties not specified, you will get an error.
2. **Index Signatures**: You can use index signatures in both `type` and `interface`.
3. **Function Types**: Both support defining function types.
4. **Generics Support**: Both constructs can be used as generics.

Code examples:
```typescript
// Using type for a dictionary
type TDict = { [key: string]: string };

// Using interface for a dictionary
interface IDict {
    [key: string]: string;
}

// Defining function types with both type and interface
type TFn = (x: number) => string;
interface IFn {
    (x: number): string;
}
```
x??

---

#### Function Types in Type and Interface
Background context: Both `type` and `interface` can be used to define function types, but the syntax and readability might differ slightly.

:p How do you define a simple function type using both `type` and `interface`?
??x
You can define a simple function type as follows:

Using `type`:
```typescript
type TFn = (x: number) => string;
```

Using `interface`:
```typescript
interface IFn {
    (x: number): string;
}
```
The main difference is that the `type` syntax might be more concise, while the interface form emphasizes the callable nature of functions.

x??

---

#### Generic Types in Type and Interface
Background context: Both `type` and `interface` can support generics to create reusable type definitions. This feature allows you to define types that work with any data type.

:p How do you define a generic type or interface in TypeScript?
??x
You can define a generic type or interface by adding a type parameter within angle brackets `<T>`.

Example using `type`:
```typescript
type TBox<T> = {
    value: T;
};
```

Example using `interface`:
```typescript
interface IBox<T> {
    value: T;
}
```
Both allow you to define types that can work with any type of data, providing flexibility in your code.

x??

---

#### Extending Types and Interfaces
Background context: An interface can extend another interface or a specific object type defined using `type`. However, extending a union type is not allowed. A type cannot directly extend an interface but can be extended through intersection types.

:p Can you explain the differences when it comes to extending types and interfaces in TypeScript?
??x
An interface can extend another interface:
```typescript
interface IStateWithPop extends TState {
    population: number;
}
```

A type can use intersection types to achieve similar behavior:
```typescript
type TStateWithPop = IState & { population: number; };
```
However, a `type` cannot directly extend an interface. The only way to combine types is through intersection types.

x??

---

#### Consistency and Best Practices in Type and Interface Usage
Background context: While both `type` and `interface` share many similarities, it's good practice to be consistent within your codebase about which one you choose for defining a type. Generally, using interfaces is preferred unless there are specific cases where `type` is more appropriate.

:p Why might someone prefer using an interface over a type in TypeScript?
??x
Interfaces are generally preferred because they provide better documentation and consistency with object-oriented programming principles. They make the code more readable and maintainable by clearly indicating that you expect certain properties to be present on objects of this type.

However, `type` is useful for scenarios where:
1. You need a union type (e.g., `type T = 'a' | 'b';`).
2. Function types have a cleaner syntax (`type TFn = (x: number) => string;`).

Consistency in your codebase helps avoid confusion, and following community conventions like not prefixing interfaces with `I` improves readability.

x??

---

#### Type vs Interface: Similarities and Differences
Background context explaining the concept. Both types and interfaces can be recursive (Item 57). Types can be unions, while interfaces cannot. Interfaces can extend other types but not union types.

:p What are some similarities between types and interfaces?
??x
Both types and interfaces share several characteristics such as being able to implement interfaces or simple types. They both can also be recursive according to Item 57.
x??

---

#### Type vs Interface: Differences
Background context explaining the concept. Types have more capabilities than interfaces, including unions and advanced type-level features like mapped types (Item 15) and conditional types (Item 52). Interfaces provide stronger error checking through `extends`.

:p What is a key difference between using types and interfaces in TypeScript?
??x
A key difference is that while both can implement or extend other types, interfaces cannot be unioned together directly. Additionally, extending union types with an interface is not possible, whereas you can use the union operator (`|`) for types.

Examples:
```typescript
// Type alias example using a union
type AorB = 'a' | 'b';

// Interface extension is not allowed
interface IState {
    name: string;
    capital: string;
}

// This would result in an error if you tried to extend with a union type directly
// interface ExtendedState extends IState, Input, Output {} // Error

// Instead, use the & operator for types
type NamedVariable = (Input | Output) & { name: string };
```
x??

---

#### Type vs Interface: Declaration Merging and Augmentation
Background context explaining the concept. Interfaces can be augmented to add or modify properties without changing their base definitions.

:p What is declaration merging in TypeScript?
??x
Declaration merging allows you to extend an existing interface by adding new properties or modifying existing ones within another interface definition block, effectively extending it with additional features while not breaking compatibility.

Example:
```typescript
interface IState {
    name: string;
    capital: string;
}

// Declaration merging example
interface IState {
    population: number; // Adding a new property to the interface
}

const wyoming: IState = {
    name: 'Wyoming',
    capital: 'Cheyenne',
    population: 578_000, // Valid because of declaration merging
};
```
x??

---

#### Type vs Interface: Error Checking and Extensibility
Background context explaining the concept. Interfaces provide stronger error checking compared to types when using `extends`.

:p How do interfaces provide more safety checks than types in TypeScript?
??x
Interfaces provide stronger error checking, especially with properties that have stricter type requirements or incompatible changes.

Example:
```typescript
interface Person {
    name: string;
    age: string; // Note the strict string requirement for 'age'
}

// Type alias example without error
type TPerson = Person & { age: number }; // No error

// Interface extension example with a type mismatch error
interface IPerson extends Person {
    age: number; // Error: Type 'number' is not assignable to type 'string'.
}
```
x??

---

#### Type vs Interface: Expressing Tuple and Array Types
Background context explaining the concept. Type aliases are used naturally to express tuple and array types in TypeScript.

:p How do you represent a tuple of numbers using a type alias in TypeScript?
??x
You can represent a tuple of numbers using a type alias like this:

```typescript
type Pair = [a: number, b: number];
```

This defines `Pair` as an array with two elements where each element is of type `number`.

Example usage:
```typescript
const pair: Pair = [1, 2]; // Valid
// const invalidPair: Pair = ['a', 'b']; // Error: Both elements must be numbers.
```
x??

---

#### Type vs Interface: Declaration Files and Augmentation
Background context explaining the concept. Interfaces are used to support declaration merging in TypeScript, which is primarily used with type declaration files.

:p What is an example of using interfaces for declaration merging?
??x
Declaration merging allows users to add properties or modify existing ones within another interface definition block without changing the original base definitions. This is often used in TypeScript's standard library to model different versions of JavaScript’s standard library.

Example:
```typescript
interface IState {
    name: string;
    capital: string;
}

// Declaration merging example
interface IState {
    population: number; // Adding a new property to the interface
}

const wyoming: IState = {
    name: 'Wyoming',
    capital: 'Cheyenne',
    population: 578_000, // Valid because of declaration merging
};
```
x??

---

---
#### Declaration Merging for Array Interface
Background context: In TypeScript, when targeting different ECMAScript versions (ES5 or ES2015), you get different sets of methods and properties on the `Array` interface due to declaration merging. This means that interfaces can be merged from multiple sources, but this typically occurs in declaration files.
:p How does TypeScript handle the `Array` interface when targeting ES2015?
??x
When targeting ES2015 with TypeScript, it includes both the ES5 definition and the ES2015 core definitions. The ES2015 core adds new methods like `find`, `findIndex`, `fill`, and `copyWithin` to the existing ES5 array interface. This ensures you have a comprehensive set of methods for the target version.
x??

---
#### Declaration Merging in Declaration Files
Background context: Declaration merging is particularly useful in declaration files, where multiple sources can provide interfaces or types that are merged into one. However, this should be used carefully to avoid conflicts with global interface names like `Location` and `FormData`.
:p What is the main purpose of declaration merging?
??x
The main purpose of declaration merging is to combine definitions from multiple sources in a way that they do not conflict with each other or existing global interfaces. This allows for more comprehensive type definitions without issues arising from overlapping or similar names.
x??

---
#### Inlining vs Referencing Types
Background context: TypeScript decides whether to inline types (like `type` aliases) or reference them by name based on where the code is located. Inline behavior can be observed in certain scenarios, but it may cause duplication and affect compiler performance if used excessively.
:p What happens when a type alias is defined within a function body?
??x
When a type alias is defined within a function body, TypeScript will inline the type definition rather than use its name. This means that instead of referencing the `type` alias, the structure directly replaces it. This can be seen in generated `.d.ts` files where the alias doesn't exist.
x??

---
#### Choosing Between Type and Interface
Background context: For simple object types, both `type` and `interface` can be used interchangeably. However, for complex types or when working with an established codebase, it's generally better to use interfaces due to their names appearing more consistently in error messages and type display.
:p In what scenario would you prefer using a `type` alias over an `interface`?
??x
You should prefer using a `type` alias over an interface for simpler object types that can be represented either way. This is because the name of the type alias will appear more consistently in error messages and type display, making it easier to maintain consistency throughout your codebase.
x??

---

#### Interfaces vs. Types in TypeScript
Background context explaining how interfaces and types are used in TypeScript. Explain that interfaces describe blueprints for objects, while types can be more flexible, including unions, tuples, etc.
:p How do you decide between using an interface or a type in TypeScript?
??x
You should use interfaces when the object structure is consistent across multiple objects, as it provides a clear blueprint and easier refactoring. Use types when you need more flexibility, such as when working with union types or complex type definitions that don't necessarily follow a strict object structure.
x??

---

#### Read-Only Modifier in TypeScript
Background context explaining mutable vs. immutable data structures in JavaScript and how TypeScript can help catch mutations using the readonly modifier.
:p How does the readonly modifier work in TypeScript?
??x
The `readonly` modifier prevents assignments to a property after it has been set, ensuring that certain properties remain constant throughout the object's lifecycle. This helps prevent accidental modifications that could lead to bugs.
x??

---

#### Destructive Array Mutation in JavaScript
Background context explaining why arrays are mutable by default in JavaScript and how this can cause issues if not handled carefully.
:p Why does the `arraySum` function modify the array it receives?
??x
The `arraySum` function uses the `pop()` method, which removes the last element from the array and returns its value. This operation mutates the original array by removing elements from it one by one until the array is empty.
x??

---

#### Using Readonly Modifier to Prevent Mutations
Background context explaining how readonly can be used to prevent unintended modifications of objects in TypeScript.
:p How can you use `readonly` to ensure that certain properties remain unchanged?
??x
You can define an object type with a `readonly` property, as shown below. This prevents any assignments to the specified property after its initial assignment:
```typescript
interface ReadOnlyExample {
    readonly id: number;
}

const exampleObject: ReadOnlyExample = { id: 1 };
exampleObject.id = 2; // Error: Cannot assign to 'id' because it is a read-only property.
```
x??

---

#### Immutable Primitives in TypeScript
Background context explaining that JavaScript primitives (strings, numbers, booleans) are immutable by nature and how this differs from objects and arrays.
:p Why are strings, numbers, and booleans considered immutable?
??x
In JavaScript, primitive values like `string`, `number`, and `boolean` are immutable. This means you can't change their value after they have been created; instead, a new value is created when an operation would seemingly modify the existing one. For example:
```typescript
let num = 5;
num++; // A new number object (6) is created, but num still points to the original object.
```
x??

---

#### Consistency in Object Types
Background context explaining that TypeScript prefers interfaces for object types when consistency and clear structure are desired.
:p When should you prefer using an interface over a type?
??x
You should prefer interfaces when defining complex object structures with consistent properties, as they provide better refactoring support and clearer documentation. Interfaces ensure that the expected fields and their types are consistently followed throughout your codebase.
x??

---

#### Readonly Property Modifier and Readonly<T>
The `Readonly<T>` utility type in TypeScript is used to make all properties of an object readonly, meaning they cannot be reassigned. However, it only performs a shallow transformation; nested properties can still be mutated if they are not also marked as readonly.
:p What does the `Readonly<T>` utility type do?
??x
The `Readonly<T>` utility type makes all top-level properties of an object immutable (readonly), meaning they cannot be reassigned. However, it does not make nested objects or arrays within those properties immutable by default; you need to mark them as readonly separately.
```typescript
type Outer = {
    inner: { x: number };
};

const obj: Readonly<Outer> = { inner: { x: 0 } };
obj.inner.x = 1; // OK, but obj.inner can still be reassigned or mutated
```
x??

---

#### Shallow vs. Deep Readonly in TypeScript
When using `Readonly<T>` on an object, it makes the top-level properties immutable (readonly) but does not recursively apply this to nested objects unless those are also marked with `Readonly`. There is no built-in support for deep readonly types, but you can implement one yourself.
:p What is a limitation of using `Readonly<T>`?
??x
A limitation of using `Readonly<T>` is that it only performs a shallow transformation. This means that while top-level properties become readonly, nested objects or arrays within those properties are not affected unless they are also marked with `Readonly`. To achieve deep readonly types, you would need to implement this functionality yourself.
```typescript
type DeepReadonly<T> = {
    readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};

const obj: DeepReadonly<{ a: { b: number } }> = { a: { b: 0 } };
obj.a.b = 1; // Error: Cannot assign to 'b' because it is a read-only property
```
x??

---

#### Readonly with Methods in TypeScript Interfaces
When you apply `Readonly<T>` to an interface that contains methods, the readonly modifier only affects properties. It does not modify or remove any methods defined on the object.
:p How does `Readonly<T>` affect interfaces containing methods?
??x
`Readonly<T>` only modifies top-level properties of an interface, making them immutable (readonly). However, it does not change or remove any methods that are part of the interface. Methods can still mutate the underlying object if they are defined to do so.
```typescript
interface Outer {
    inner: { x: number };
}

const obj: Readonly<Outer> = { inner: { x: 0 } };
obj.inner.x = 1; // OK, because only properties are made readonly
obj.inner = { x: 1 }; // Error: Cannot assign to 'inner' because it is a read-only property
```
x??

---

#### Mutable and Immutable Versions of Classes in TypeScript
The standard library provides both mutable and immutable versions of common classes like `Array`. The mutable version (`Array<T>`) contains methods that can mutate the underlying array, while the immutable version (`ReadonlyArray<T>`) does not.
:p What are the key differences between `Array<T>` and `ReadonlyArray<T>`?
??x
The key differences between `Array<T>` and `ReadonlyArray<T>` are:
1. **Mutating Methods**: `Array<T>` has methods like `pop()`, `shift()`, etc., which can mutate the underlying array.
2. **Readonly Modifier**: `ReadonlyArray<T>` marks properties like `length` as readonly and does not define any mutating methods, preventing modifications to the array.

```typescript
const a: number[] = [1, 2, 3];
const b: readonly number[] = a;
b.pop(); // OK, but mutates 'a'
// const c: number[] = b; // Error: Type 'readonly number[]' is 'readonly' and cannot be assigned to the mutable type 'number[]'
```
x??

---

#### Subtype Relationship Between `T[]` and `readonly T[]`
In TypeScript, an array of a certain type (`T[]`) is considered strictly more capable than a readonly array of that type (`readonly T[]`). This means that any array can be assigned to a variable of the readonly array type, but not vice versa.
:p What is the subtype relationship between `T[]` and `readonly T[]`?
??x
The subtype relationship in TypeScript is such that an array of a certain type (`T[]`) is considered strictly more capable than a readonly array of that type (`readonly T[]`). This means:
- You can assign a mutable array to a variable of the readonly array type.
- You cannot assign a readonly array to a variable of the mutable array type.

```typescript
const a: number[] = [1, 2, 3];
const b: readonly number[] = a; // OK, because 'a' is an array and can be assigned to 'readonly'
const c: number[] = b; // Error: Type 'readonly number[]' is 'readonly' and cannot be assigned to the mutable type 'number[]'
```
x??

#### Readonly Arrays and Function Parameters

In TypeScript, when you want to prevent a function from mutating an array passed as a parameter, you can use `readonly` arrays. This ensures that callers understand that the array will not be modified within the function.

Background context: When you declare an array as `readonly`, it prevents any operations that would mutate the array, such as `push`, `pop`, etc., but allows for iteration over its elements. This is useful in scenarios where you want to ensure that data integrity or immutability is maintained during function execution.

:p What happens when a parameter is declared as `readonly` in TypeScript?
??x
Declaring a parameter as `readonly` means that the array will not be mutated within the function body. The type checker ensures that no operations like `push`, `pop`, etc., are performed on the array, but it allows for iteration using loops or other read-only operations.

For example:
```typescript
function printTriangles(n: number) {
    const nums = [];
    for (let i = 0; i < n; i++) {
        nums.push(i);
        console.log(arraySum(nums as readonly number[]));
    }
}

function arraySum(arr: readonly number[]) {
    let sum = 0;
    for (const num of arr) {
        sum += num;
    }
    return sum;
}
```
x??

---

#### Error Handling with Readonly Arrays

When working with `readonly` arrays, certain operations like `pop` are not allowed because they mutate the array. To fix such type errors, you can use a simple loop instead.

Background context: TypeScript differentiates between mutable and immutable collections using keywords like `readonly`. Using `readonly` ensures that only read operations can be performed on the collection. However, this comes with the constraint that certain methods (like `pop`) are not available for `readonly` arrays.

:p What error occurs when trying to use `pop` on a `readonly` array in TypeScript?
??x
The error that occurs is: `'pop' does not exist on type 'readonly number[]'.` This happens because the `pop` method, which mutates the array by removing and returning its last element, is not defined for objects of type `readonly`.

For example:
```typescript
function arraySum(arr: readonly number[]) {
    let sum = 0;
    while ((num = arr.pop()) !== undefined) { // Error here
        // sum += num; // This would be invalid as 'pop' does not exist on 'readonly number[]'
    }
    return sum;
}
```
x??

---

#### Readonly vs. Const

`const` and `readonly` serve different purposes in TypeScript. While `const` prevents reassignment of the variable, `readonly` prevents mutation but allows for reassignment.

Background context: In JavaScript/TypeScript, understanding the differences between `const`, `let`, and `readonly` is crucial to avoid common pitfalls related to mutability. `const` makes a value immutable (cannot be reassigned), whereas `readonly` ensures that an object or array cannot be mutated but can still be reassigned.

:p What are the key differences between `const` and `readonly` in TypeScript?
??x
The key differences are:
- `const`: Prevents reassignment of the variable. The value itself is immutable.
- `readonly`: Ensures that the variable's type (like an array) cannot be mutated but allows for reassignment.

For example:
```typescript
let arr1: number[] = [1, 2, 3];
// const arr2: readonly number[] = [1, 2, 3]; // This is not valid as 'readonly' only applies to the type, not the value itself.
arr1.push(4); // Valid since it's a regular array
// (arr2 as any).push(5); // This would be invalid if allowed

const arr2: number[] = [1, 2, 3];
arr2 = [4, 5, 6]; // Valid reassignment of the value
```
x??

---

#### Function Contracts and Readonly Parameters

Declaring parameters as `readonly` in functions can improve type safety and make function contracts clearer. This ensures that callers understand which parts of an object might be mutated.

Background context: When a function is called, it’s important to communicate what modifications might occur. Declaring parameters as `readonly` helps in maintaining immutability principles and making the function contract explicit.

:p Why should you declare parameters as `readonly` if they are not intended to be modified?
??x
You should declare parameters as `readonly` if they are not intended to be modified because it makes the function's contract clearer, preventing accidental mutations. It also helps in maintaining immutability principles and ensuring that callers understand which parts of an object might or might not change.

For example:
```typescript
function printTriangles(n: number) {
    const nums = [];
    for (let i = 0; i < n; i++) {
        nums.push(i);
        console.log(arraySum(nums as readonly number[]));
    }
}

function arraySum(arr: readonly number[]) {
    let sum = 0;
    for (const num of arr) {
        sum += num;
    }
    return sum;
}
```
x??

---

#### Type Safety and Readonly Parameters

TypeScript’s `readonly` keyword helps in ensuring type safety by preventing operations that would mutate the passed array.

Background context: When a function is called, it's crucial to ensure that its parameters remain unchanged. Using `readonly` ensures that only read operations can be performed on the array, thus maintaining data integrity and preventing unintended modifications.

:p How does using `readonly` for an array parameter improve type safety?
??x
Using `readonly` for an array parameter improves type safety by ensuring that only read-only operations (like iteration) can be performed on the array. This prevents any mutations such as `push`, `pop`, etc., which could alter the data integrity of the original array.

For example:
```typescript
function printTriangles(n: number) {
    const nums = [];
    for (let i = 0; i < n; i++) {
        nums.push(i);
        console.log(arraySum(nums as readonly number[]));
    }
}

function arraySum(arr: readonly number[]) {
    let sum = 0;
    for (const num of arr) {
        sum += num;
    }
    return sum;
}
```
x??

---

#### DRY Principle: Don’t Repeat Yourself

**Background context:** The DRY principle, which stands for "Don't Repeat Yourself," is a software development principle that suggests every piece of knowledge should have a single, unambiguous, authoritative representation within a system. This helps to avoid duplication of information and ensures consistency.

In the provided code example, there are several instances where the same calculations (surface area and volume) are repeated with different values. By defining functions for these calculations, you can reduce redundancy and improve maintainability of your code.

:p How does the DRY principle apply to reducing repetition in JavaScript/TypeScript code?
??x
The DRY principle applies by encouraging developers to refactor their code into reusable functions or constants instead of repeating the same logic multiple times. In the given example, we define `surfaceArea` and `volume` functions that take radius (`r`) and height (`h`) as parameters. This way, you can easily reuse these functions for different cylinders without rewriting the calculations.

```typescript
type CylinderFn = (r: number, h: number) => number;

const surfaceArea: CylinderFn = (r, h) => 2 * Math.PI * r * (r + h);
const volume: CylinderFn = (r, h) => Math.PI * r * r * h;
```

```typescript
for (const [r, h] of [[1, 1], [1, 2], [2, 1]]) {
    console.log(
        `Cylinder r=${r} × h=${h}`,
        `Surface area: ${surfaceArea(r, h)}`,
        `Volume: ${volume(r, h)}`
    );
}
```
x??

---

#### Type Duplication and Factoring

**Background context:** In TypeScript or other statically typed languages, duplication in types can lead to inconsistencies similar to code repetition. Just as you would factor out repeated code into functions, you should also consider factoring out common type patterns.

In the provided example, two interfaces (`Person` and `PersonWithBirthDate`) are defined with a lot of overlap but some differences (like an optional field in one). This can lead to bugs or inconsistencies if not managed carefully. By factoring out shared parts, you ensure that changes propagate correctly across related types.

:p How does type duplication affect code maintainability?
??x
Type duplication affects code maintainability by introducing redundancy and potential inconsistencies. When you have overlapping interfaces with slight differences (like adding an optional field in one but not the other), it becomes harder to manage updates or introduce new fields consistently. This can lead to bugs if developers don't notice that a type is missing something.

For example, consider the `Person` and `PersonWithBirthDate` interfaces:
```typescript
interface Person {
    firstName: string;
    lastName: string;
}

interface PersonWithBirthDate extends Person {
    birth: Date;
}
```

Here, updating `Person` might not always be reflected in `PersonWithBirthDate`, leading to potential issues.

Factoring out shared parts can help maintain consistency:
```typescript
interface BasePerson {
    firstName: string;
    lastName: string;
}

interface Person extends BasePerson {
}

interface PersonWithBirthDate extends BasePerson {
    birth: Date;
}
```
x??

---

#### Using Interfaces for Common Patterns

**Background context:** Interfaces in TypeScript allow you to define the structure of objects and can be used to create common patterns. By defining a base interface that is extended by other more specific interfaces, you can reduce duplication and ensure consistency across related types.

In the example provided, we can see how a `Point2D` interface simplifies distance calculations compared to writing out the same object shape multiple times.

:p How does using an interface like `Point2D` help in reducing code duplication?
??x
Using an interface like `Point2D` helps reduce code duplication by defining a common structure that can be reused. Instead of repeatedly writing the `{ x: number; y: number }` pattern, you define it once and reuse it.

For example:
```typescript
interface Point2D {
    x: number;
    y: number;
}

function distance(a: Point2D, b: Point2D) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}
```

This approach makes your code more readable and maintainable. If you need to change the structure of a point or add methods, you only need to update the `Point2D` interface.

```typescript
interface Point2D {
    x: number;
    y: number;

    // Add new properties or methods here
}
```
x??

---

#### Factoring Shared Function Signatures

**Background context:** When several functions share the same type signature, it's a good practice to factor out this shared pattern into a named type. This can simplify function definitions and make your code more readable.

In the example provided, we have two HTTP functions (`get` and `post`) that both take similar arguments and return the same promise type. By factoring out their common signature, you can define a single type for them to use.

:p How does factoring shared function signatures improve code organization?
??x
Factoring shared function signatures improves code organization by making your functions more concise and easier to understand. It also centralizes the definition of common types or interfaces, reducing redundancy and ensuring consistency across related functions.

For example:
```typescript
type HTTPFunction = (url: string, opts: Options) => Promise<Response>;

const get: HTTPFunction = (url, opts) => { /* ... */ };
const post: HTTPFunction = (url, opts) => { /* ... */ };
```

Here, both `get` and `post` functions share the same type signature. By defining a named type (`HTTPFunction`), you avoid repeating the function parameters everywhere.

```typescript
interface Options {
    // options definition here
}

type HTTPFunction = (url: string, opts: Options) => Promise<Response>;

const get: HTTPFunction = (url, opts) => { /* ... */ };
const post: HTTPFunction = (url, opts) => { /* ... */ };
```
x??

---

#### Reducing Code Duplication with Interfaces and Types

Background context: In TypeScript, it is common to encounter situations where you have two interfaces or types that share many fields. Instead of duplicating these shared fields, you can reduce redundancy by using inheritance (interfaces) or intersections.

If the two interfaces share a subset of their fields, you can factor out a base interface with just these common fields. This allows you to avoid rewriting duplicated code and makes maintenance easier.

:p How can you use interfaces to avoid code duplication between `Person` and `PersonWithBirthDate`?
??x
You can define `Person` as the base interface containing shared properties, then extend it for `PersonWithBirthDate`. Here’s how:

```typescript
interface Person {
    firstName: string;
    lastName: string;
}

interface PersonWithBirthDate extends Person {
    birth: Date;
}
```

This approach ensures that changes to the common fields in `Person` will be automatically applied to `PersonWithBirthDate`.

x??

---

#### Using Mapped Types for Subset Interfaces

Background context: Sometimes, you need a subset of properties from another interface or type. Directly extending the original type might not always fit your needs because it adds all properties from the base interface.

Using mapped types allows you to select specific fields from an existing interface or type and create a new one based on them.

:p How can you define `TopNavState` as a subset of `State` using a mapped type?
??x
You can use a mapped type to pick out selected properties from the base `State` interface. Here’s how:

```typescript
type TopNavState = {
    [K in 'userId' | 'pageTitle' | 'recentFiles']: State[K];
};
```

This syntax iterates over the keys specified and creates a new object with those key-value pairs.

x??

---

#### Extending Interfaces vs. Intersection Types

Background context: In TypeScript, you can extend an interface to add new properties or methods while retaining all existing ones from the base interface. Alternatively, you can use intersection types to combine multiple types into one.

However, extending an interface is more common and straightforward for adding fields compared to using intersection types.

:p How do you define `TopNavState` by extending `State`, instead of creating a mapped type?
??x
You can extend the base `State` interface directly with additional properties in `TopNavState`. Here’s how:

```typescript
interface State {
    userId: string;
    pageTitle: string;
    recentFiles: string[];
    pageContents: string;
}

interface TopNavState extends State {
    // No new fields needed since we are extending the full State
}
```

In this case, `TopNavState` inherits all properties from `State`. You only need to extend it if you want to add new unique fields.

x??

---

#### Using Intersections for Combining Types

Background context: When you have a complex type like `PersonWithBirthDate`, which includes properties of two different interfaces (`Person` and an object with the birth date), you can use intersection types to combine them easily.

Intersection types allow you to merge multiple types into one, keeping all their properties together.

:p How do you define `PersonWithBirthDate` using intersection types?
??x
You can use the intersection operator (&) to combine `Person` and an object with the birth date. Here’s how:

```typescript
type PersonWithBirthDate = Person & { birth: Date; };
```

This approach combines all properties from both `Person` and the specified object, providing a single type that includes all these fields.

x??

---

#### Mapped Types and Generic Types
Mapped types allow you to transform one type into another by iterating over its keys. This is particularly useful for creating subsets of existing interfaces, making your code more concise and avoiding duplication. The generic type `Pick<T, K>` from the TypeScript standard library does this but isn't fully explained here.

:p What is a mapped type in TypeScript?
??x
A mapped type in TypeScript is a type that iterates over the keys of an existing type to transform or filter its properties into a new type. It's akin to looping over fields in an array, where each property can be updated or filtered based on certain conditions.
x??

---

#### Using Pick for Type Selection
`Pick<T, K>` allows you to select specific properties from an interface `T`, making it easier to work with subsets of larger types without duplicating the entire type.

:p How do you use the `Pick` utility type in TypeScript?
??x
You use `Pick` by specifying the original type and the keys you want to retain. For example, if you have a state object with multiple properties but only need some of them:

```typescript
type State = {
    userId: string;
    pageTitle: string;
    recentFiles: File[];
};

// Selecting specific fields from the State interface
type TopNavState = Pick<State, 'userId' | 'pageTitle' | 'recentFiles'>;

// TopNavState is now { userId: string; pageTitle: string; recentFiles: File[]; }
```
x??

---

#### Extracting Tag Types from Union Types
Tagged unions are a common pattern in TypeScript where you have multiple types with a shared property called `type`. You can use indexing to extract just the type.

:p How do you extract the tag type from an action union using indexing?
??x
You can extract the tag type by indexing into the union:

```typescript
interface SaveAction {
    type: 'save';
    // ...
}

interface LoadAction {
    type: 'load';
    // ...
}

type Action = SaveAction | LoadAction;

// Extracting the 'type' property from the union
type ActionType = Action['type'];
```
In this case, `ActionType` would be `'save' | 'load'`.

Alternatively, you can use a mapped type to achieve the same result:

```typescript
type ActionType = Pick<Action, 'type'>;
// This results in { type: "save" | "load"; }
```

However, indexing directly is more concise.
x??

---

#### Creating Partial Types for Update Methods
When designing classes with constructor and update methods, it's common to have similar parameters. The `Partial<T>` utility type can be used to make properties optional.

:p How do you create a partial type using mapped types?
??x
You can use a mapped type to make all the properties of an interface optional:

```typescript
interface Options {
    width: number;
    height: number;
    color: string;
    label: string;
}

// Creating a partial type for options
type OptionsUpdate = {[k in keyof Options]?: Options[k]};
```

Here, `OptionsUpdate` will be `{ width?: number; height?: number; color?: string; label?: string; }`.

Alternatively, you can use the built-in `Partial<T>` utility:

```typescript
class UIWidget {
    constructor(init: Options) { /* ... */ }
    update(options: Partial<Options>) { /* ... */ }
}
```
Both methods achieve the same result but using a mapped type gives you more control over which properties are made optional.
x??

---

#### Inverting Key-Value Pairs with Mapped Types
You can use mapped types to invert key-value pairs in an object, which is useful for creating reverse mappings or other transformations.

:p How do you create a reversed mapping of key-value pairs using a mapped type?
??x
To invert the keys and values in an object:

```typescript
interface ShortToLong {
    q: 'search';
    n: 'numberOfResults';
}

type LongToShort = { [k in keyof ShortToLong as ShortToLong[k]]: k };

// Resulting type is { search: "q"; numberOfResults: "n" }
```

The `as` clause in the mapped type allows you to specify a transformation on the keys. In this case, it's transforming values back into keys.
x??

---

#### Homomorphic Mapped Types
Background context: In TypeScript, when you use `keyof` to create a mapped type, it can behave as homomorphic. This means that certain properties like `readonly` and `?` (optional) are transferred over to the new type. This behavior is distinct from non-homomorphic types.
:p What does "homomorphic" mean in TypeScript's context of mapped types?
??x
Homomorphic mapped types are those where modifiers such as `readonly` and `optional` (`?`) are preserved when creating a new type through mapping operations that use `keyof`. This is distinct from non-homomorphic types, which do not transfer these properties.
x??

---
#### Non-Homomorphic Mapped Types
Background context: A mapped type is considered non-homomorphic if it doesn't use `keyof` to define the key set. In such cases, modifiers and documentation are not transferred over to the new type. This distinction helps in understanding how TypeScript handles different types of mappings.
:p How does a non-homomorphic mapped type differ from a homomorphic one?
??x
A non-homomorphic mapped type does not transfer modifiers like `readonly` or optional status (`?`) when creating a new type, unlike homomorphic types which do. This behavior is determined by whether the key set in the mapped type uses `keyof`.
x??

---
#### Partial Type with Primitives
Background context: TypeScript's `Partial<T>` utility type creates an object where all properties of T are optional, even if they were not originally marked as such. However, this process does not modify primitive types.
:p Why doesn't `Partial<number>` change the `number` type?
??x
The `Partial<number>` operation does not affect the `number` type because `number` is a primitive type in TypeScript. Primitives are treated specially and do not undergo the same transformations as objects when using utility types like `Partial`.
x??

---
#### Using `typeof` for Type Inference
Background context: The `typeof` operator in TypeScript can be used to infer a type based on an existing value or object literal. This is useful for creating precise types without manually typing out all properties.
:p How does the `typeof` keyword work for inferring types?
??x
The `typeof` keyword in TypeScript infers the type of a value by looking at its structure and then creates a new type based on that structure. For example, using `typeof INIT_OPTIONS`, you can get an exact type representation of an object literal without having to manually define it.
x??

---
#### Deriving Types from Values
Background context: While deriving types directly from values can be convenient, it is generally better practice to define the type first and then declare that a value conforms to this type. This approach avoids issues with widening (Item 20) and makes your code more explicit.
:p What are the potential drawbacks of deriving a type directly from a value?
??x
Deriving a type directly from a value can lead to issues with widening, where TypeScript may widen the inferred type unexpectedly. Defining the type first and then declaring that values conform to it helps maintain type safety and clarity in your code.
x??

---
#### Using `typeof` for Function Return Types
Background context: You can use `typeof` to infer the return type of a function or method, which is particularly useful when you want to avoid duplicating type definitions. The standard library often provides generic types that follow common patterns.
:p How can `typeof` be used to create precise return types for functions?
??x
You can use `typeof` with a function to infer its return type directly from the structure of the object it returns. For example, using `typeof getUserInfo`, you can get an exact representation of the return type without manually typing out all properties.
x??

---

#### ReturnType Generic Usage
Background context explaining the `ReturnType` generic. It is used to extract the return type of a function based on its static type, not its value.

:p How does the `ReturnType` generic work?
??x
The `ReturnType` generic operates on the type of the function itself (using `typeof`) rather than the function's execution result. This means it focuses on the static type information to infer what the function returns.

```typescript
// Example usage
type UserInfo = ReturnType<typeof getUserInfo>;

function getUserInfo(): { id: number; name: string } {
  return { id: 1, name: 'John Doe' };
}
```
x??

---

#### Using typeof with Functions
Background context explaining how `typeof` is used to get the static type of a function. This is different from calling the function and getting its result.

:p How does using `typeof` help in working with functions?
??x
Using `typeof` allows you to inspect the static type information of a function without executing it. It returns the signature (including parameters and return types) of the function, which can be useful for type inference or for generating generic utilities like `ReturnType`.

```typescript
type MyFunction = typeof getUserInfo;
```
x??

---

#### Judicious Use of Abstractions
Background context explaining why creating abstractions should be done with care. Too much abstraction can lead to inflexibility in the future if underlying types or functions change.

:p What are some guidelines for creating useful abstractions?
??x
When creating abstractions, ensure they add value and don’t overcomplicate things. A good rule of thumb is that if it’s hard to name a type (or function) meaningfully, then it may not be worth the abstraction.

For example:

```typescript
interface NamedAndIdentified {
    id: number;
    name: string;
}

// This can lead to issues:
interface Product extends NamedAndIdentified {
    priceDollars: number;
}

interface Customer extends NamedAndIdentified {
    address: string;
}
```
This is problematic because `id` and `name` might evolve differently in the future, making the abstraction less flexible.

x??

---

#### Duplication vs. Abstraction
Background context explaining that while duplication can be bad, over-abstraction can also introduce unnecessary complexity.

:p When should you avoid creating abstractions?
??x
Create abstractions when they add meaningful value and make your codebase more maintainable. Avoid premature abstractions where the underlying types or functions might change independently in the future.

For example:

```typescript
// Bad abstraction:
interface NamedAndIdentified {
    id: number;
    name: string;
}

// Good abstraction:
interface Vertebrate {
    boneCount: number;
}
```
The `Vertebrate` interface is meaningful on its own, while `NamedAndIdentified` just describes the structure of the type.

x??

---

#### Don't Repeat Yourself (DRY)
Background context explaining the principle of DRY in both value and type space. This applies to reducing redundancy not only in code but also in types.

:p Why should we avoid repeating ourselves in TypeScript?
??x
Avoiding repetition in your TypeScript code is important because it reduces maintenance overhead, makes refactoring easier, and can prevent bugs caused by inconsistent updates. While repetition is bad, over-abstraction can introduce unnecessary complexity and make the code harder to evolve.

```typescript
// Example of DRY violation:
interface Product {
    id: number;
    name: string;
    priceDollars: number;
}

interface Customer {
    id: number; // This could change in the future
    name: string;
    address: string;
}
```
x??

