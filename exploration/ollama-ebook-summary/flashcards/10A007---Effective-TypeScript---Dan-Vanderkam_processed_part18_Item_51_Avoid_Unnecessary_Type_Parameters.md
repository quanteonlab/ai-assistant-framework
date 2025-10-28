# Flashcards: 10A007---Effective-TypeScript---Dan-Vanderkam_processed (Part 18)

**Starting Chapter:** Item 51 Avoid Unnecessary Type Parameters

---

#### Type Parameters Should Appear Twice - Golden Rule of Generics
Background context explaining the concept. The "Golden Rule of Generics" is a principle to ensure that generic type parameters are used effectively by appearing at least twice within a function signature, which helps with type inference and makes code more readable.
:p What is the golden rule of generics in TypeScript?
??x
The golden rule states that if a type parameter appears only once in a function signature, it should be reconsidered whether it's necessary. This rule ensures that each type parameter relates to at least two different types within the function, facilitating better type inference and making the code more comprehensible.
```typescript
function identity<T>(arg: T): T {
    return arg;
}
```
In this example, `T` appears twice in the function signature (once as a parameter type and once as a return type), adhering to the rule. 
x??

---

#### Generic Types as Functions Between Types
The concept of generic types being analogous to functions that operate on types is introduced. This analogy helps in understanding how generics can be used to define reusable, type-safe code.
:p How are generic types similar to regular functions?
??x
Generic types act like functions between types. Just as a function might take parameters and return a result, a generic type takes type parameters and produces a new type based on those parameters. This analogy helps in designing more flexible and reusability-oriented TypeScript code.

For example:
```typescript
type MapValues<T extends object, F> = {
    [K in keyof T]: F<T[K]>;
};
```
Here, `MapValues` is a generic type that takes two type arguments: `T`, which must be an object type, and `F`, a function type. The result is another type where each key of the input object maps to the output of applying `F` to the corresponding value.
x??

---

#### Constraining Type Parameters with Extends
The use of `extends` in TypeScript generics allows for constraining type parameters, similar to how type annotations constrain function parameters.
:p How does extending work in generic types?
??x
Extending is used to define constraints on type parameters. When you declare a generic parameter with `extends`, you specify that the type must conform to certain interfaces or classes.

For example:
```typescript
type MapValues<T extends object, F> = {
    [K in keyof T]: F<T[K]>;
};
```
Here, `T` is constrained to be an object using `extends object`. This ensures that only objects can be used as the first type argument when defining a `MapValues` instance.
x??

---

#### Anonymous Generic Types
The absence of anonymous generic types in TypeScript means that all generic types must have explicit names. This design choice simplifies reasoning about and documenting code.
:p Why are there no anonymous generic types in TypeScript?
??x
Anonymous generic types, which would allow defining generics without naming them explicitly, do not exist in TypeScript because the language designers chose to require explicit type parameter declarations for clarity and maintainability.

For instance, consider this hypothetical example:
```typescript
// Hypothetical Anonymous Generic Type
let values: <T>(arr: T[]) => T[] = (arr) => arr;
```
In actual TypeScript, you must explicitly name the generic parameters:
```typescript
function mapValues<T>(arr: T[]): T[] {
    // Function implementation here
}
```
This explicit naming helps in documenting and reasoning about the type relationships more clearly.
x??

---

#### Legibility of Generic Types through Parameter Naming
Choosing clear names for type parameters can significantly improve code readability. Additionally, writing `TSDoc` comments can help document these generic types effectively.
:p How do you choose a good name for a type parameter?
??x
Choosing a good name for a type parameter involves selecting identifiers that clearly indicate the role or nature of the type in the context of its usage. This improves code readability and maintainability.

For example, instead of using `T`:
```typescript
type MapValues<T extends object, F> = {
    [K in keyof T]: F<T[K]>;
};
```
Using a more descriptive name like `Item` could be better:
```typescript
type MapValues<Item extends object, Mapper> = {
    [K in keyof Item]: Mapper<Item[K]>;
};
```
This makes the purpose of the type parameter (`Item`) and its role clearer to other developers reading the code.
x??

---

#### Type Inference with Generic Functions and Classes
Generic functions and classes are seen as defining generic types that can be conducive to better type inference. This is beneficial for both the author and users of such generic constructs.
:p How do generic functions help with type inference?
??x
Generic functions and classes contribute positively to type inference by allowing TypeScript to deduce more precise and accurate types when used in various contexts. By using generics, you can create reusable code that works well with a wide variety of data types.

For example:
```typescript
function identity<T>(arg: T): T {
    return arg;
}
```
Using the `identity` function, TypeScript can infer the correct type for both the argument and the result based on how it's used. This reduces redundancy in type annotations and enhances code maintainability.
x??

---

#### Generics Usage Analysis
Generics provide a way to make your functions and classes reusable across different data types. However, using generics effectively requires understanding when they are necessary and how to structure them properly.

:p How does the presence of type parameters affect the validity of a function's use of generics?
??x
A function is valid in terms of its generic usage if each type parameter appears at least twice: once as an input type and once as an output or internal processing type. This ensures that the type system can infer relationships between types correctly.

For example, consider the following invalid function:
```typescript
function third<A, B>(a: A, b: B): A {
    return a;  // Error: A only appears once.
}
```
This fails because `A` is used as an input but not involved in any processing or output. Rewriting it with a relevant type parameter usage:
```typescript
function third<C>(a: C, b: C, c: C): C {
    return c;  // Now C appears both as input and output.
}
```
This makes the function valid by ensuring `C` is used consistently.

x??

---
#### YAML Parsing Function Generics

:p How should a generics declaration be structured for the `parseYAML` function to ensure type safety?
??x
The `parseYAML` function should use a generic return type, but it's important that the type parameter appears in both input and output positions. The initial declaration:
```typescript
declare function parseYAML<T>(input: string): T;
```
is problematic because `T` only appears as an output, leading to potential misuse without explicit type assertions.

To fix this, change the return type to `unknown`, ensuring users must assert types explicitly:
```typescript
declare function parseYAML(input: string): unknown;
const w = parseYAML('') as Weight;  // User must provide a type assertion.
```
This approach avoids false illusions of safety and enforces explicit typing.

x??

---
#### Property Printing Function Generics

:p How can the `printProperty` function be refactored to improve its use of generics?
??x
The initial `printProperty` function:
```typescript
function printProperty<T, K extends keyof T>(obj: T, key: K) {
    console.log(obj[key]);
}
```
is a bad use of generics because `K` only appears once in the parameter list. To improve this, move the type constraint into the parameter type and eliminate the separate generic type for `K`:
```typescript
function printProperty<T>(obj: T, key: keyof T) {
    console.log(obj[key]);
}
```
This ensures that `keyof T` is used in both input and output positions, making it a better use of generics. The inferred return type remains consistent with the original function's logic.

x??

---
#### Class Generics Implementation

:p How does the `ClassyArray` class exemplify good usage of generics?
??x
The `ClassyArray` class demonstrates effective generic usage because its type parameter `T` appears multiple times in the implementation. This ensures strong type relationships between properties and methods, maintaining type safety.

Here's the class definition:
```typescript
class ClassyArray<T> {
    arr: T[];
    constructor(arr: T[]) { this.arr = arr; }
    get(): T[] { return this.arr; }
    add(item: T) { this.arr.push(item); }
    remove(item: T) {
        this.arr = this.arr.filter(el => el != item);
    }
}
```
The type parameter `T` is used in the constructor, property definition, and method parameters, ensuring consistent typing throughout. This makes the class robust and type-safe.

x??

#### Joiner Class and Generic Methods
Background context: The example provided discusses how to properly implement a generic method within a class. It emphasizes that type parameters should be used only where necessary, as they can make code more complex without providing additional benefits.

:p How does moving a type parameter from the class level to the method level improve type inference in TypeScript?

??x
Moving the type parameter `T` from the class level to the method level allows TypeScript to infer the type of `T` directly from its usage within the method. This makes the code more concise and easier to understand, as the type relationship is established closer to where it is used.

```typescript
class Joiner {
    join<T extends string | number>(els: T[]): string {
        return els.map(el => String(el)).join(',');
    }
}
```
x??

---

#### Simplifying Generic Methods with Non-Generic Types
Background context: The text suggests that if a type parameter only appears once, it can be removed and replaced with a concrete type or interface. This simplifies the code while still maintaining type safety.

:p In what scenario would you remove a generic type parameter from a method?

??x
You should remove a generic type parameter when it only appears once in the method body and is not reused elsewhere. Removing the type parameter can simplify the code, making it easier to understand and maintain without losing its essential functionality.

```typescript
function join(els: (string | number)[]): string {
    return els.map(el => String(el)).join(',');
}
```
x??

---

#### Standalone Functions vs. Wrapper Classes
Background context: The text advocates for using standalone functions over wrapper classes when possible, as this is more idiomatic in JavaScript and easier to understand.

:p Why might a generic class be unnecessary in favor of a standalone function?

??x
A generic class may be unnecessary if the functionality can be encapsulated within a standalone function. This is particularly true in JavaScript, where functions are first-class citizens and can be used without the overhead of classes. Using a standalone function simplifies the code and improves readability.

```typescript
function join(els: (string | number)[]): string {
    return els.map(el => String(el)).join(',');
}
```
x??

---

#### Lengthy Interface with Generics
Background context: The example demonstrates how to use generics for a length property, but notes that if the type parameter only appears once, it can be simplified.

:p Why is using a generic type parameter unnecessary in this case?

??x
Using a generic type parameter `T` when it only appears once and does not add any additional utility or flexibility is unnecessary. It can be replaced with a more straightforward approach, such as directly using the interface definition.

```typescript
interface Lengthy {
    length: number;
}

function getLength(x: Lengthy): number {
    return x.length;
}
```
x??

---

#### Overloading Functions with Generics
Background context: The text provides an example of how to handle situations where type parameters are extraneous and can be simplified. It also discusses the implications on function implementation.

:p How does using `unknown` instead of a generic type parameter affect the implementation?

??x
Using `unknown` as a type parameter in the function signature allows for more flexible assignment within the function body, as it removes the restriction that types must match exactly. This can make the code easier to implement and read without losing type safety.

```typescript
function processUnrelatedTypes(a: unknown, b: unknown): void {
    a = b; // now allowed
    b = a; // also allowed
}
```
x??

---

#### Applying Generics Wisely
Background context: The text concludes with the rule of "don't" use generics where they are unnecessary. It emphasizes that type parameters should be used only when necessary to establish relationships between types.

:p What is the main principle discussed in this section?

??x
The main principle is to avoid using generic type parameters when they are not necessary. This helps keep the code simpler and more maintainable, as extraneous type parameters can make the code harder to understand without providing any additional benefits.

```typescript
function getLength(x: ArrayLike<unknown>): number {
    return x.length;
}
```
x??

---

#### Overload Signatures vs Conditional Types

Background context: This concept discusses the nuances between using overload signatures and conditional types to define function types in TypeScript. Overload signatures provide a straightforward approach, but can sometimes be too precise or imprecise. On the other hand, conditional types offer a more flexible and generalized solution.

:p How does overloading work in TypeScript when defining function types?
??x
Overloading in TypeScript allows you to declare multiple definitions for a single function based on its parameters. Each definition specifies a unique set of parameter types and return types. During type checking, TypeScript tries each overload one by one until it finds the best match.
```typescript
declare function double(x: number): number;
declare function double(x: string): string;
```
In this example, `double` can be called with either a `number` or a `string`. The specific overload that matches the argument type is used for type checking and inference.

x??

---

#### Conditional Types

Background context: Conditional types are a powerful feature in TypeScript that act like if-else statements at the type level. They allow you to define return types based on conditions, making your function types more precise and flexible.

:p How do conditional types work in defining function return types?
??x
Conditional types enable you to specify different return types based on whether a type condition is true or false. The syntax for a conditional type looks like this: `T extends U ? A : B`. This means if the type `T` is assignable to `U`, then the result of the conditional type is `A`; otherwise, it's `B`.

Example:
```typescript
declare function double<T extends string | number>(x: T): T extends string ? string : number;
```
In this example, if `T` (the type of the input parameter) is a subtype of `string`, the return type will be `string`. Otherwise, it will be `number`.

x??

---

#### Overload Signatures and Union Types

Background context: This concept illustrates why overloading signatures can sometimes lead to imprecise types. It also explains how using conditional types can provide more accurate and generalized solutions.

:p Why might an overload signature for a function that accepts either a string or a number result in inaccurate type inference?
??x
Overload signatures for functions that accept a union of types (like `string | number`) can sometimes produce inaccurate type inference because the final overload might match only one part of the union, leading to incorrect return types. For instance:

```typescript
declare function double(x: string | number): string | number; // Inaccurate
```
This signature means that when `double` is called with a `number`, it will return a `number`. When called with a `string`, it returns a `string`. However, this doesn't capture the precise behavior of doubling strings (like 'xx').

x??

---

#### Conditional Types for More Precise Function Definitions

Background context: This concept explains how to use conditional types to define function types more accurately. It highlights the benefits and nuances of using conditional types over overload signatures.

:p How can you improve type inference in a function that accepts both numbers and strings by using a conditional type?
??x
You can use a conditional type to make the return type more precise based on whether the input type is `string` or `number`. Here's an example:

```typescript
declare function double<T extends string | number>(x: T): T extends string ? string : number;
```
In this definition, if `T` (the parameter type) is a subtype of `string`, the return type will be `string`; otherwise, it will be `number`. This approach generalizes to cover all possible cases in a more accurate manner.

x??

---

#### Overload Signatures vs Conditional Types - Practical Example

Background context: This example demonstrates how to handle different types of input parameters using both overload signatures and conditional types. It highlights the trade-offs between simplicity and accuracy.

:p How would you define `double` with overloads for number and string, and why might this approach be less accurate?
??x
You can define `double` with overloads like this:

```typescript
declare function double(x: number): number;
declare function double(x: string): string;
```
This definition works well if you always know the input type (either a `number` or a `string`). However, it becomes less accurate when dealing with unions because TypeScript only tries each overload one by one and stops at the first match. In this case, for an input of type `string | number`, only the last overload would be considered.

x??

---

#### Overload Signatures Limitations

Background context: This concept points out scenarios where overloads might not be suitable due to limitations in handling unions properly. It emphasizes the importance of considering different cases and choosing the right approach.

:p When is it appropriate to use separate functions instead of overloads?
??x
It's appropriate to use separate functions when the union case is implausible or when your function behaves very differently depending on its arguments, leading to distinct functionalities with completely different signatures. For example:

```typescript
function readFileCallback(filename: string): void {
    // Callback-based implementation
}

function readFilePromise(filename: string): Promise<void> {
    // Promise-based implementation
}
```
In such cases, maintaining two separate functions can make the code clearer and easier to understand.

x??

---

#### Conditional Types vs Overloaded Signatures
Conditional types offer a way to conditionally specify return types based on input types, while overloaded signatures are traditional ways of defining functions with different behaviors. Using conditional types can simplify implementation by allowing support for union types without additional overloads.

:p When should you prefer using conditional types over overloaded type signatures?
??x
Conditional types are generally preferred when they make the implementation simpler and more readable. They allow supporting union types without needing to write multiple overload declarations, which can be cumbersome. However, it's still important to ensure that your types are correctly distributed over unions as expected.
x??

---

#### Single Overload Strategy for Conditional Types
For functions declared with conditional types, using a single overload strategy is often clearer and more maintainable compared to multiple overloads.

:p Why would you consider using the single overload strategy for implementing functions declared with conditional types?
??x
Using a single overload strategy can make the implementation cleaner by combining all possible input scenarios into one function. This reduces redundancy and makes the code easier to understand, especially when dealing with union types. It simplifies type assertions and can improve performance since there's only one function to call.
x??

---

#### Distribution of Unions Over Conditional Types
Conditional types distribute over unions in a way that may not always align with the desired behavior.

:p How does TypeScript handle distribution of unions over conditional types?
??x
TypeScript uses distributive properties of conditional types to apply them to each member of a union type. While this can be helpful, it doesn't always produce the intended result, as seen in examples where you may want intersections rather than unions. The `Comparable` example demonstrates how distribution might not align with desired outcomes.
x??

---

#### IsLessThan Function Example
The `isLessThan` function uses a conditional type to handle comparisons between different types (dates, numbers, strings), but the way it distributes over unions can lead to unexpected behavior.

:p What issue does the `Comparable` type in the `isLessThan` function illustrate?
??x
The `Comparable` type incorrectly distributes over unions, leading to an intersection of possibilities instead of a union. This results in incorrect type inference and allows invalid calls that should be errors. For example, passing a string as the second argument when the first is a date or number should not be allowed.
x??

---

#### Correcting Union Distribution
To correct for incorrect union distribution, you might need to rethink how conditional types are applied.

:p How can you modify the `Comparable` type to better handle unions?
??x
You can use intersection types instead of unions to ensure that the second argument is only one of the valid types. For example:
```typescript
type Comparable<T> = T extends Date ? T & number : 
                     T extends number ? T & string :
                     T extends string ? T & Date :
                     never;
```
This ensures that when `a` is a date, `b` can only be both a number and a date. Similarly for other types.
x??

---

#### Preventing Distribution of Unions

Background context: In TypeScript, unions distribute over conditional types when the condition is a bare type. However, sometimes we want to prevent this distribution for specific use cases.

If applicable, add code examples with explanations:

```typescript
type Comparable<T> = 
  [T] extends [Date] ? Date | number : // Check if T is Date
  [T] extends [number] ? number :      // Check if T is number
  [T] extends [string] ? string :      // Check if T is string
  never;
```

:p How can we prevent a union from distributing over a conditional type?
??x
By wrapping the type `T` in a one-element tuple `[T]`, you can prevent unions from distributing. This works because the condition `[T] extends [Type]` is no longer a bare type but an array type, which stops distribution.

```typescript
type Comparable<T> = 
  [T] extends [Date] ? Date | number : // Check if T is Date
  [T] extends [number] ? number :      // Check if T is number
  [T] extends [string] ? string :      // Check if T is string
  never;
```
x??

---

#### Conditional Type Not Distributing

Background context: Sometimes, a conditional type does not distribute over unions due to the structure of the condition.

:p Why might a conditional type not distribute over a union?
??x
A conditional type may not distribute over a union if the condition is not structured as `N extends ...`. In the example provided, `NTuple<T, N>` uses an accumulator but does not distribute because the condition `Acc['length'] extends N` starts with the length of the accumulated array rather than checking if `N` extends any specific type.

```typescript
type NTuple<T, N extends number> = 
  NTupleHelp<T, N, []>;
  
type NTupleHelp<T, N extends number, Acc extends T[]> = 
  Acc['length'] extends N // This does not start with "N extends"
    ? Acc 
    : NTupleHelp<T, N, [T, ...Acc]>;

// Example
type PairOrTriple = NTuple<bigint, 2 | 3>; 
// Incorrectly results in: type PairOrTriple = [bigint, bigint]
```
x??

---

#### Fixing Conditional Type to Distribute

Background context: To fix the issue where a conditional type does not distribute over unions, we need to ensure that the condition starts with `N extends ...`.

:p How can you make sure a conditional type distributes over a union?
??x
To make a conditional type distribute over a union, the condition must start with `N extends ...`. In the provided example, this can be achieved by directly checking if `N` is one of the values in the union.

```typescript
type NTuple<T, N extends number> = 
  NTupleHelp<T, N, []>;
  
type NTupleHelp<T, N extends number, Acc extends T[]> = 
  (N extends 2 | 3 // Start with "N extends"
    ? (Acc['length'] extends N ? Acc : [])
    : []);

// Example
type PairOrTriple = NTuple<bigint, 2 | 3>;
// Correctly results in: type PairOrTriple = [bigint, bigint] | [bigint, bigint, bigint]
```
x??

---

#### Summary of Key Concepts

- **Preventing Distribution**: Use `[T]` to wrap the type and ensure unions do not distribute.
- **Distributing Conditionally**: Ensure the condition starts with `N extends ...` to allow distribution over unions.

#### Conditional Types and Union Distribution
Background context: In TypeScript, conditional types can be distributed over union types. This means that when a type parameter is a union, each member of the union will be processed individually by the conditional type.

:p What happens when a conditional type is applied to a union type?
??x
When a conditional type is applied to a union type, it processes each member of the union separately. This can result in unions of types or in some cases, `never` if no members match.
```typescript
type CelebrateIfTrue<V> = V extends true ? 'Huzzah.' : never;
type SurpriseParty = CelebrateIfTrue<boolean>;
```
x??

---

#### Recursive Generic Types with Accumulators
Background context: Using accumulators in recursive generic types can improve performance by reducing the number of type operations. This technique helps manage complex type transformations.

:p How does adding an accumulator to a recursive generic type help?
??x
Adding an accumulator to a recursive generic type helps avoid unnecessary instantiation and combination of intermediate results, thereby improving performance. The accumulator captures parts of the result as they are computed, reducing the overall complexity.
```typescript
type NTuple<T, N extends number> = 
    N extends number 
    ? NTupleHelp<T, N, []> 
    : never;
type NTupleHelp<T, N extends number, Acc extends any[]> =
  N extends (infer R) & 1
  ? NTupleHelp<T, N - 1, [...Acc, T]>
  : Acc;

type PairOrTriple = NTuple<bigint, 2 | 3>; // [bigint, bigint] | [bigint, bigint, bigint]
```
x??

---

#### Conditional Types and Booleans
Background context: When a conditional type is applied to a boolean union, it distributes over each member of the boolean union. This can lead to unexpected results if not handled carefully.

:p What happens when a conditional type is applied to `boolean`?
??x
When a conditional type is applied to `boolean`, it distributes over both `true` and `false`. As a result, the output includes types from both branches of the conditional type.
```typescript
type CelebrateIfTrue<V> = V extends true ? 'Huzzah.' : never;
type SurpriseParty = CelebrateIfTrue<boolean>;
```
x??

---

#### Conditional Types and the Never Type
Background context: The `never` type in TypeScript is a subtype of every other type. When used in conditional types, it can lead to surprising results due to distribution.

:p What happens when a conditional type involves the `never` type?
??x
When a conditional type involves the `never` type and no member matches, TypeScript will resolve it as `never`. This is because `never` acts as a wildcard that can match any type, but in this context, no branch of the conditional type matches.
```typescript
type AllowIn<T> = T extends { password: "open-sesame" } ? "Yes" : "No";
type N = AllowIn<never>; // never
```
x??

---

