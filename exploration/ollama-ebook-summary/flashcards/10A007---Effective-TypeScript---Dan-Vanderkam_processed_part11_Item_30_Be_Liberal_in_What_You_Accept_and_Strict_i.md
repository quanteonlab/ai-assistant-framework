# Flashcards: 10A007---Effective-TypeScript---Dan-Vanderkam_processed (Part 11)

**Starting Chapter:** Item 30 Be Liberal in What You Accept and Strict in What You Produce

---

#### Robustness Principle (Postel's Law)
Background context: The robustness principle, also known as Postel’s Law, advises developers to be conservative in what they produce and liberal in what they accept. This principle is crucial for creating resilient software systems, especially in network protocols where different implementations can interact with each other.
:p What does the robustness principle suggest?
??x
The robustness principle suggests that when producing data or functionality, one should adhere strictly to standards; however, when accepting inputs or processing data from others, one should be more lenient and forgiving. This approach helps in accommodating variations across different implementations.
x??

---

#### Liberal Acceptance vs Strict Production
Background context: In software development, it is often convenient for functions to accept a broad range of input types (liberal) but produce specific output types (strict). This distinction can affect how robust the codebase is and ease of use.
:p What is the difference between liberal acceptance and strict production in function contracts?
??x
Liberal acceptance means that a function should be able to handle a wide variety of input formats, making it easy for users to call the function. In contrast, strict production implies that a function produces a specific type or set of types as output, ensuring more predictable behavior downstream.
x??

---

#### Camera and Viewport Example
Background context: The example provided uses a 3D mapping API where `setCamera` and `viewportForBounds` functions are defined with liberal input types but produce union types for outputs. This leads to type safety issues when using the resulting camera object.
:p Why did the initial implementation of `focusOnFeature` function fail?
??x
The initial implementation failed because the `viewportForBounds` function returned a union type that included optional properties, making it difficult to safely access specific properties like `lat`, `lng`, and `zoom`. The type inference resulted in properties being typed as `number | undefined`.
x??

---

#### Distinguishing Between Canonical and Loose Types
Background context: To address the issues with liberal return types, the solution involves distinguishing between a canonical form (strict) and loose forms. This allows for more precise typing without sacrificing usability.
:p How can you create a distinction between `LngLat` and `LngLatLike`?
??x
You can create a distinction by defining `LngLat` as an exact type with all required properties, while `LngLatLike` is a union that includes the strict `LngLat` along with other forms. This way, you maintain flexibility in input types but enforce stricter typing on outputs.
```typescript
interface LngLat {
    lng: number;
    lat: number;
};

type LngLatLike = LngLat | { lon: number; lat: number } | [number, number];
```
x??

---

#### Using `Iterable` for Parameter Types
Background context: When a function only needs to iterate over its input, using `Iterable` can be more flexible than using specific array types like `Array`, `ArrayLike`, or `readonly Array`. This allows for greater flexibility in the data source.
:p Why might you choose to use `Iterable<number>` as the parameter type instead of `number[]`?
??x
You might choose to use `Iterable<number>` if your function only needs to iterate over the elements but does not need to modify them. Using `Iterable` makes it compatible with both arrays and generator expressions, providing more flexibility without sacrificing iteration capabilities.
```typescript
function sum(xs: Iterable<number>): number {
    let sum = 0;
    for (const x of xs) {
        sum += x;
    }
    return sum;
}
```
x??

---

#### Input Types vs Output Types
Background context: The text highlights that input types often have broader types compared to output types. Optional properties and union types are more common in parameter types, while void returns can be awkward for clients.

:p What is a key difference between input and output types as described?
??x
Input types tend to be broader than output types because functions may accept various inputs or even optional arguments, whereas outputs are typically constrained to specific return values. For instance, the function `getForegroundColor` in the text can take an optional string argument but returns an object.

```typescript
function getForegroundColor(page?: string): Color {
    // Implementation details
}
```
x??

---

#### JSDoc Annotations and Documentation
Background context: The text suggests using JSDoc annotations for describing function parameters, which helps maintain consistent type information between the code and documentation. It also emphasizes that comments should not contradict the actual implementation.

:p What is a better way to document the `getForegroundColor` function?
??x
A better way to document the `getForegroundColor` function would be through JSDoc annotations:

```typescript
/**
 * Get the foreground color for the application or a specific page.
 */
function getForegroundColor(page?: string): Color {
    // Implementation details
}
```
The comment is concise and does not contradict the implementation.

x??

---

#### Using Iterable<T> vs Array
Background context: The text advises using `Iterable<T>` instead of arrays (`T[]`) if you only need to iterate over a function parameter, as this makes your code more flexible and works well with generators.

:p Why should one use `Iterable<T>` over `Array`?
??x
Using `Iterable<T>` is beneficial because it allows for greater flexibility. It works seamlessly with both arrays and generators, making the function compatible with various iterable structures without changing its logic. For example:

```typescript
function processElements(elements: Iterable<number>) {
    for (const element of elements) {
        // Process each element
    }
}
```

This function can handle any iterable structure, including arrays and generator functions.

x??

---

#### Void Return Types and Mutation Claims
Background context: The text discusses the importance of using types to enforce immutability claims. It explains that type annotations in TypeScript are checked by the compiler, ensuring consistency between code and comments.

:p What is a better way to ensure an array does not get modified?
??x
A better way to ensure an array does not get modified is to use `readonly` instead of making assumptions in comments. For example:

```typescript
function sortNumerically(nums: readonly string[]): string[] {
    return nums.sort((a, b) => Number(a) - Number(b));
}
```

Using `readonly` ensures that the function cannot modify the input array, and if it does attempt to do so, TypeScript will throw an error.

x??

---

#### Type Annotations for Variables
Background context: The text emphasizes the importance of using type annotations directly in code rather than relying on comments. This approach helps maintain consistency and reduces the chance of errors due to out-of-sync documentation.

:p Why is it better to use `age` instead of `ageNum`?
??x
Using a variable name like `age` without the type annotation can lead to potential errors, as TypeScript cannot enforce that `age` should be a number. By naming the variable `age: number`, you ensure that only numbers are assigned to it.

```typescript
const age = 25; // Correct usage with type annotation

// Without type annotation:
const ageNum = "twenty-five"; // This would not trigger an error in TypeScript, but is incorrect logic
```

Using named types like `age: number` ensures that the variable can only hold values of a specific type.

x??

---

#### Avoid Repeating Type Information in Comments and Variable Names
Background context: When working with programming, it's important to avoid redundancy by not repeating type information within comments or variable names. This practice can lead to inconsistencies if the types are updated but the documentation isn't.

Explanation: If a variable is declared as `readonly` or has an associated unit (like `temperatureC`), this information should be reflected in the type declaration, rather than redundantly stated in the comments.
:p How can you avoid redundancy when naming variables and adding comments?
??x
To avoid redundancy, ensure that your variable names include relevant units where necessary. For example, use `timeMs` instead of just `time`, which might imply a different unit such as seconds or minutes.

In comments, avoid repeating the type information if it's already clear from the type declaration.
```ts
// Incorrect: // This is in milliseconds. const timeMs = 1000;
// Correct: const timeMs = 1000; // Time duration of 1 second (1000ms)
```
x??

---

#### Declaring Parameters as Readonly
Background context: In programming, marking a parameter as `readonly` indicates that the value should not be changed after it is initialized. This can help prevent accidental modification and ensure the integrity of data.

Explanation: By declaring parameters as `readonly`, you provide clear documentation to other developers about how they should interact with the function. This also helps in understanding the function's contract without needing external comments.
:p Why would you declare a parameter as `readonly`?
??x
Declaring a parameter as `readonly` is useful because it signals that the value passed into the function will not be modified within the function. This can help prevent bugs and ensure that data integrity is maintained.

For example, in TypeScript:
```ts
function processReadOnlyData(data: readonly string[]) {
  // You cannot modify the array elements here.
}
```
x??

---

#### Avoiding `null` or `undefined` in Type Aliases
Background context: In TypeScript, allowing type aliases to include `null` or `undefined` can lead to confusion and potential bugs. This is because it's generally better practice to handle optional values explicitly rather than implicitly through the type system.

Explanation: If a type alias might allow null or undefined, it should be clearly indicated in the function signature. Using an optional chain (`?.`) ensures that properties are checked for null before accessing them.
:p Why should you avoid including `null` or `undefined` in type aliases?
??x
You should avoid including `null` or `undefined` in type aliases because it can lead to confusion and potential bugs, especially if the alias is used in multiple places. Instead, use a union type that includes both the expected object shape and null/undefined, and utilize optional chaining (`?.`) when accessing properties.

For example:
```ts
type NullableUser = { id: string; name: string } | null;
function getCommentsForUser(comments: readonly Comment[], user: User | null) {
  return comments.filter(comment => comment.userId === (user?.id ?? ''));
}
```
x??

---

#### Handling `null` or `undefined` in Object Types
Background context: When defining object types, allowing properties to be `null` or `undefined` can sometimes make the type definition more complex and harder to understand. However, it's often better practice to explicitly handle such cases.

Explanation: If a property might be null or undefined, consider using an optional field (with a question mark) instead of adding `| null | undefined` to the entire object shape. This makes the structure clearer and avoids unnecessary complexity.
:p How should you handle properties that can be `null` or `undefined` in TypeScript?
??x
You should handle properties that might be `null` or `undefined` by using optional fields (with a question mark) instead of adding `| null | undefined` to the entire object shape. This makes the structure clearer and avoids unnecessary complexity.

For example:
```ts
interface User {
  id: string;
  name?: string; // Optional property
}
```
x??

---

#### Using `null` in Type Aliases vs. Union Types
Background context: In TypeScript, type aliases that include `null` or `undefined` can be confusing and lead to potential bugs. It's generally better practice to use union types with explicit definitions for optional properties.

Explanation: If a type alias might allow null or undefined, it should be clearly indicated in the function signature using a union type. This makes the type system more clear and avoids redundancy.
:p Why is it better to avoid `null` or `undefined` in type aliases?
??x
It's better to avoid `null` or `undefined` in type aliases because it can lead to confusion, especially if the alias is used across multiple places in your codebase. Using a union type with explicit definitions for optional properties makes the structure clearer and avoids unnecessary complexity.

For example:
```ts
// Incorrect: type NullableUser = { id: string; name: string } | null;
type User = {
  id: string;
  name?: string; // Optional property
};
```
x??

---

