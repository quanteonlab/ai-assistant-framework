# Flashcards: 10A007---Effective-TypeScript---Dan-Vanderkam_processed (Part 16)

**Starting Chapter:** Item 45 Hide Unsafe Type Assertions in Well-Typed Functions

---

#### Prefer More Precise Types Over `any[]` for Rest Parameters
Background context: In TypeScript, rest parameters can be used to accept a variable number of arguments. The type `any[]` allows any JavaScript value as an argument but is less precise compared to more specific types like `unknown[]`. Using more precise types enhances code safety and maintainability.
:p What is the difference between using `any` and `unknown[]` for rest parameters?
??x
Using `any` allows any JavaScript value, which can introduce type safety issues. On the other hand, `unknown[]` is safer because it requires you to explicitly narrow down the types before using them. This means that TypeScript will not allow operations on elements until they are checked, reducing potential runtime errors.
```typescript
const numArgsBad = (...args: any) => args.length; // returns any
const numArgsBetter = (...args: unknown[]) => args.length; // returns number after checking each element's type
```
x??

---

#### Prefer `unknown[]` Over `any[]` for Less Precise Arrays
Background context: When you need an array but don’t care about the specific types of its elements, using `any[]` can be tempting. However, `unknown[]` is generally safer because it requires you to explicitly narrow down the types before operations are performed.
:p Why would one prefer `unknown[]` over `any[]` for arrays?
??x
Using `unknown[]` is preferable over `any[]` because it enforces a level of safety. TypeScript will not allow operations on elements until they are checked, reducing potential runtime errors. This requires more explicit type checking but improves the overall robustness and maintainability of your code.
```typescript
async function fetchPeak(peakId: string): Promise<unknown> {
    return checkedFetchJSON(`/api/mountain-peaks/${peakId}`);
}
```
x??

---

#### Prefer Type Signatures Over Unsafe Implementations in Public APIs
Background context: When designing functions, it is crucial to prioritize the type signature over the implementation. The type signature represents the public API of your function and should reflect what users can expect from it. Safe implementations that avoid type assertions are preferred because they ensure a clean interface for users.
:p Why should you prefer the type signature over an unsafe implementation in public APIs?
??x
You should prefer the type signature because it is visible to users and other parts of the codebase. An unsafe implementation with type assertions or `any` types may pass internal tests but can lead to runtime errors when used elsewhere. A well-defined, safe API ensures that functions work as expected without requiring users to make additional checks.
```typescript
export async function fetchPeak(peakId: string): Promise<MountainPeak> {
    return checkedFetchJSON(`/api/mountain-peaks/${peakId}`);
}
```
x??

---

#### Safe Fetch Wrapper with `unknown` Type
Background context: A wrapper function can add safety to API calls by checking the response and setting a more specific type. In this example, `checkedFetchJSON` returns `unknown`, which is safer than `any`. However, you need to ensure that the return value matches the expected type.
:p What is the problem with using `unknown` in the `fetchPeak` function?
??x
The main issue with using `unknown` in the `fetchPeak` function is that it prevents the TypeScript compiler from inferring the correct type for the returned data. This leads to a type error when trying to use the result directly.
```typescript
async function fetchPeak(peakId: string): Promise<MountainPeak> {
    return checkedFetchJSON(`/api/mountain-peaks/${peakId}`); // Type 'unknown' is not assignable to type 'MountainPeak'.
}
```
x??

---

#### Using `unknown` for Safe API Calls
Background context: When making API calls, it’s important to handle the response safely. Using `unknown` allows you to inspect and cast the data appropriately before using it in your code. This prevents potential runtime errors.
:p How can you use `unknown` effectively in an API call?
??x
You can use `unknown` to safely receive a JSON response from an API and then cast it to the expected type or perform explicit checks. For example, after fetching data, you can inspect its structure before using it in your codebase.
```typescript
export async function fetchPeak(peakId: string): Promise<MountainPeak> {
    const result = await checkedFetchJSON(`/api/mountain-peaks/${peakId}`);
    if (result instanceof MountainPeak) {
        return result;
    } else {
        throw new Error('Unexpected response format');
    }
}
```
x??

---

#### Type Assertions and Assertions in TypeScript

Background context: In TypeScript, type assertions are used to tell the compiler that you expect a certain type for an expression. However, overusing or incorrectly using type assertions can lead to runtime errors. It is better to keep type signatures clean and add type assertions or assertions within function bodies where necessary.

:p What is the benefit of localizing type assertions in functions?
??x
By localizing type assertions, you make your code more maintainable and safer. Type assertions hidden away in the function implementation allow calling code to be written cleanly without any knowledge of potential unsafe secrets. Additionally, it makes it easier to improve safety by adding validations or shape checks within the function.

```typescript
export async function fetchPeak(peakId: string): Promise<MountainPeak> {
    const maybePeak = checkedFetchJSON(`/api/mountain-peaks/${peakId}`);
    if (maybePeak && typeof maybePeak === 'object' && 'firstAscentYear' in maybePeak) {
        throw new Error(`Invalid mountain peak: ${JSON.stringify(maybePeak)}`);
    }
    return checkedFetchJSON(`/api/mountain-peaks/${peakId}`) as Promise<MountainPeak>;
}
```
x??

---
#### Overloading Functions for Type Hiding

Background context: Function overloading can be used to provide different type signatures that callers see versus the implementation. This helps in hiding internal type assertions and ensuring cleaner calling code.

:p How does function overloading help in managing types?
??x
Function overloading allows you to present a specific type signature to the caller while implementing with another signature internally. This way, the external interface remains clean, but the internal logic can handle more complex operations without exposing unsafe type assertions directly to the user.

```typescript
export async function fetchPeak(peakId: string): Promise<MountainPeak>;
export async function fetchPeak(peakId: string): Promise<unknown> {
    return checkedFetchJSON(`/api/mountain-peaks/${peakId}`); // OK
}

const denali = fetchPeak('denali'); // ^? const denali: Promise<MountainPeak>
```
x??

---
#### Handling Type Checking and Assertions in Functions

Background context: Sometimes, the TypeScript type checker cannot infer certain types correctly. To handle this, you can perform type checks or assertions within your function to ensure that data is as expected before using it.

:p What are some methods for handling type checking within functions?
??x
You can use explicit type assertions along with validations inside the function body to ensure that the data matches the expected shape and types. This approach keeps the calling code clean while allowing internal checks to be performed safely.

```typescript
export async function fetchPeak(peakId: string): Promise<MountainPeak> {
    const maybePeak = checkedFetchJSON(`/api/mountain-peaks/${peakId}`);
    if (maybePeak && typeof maybePeak === 'object' && 'firstAscentYear' in maybePeak) {
        throw new Error(`Invalid mountain peak: ${JSON.stringify(maybePeak)}`);
    }
    return checkedFetchJSON(`/api/mountain-peaks/${peakId}`) as Promise<MountainPeak>;
}
```
x??

---
#### Shallow Object Equality Check

Background context: When checking if two objects are shallowly equal, you need to ensure that both have the same keys and corresponding values. However, TypeScript may not infer certain types correctly during this process.

:p How can you perform a shallow object equality check in TypeScript?
??x
To perform a shallow object equality check, you need to validate that all keys present in one object are also present in the other and that their corresponding values match. This ensures that the objects have the same structure and content without deep diving into nested properties.

```typescript
function shallowObjectEqual(a: object, b: object): boolean {
    for (const [k, aVal] of Object.entries(a)) {
        if (!(k in b) || aVal !== b[k]) { // Corrected to use === instead of .==
            return false;
        }
    }
    return Object.keys(a).length === Object.keys(b).length;
}
```
x??

---

#### Use of @ts-expect-error and any Types
Background context: In TypeScript, encountering type errors can sometimes lead to situations where you need to bypass or ignore certain types. This is typically done using `@ts-expect-error` or by using the `any` type. However, using these incorrectly can lead to runtime issues.

:p How should you handle a situation where you expect a type error in TypeScript?
??x
To handle such situations correctly, use `@ts-expect-error` for explicitly ignoring type errors or use the `any` type with proper scoping and explanation within your function implementation. It's crucial not to compromise the function’s signature when fixing internal implementation issues.

Example:
```typescript
// Incorrect approach: Changing the parameter type to any globally
function shallowObjectEqualBad(a: object, b: any): boolean {
  for (const [k, aVal] of Object.entries(a)) {
    if ((k in b) || aVal == b[k]) { // ok
      return false;
    }
  }
  return Object.keys(a).length === Object.keys(b).length;
}

// Correct approach: Using any type narrowly scoped and explaining why it's valid
function shallowObjectEqualGood(a: object, b: object): boolean {
  for (const [k, aVal] of Object.entries(a)) {
    if ((k in b) || aVal == (b as any)[k]) { // `(b as any)[k]` is OK because we've just checked `k in b`
      return false;
    }
  }
  return Object.keys(a).length === Object.keys(b).length;
}
```
x??

---

#### Importance of Correct Function Signatures
Background context: When fixing internal implementation issues, it's important not to compromise the function’s public type signature. Modifying a function's signature can change its behavior and expectations for callers.

:p Why should you avoid changing a function’s type signature when fixing implementation errors?
??x
You should avoid changing a function’s type signature because doing so might affect how other parts of your codebase interact with that function. Changing the public API can lead to unexpected bugs or runtime errors in consumer code, and it's generally best practice to keep internal implementations hidden from external users.

For example, consider a public function `shallowObjectEqual` which should take two objects and return a boolean:
```typescript
function shallowObjectEqual(a: object, b: object): boolean {
  // Implementation details go here
}
```
If you find an error in the implementation but fixing it would change the type of one of the parameters to `any`, you should instead wrap that `any` usage within the function and explain why it is valid.

Example:
```typescript
function shallowObjectEqual(a: object, b: object): boolean {
  for (const [k, aVal] of Object.entries(a)) {
    if ((k in b) || aVal == (b as any)[k]) { // `(b as any)[k]` is OK because we've just checked `k in b`
      return false;
    }
  }
  return Object.keys(a).length === Object.keys(b).length;
}
```
x??

---

#### Thorough Testing and Documentation
Background context: Even when you use type assertions or any types to bypass type errors, it's essential to ensure the correctness of your code. This includes writing comprehensive unit tests and providing clear explanations for any type assertions.

:p Why is thorough testing important when using type assertions in TypeScript?
??x
Thorough testing is crucial because while type assertions can help you write more flexible and dynamic code, they don't guarantee runtime correctness. Tests ensure that the logic inside your function behaves as expected even after making assumptions with type assertions.

Example:
```typescript
// Function under test
function shallowObjectEqual(a: object, b: object): boolean {
  for (const [k, aVal] of Object.entries(a)) {
    if ((k in b) || aVal == (b as any)[k]) { // `(b as any)[k]` is OK because we've just checked `k in b`
      return false;
    }
  }
  return Object.keys(a).length === Object.keys(b).length;
}

// Unit test
describe('shallowObjectEqual', () => {
  it('returns true for identical objects', () => {
    expect(shallowObjectEqual({ x: 1 }, { x: 1 })).toBe(true);
  });

  it('returns false for non-identical objects', () => {
    expect(shallowObjectEqual({ x: 1, y: 2 }, { x: 1, z: 3 })).toBe(false);
  });
});
```
x??

---

#### Explanation of @ts-expect-error
Background context: `@ts-expect-error` is a directive used in TypeScript to tell the compiler that you intentionally want to suppress a type error. This can be useful for code that needs to handle unexpected data types or edge cases.

:p What is the purpose of using `@ts-expect-error`?
??x
The purpose of using `@ts-expect-error` is to inform the TypeScript compiler that you are aware of and intentionally ignoring a specific type error. This can be useful when dealing with code that needs to handle unexpected data types or edge cases where strict typing might not be practical.

Example:
```typescript
function processArray(arr: any[]): void {
  @ts-expect-error // Intentionally bypassing the type check
  for (const item of arr) {
    console.log(item.length); // May cause a runtime error if `arr` contains non-string items
  }
}
```
x??

---

#### Unknown Type vs Any Type
Background context: The `unknown` type is introduced as a safer alternative to using `any` when you have values but don't know their exact types. This helps maintain stricter type safety and enables better error messages from TypeScript.
:p What is the main difference between `unknown` and `any` in TypeScript?
??x
The main difference lies in how they handle type checking:
- `any`: Allows any value to be assigned, making it a superset of all types. This can lead to less safe code but provides more flexibility.
- `unknown`: Enforces that the type is known before using its properties or methods. Any operation on an `unknown` type results in an error if not narrowed down properly.

Code examples:
```typescript
function processValue(value: any) {
  value.read(); // No error, could be anything
}

function safeProcessValue(value: unknown) {
  if (value instanceof Date) { 
    value; // ^? (parameter) value: Date 
  } else {
    throw new Error("Unknown type");
  }
}
```
x??

---

#### Return Type of parseYAML Method
Background context: The `parseYAML` method is designed to return a structured object, but using `any` as the return type would defeat the purpose of TypeScript's static typing. Instead, returning `unknown` allows for better error messages and forces users to narrow down the type.
:p Why should the `parseYAML` method use `unknown` instead of `any`?
??x
Using `unknown` ensures that the resulting value is checked more rigorously by TypeScript:
- TypeScript won't allow operations on an `unknown` value unless it's explicitly narrowed down, leading to more precise type errors.
- This encourages users to provide additional type assertions or narrow the type themselves.

Code example:
```typescript
function safeParseYAML(yaml: string): unknown {
  return parseYAML(yaml);
}

const book = safeParseYAML(`name: Wuthering Heights author: Emily Brontë`);
console.log(book.title); // Error: Property 'title' does not exist on type 'unknown'
```
x??

---

#### Type Narrowing with instanceof and User-Defined Guards
Background context: To work with values of the `unknown` type, you can use a combination of TypeScript's built-in checks like `instanceof` or user-defined guards. These methods help narrow down the unknown type to something more specific.
:p How can you use `instanceof` and user-defined type guards for narrowing down an `unknown` value?
??x
You can use `instanceof` to check if a value is of a certain class, or define a user-defined type guard function that returns a boolean indicating the type.

Code examples:
```typescript
function processValue(value: unknown) {
  if (value instanceof Date) { 
    // value is narrowed down to Date
    value; // ^? (parameter) value: Date
  }
}

// User-defined type guard
function isBook(value: unknown): value is Book {
  return (
    typeof value === 'object' &&
    value !== null &&
    'name' in value && 'author' in value
  );
}

function processValue(value: unknown) {
  if (isBook(value)) { 
    // value is narrowed down to Book
    value; // ^? (parameter) value: Book
  }
}
```
x??

---

#### Type Parameters vs `unknown` for Function Return Types
Background context: While using a type parameter like `<T>` might seem more flexible, it often leads to unnecessary complexity. The use of `unknown` combined with explicit type assertions is generally safer and clearer.
:p Why should you prefer returning `unknown` over using a generic type parameter in the `safeParseYAML` method?
??x
Using `unknown` is preferred because:
- It enforces stricter type safety by preventing operations on unknown values until they are properly narrowed down.
- Users are forced to assert or narrow the type themselves, leading to clearer and more maintainable code.

Code examples:
```typescript
function safeParseYAML(yaml: string): unknown {
  return parseYAML(yaml);
}

const book = safeParseYAML(`name: Wuthering Heights author: Emily Brontë`);
console.log(book.title); // Error: Property 'title' does not exist on type 'unknown'
```
x??

---

#### Using `object` and `{}` Types
Background context: While `unknown` is often the best choice, there are cases where you might use other broad types like `object` or `{}`. These types are slightly narrower than `unknown`, but still allow a wide range of values.
:p How do `object`, `{}`, and `unknown` differ in their usage?
??x
The differences between these types are subtle:
- `unknown`: Enforces that the type is known before using its properties or methods, leading to more precise error messages.
- `{}`: Consists of all values except `null` and `undefined`. Allows primitives like strings and numbers but not objects.
- `object`: Consists of non-primitive types (objects, arrays, functions).

Code examples:
```typescript
declare const foo: Foo;
let barAny = foo as any as Bar; // Using 'any'
let barUnk = foo as unknown as Bar; // Using 'unknown'
```
x??

---

#### Monkey Patching in JavaScript
Monkey patching is a technique where you modify or add properties to objects, prototypes, and functions at runtime. This feature is common in dynamic languages like JavaScript but can lead to issues related to type safety and maintainability.

:p What are the primary reasons why monkey patching should generally be avoided?
??x
Monkey patching can introduce global state and side effects into your application, making it harder to reason about the codebase and leading to potential bugs. Additionally, TypeScript may not recognize custom properties added through monkey patching, causing type errors.
x??

---
#### Issues with Monkey Patching in TypeScript
When using TypeScript, adding properties to built-in objects like `window` or `document` can lead to type errors because TypeScript does not recognize these additions by default.

:p Why do you get type errors when performing operations on patched global variables in TypeScript?
??x
TypeScript's static typing system does not know about the additional properties added through monkey patching. Therefore, it will throw errors if you try to access or assign values to these properties without proper typing annotations.
x??

---
#### Using Augmentations for Monkey Patching
Augmentations allow you to extend the type definitions of existing objects in TypeScript, effectively adding new properties and their types.

:p How can augmentations be used to fix type errors caused by monkey patching?
??x
By using `declare global` along with an interface augmentation, you can inform TypeScript about additional properties that might be added at runtime. This way, the type checker will not throw errors when accessing these properties.
```typescript
declare global {
    interface Window {
        user: User | undefined;
    }
}
```
x??

---
#### Handling Race Conditions in Augmentations
When augmenting a global object like `window` with new properties that might be set after your application starts, you need to handle the possibility of these properties not being defined.

:p How can you address race conditions when using augmentations for monkey patching?
??x
You can define the augmented property with an optional type or explicitly check if it is defined before accessing it. For example:
```typescript
declare global {
    interface Window {
        user: User | undefined;
    }
}

// In a function
if (window.user) {
    alert(`Hello ${window.user.name}.`);
}
```
x??

---
#### Inline Initialization for Global Variables
Another approach to avoid race conditions is by initializing the global variable directly in the HTML, making it available before any JavaScript runs.

:p How can you initialize a global variable via inline script tags?
??x
You can define and initialize the global variable directly within an `<script>` tag in the HTML:
```html
<script type="text/javascript">
    window.user = { name: 'Bill Withers' };
</script>
```
This ensures that the variable is always available, avoiding race conditions.
x??

---
#### Using Narrower Type Assertions
Instead of using `any` for monkey patching, you can define a custom type with the added properties to maintain better type safety.

:p How does using a narrower type assertion help in managing global variables?
??x
Using a custom type like `MyWindow` that extends the original `window` type and includes the patched property allows for more precise typing while avoiding the pitfalls of `any`. This also helps with code organization by not polluting the global scope.
```typescript
type MyWindow = (typeof window) & { user: User | undefined; }
```
x??

---
#### Trade-offs in Augmentation Approaches
While augmentations provide better type safety and documentation, they come with certain trade-offs such as global modifications or handling race conditions.

:p What are the main downsides of using interface augmentations for monkey patching?
??x
The primary downsides include:
- Global scope pollution: The augmented types affect all parts of your code.
- Handling race conditions: You need to ensure that the properties are defined before accessing them, which can be complex in dynamic applications.
- Lack of encapsulation: Augmented types are visible and usable throughout the application, making it harder to manage dependencies.
x??

---

