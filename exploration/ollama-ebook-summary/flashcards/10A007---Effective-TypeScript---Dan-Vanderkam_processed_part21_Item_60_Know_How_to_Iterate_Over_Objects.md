# Flashcards: 10A007---Effective-TypeScript---Dan-Vanderkam_processed (Part 21)

**Starting Chapter:** Item 60 Know How to Iterate Over Objects

---

#### Template Literal Types for Handling Combinations of Types
Background context: Template literal types are a feature introduced by TypeScript that allows you to create string and number types using template literals. This can be particularly useful when ensuring that certain combinations of types are handled correctly.

:p How do template literal types help in handling the combination of types?
??x
Template literal types allow for more precise type definitions, especially when dealing with specific strings or numbers. For example, you can define a type like `type MyType = `${string}-${string}`` to ensure that only string combinations in a particular format are accepted.

```typescript
type Key = `${string}-${string}`
const myObj: Record<Key, string> = {
  "one-uno": "One",
  "two-dos": "Two"
};
```
x??

---

#### Iteration Over Objects Using for-in Loops
Background context: When using `for...in` loops in TypeScript to iterate over object keys, the variable used for iterating is typically inferred as a string. However, this can lead to issues if the type of the object has no explicit index signature.

:p Why does the iteration over an object with `for...in` result in the variable being inferred as a string?
??x
The inference of `string` for the loop variable in `for...in` loops is due to TypeScript's conservative approach. By default, objects can have properties added dynamically (even from prototypes), so the compiler infers the most general type that covers all possible keys.

```typescript
interface ABC {
    a: string;
    b: string;
    c: number;
}

function foo(abc: ABC) {
  for (const k in abc) { // Inferred as 'string'
    const v = abc[k]; 
  }
}
```
x??

---

#### Type Assertion to Keyof
Background context: Using `keyof` with a type assertion can provide more precise control over the types of loop variables and values. However, it is important to use this judiciously, as it can lead to less safe code if not managed correctly.

:p How does using `keyof` in a type assertion help in iterating over an object?
??x
Using `keyof` with a type assertion allows you to restrict the type of the loop variable to only the keys defined in the object's type. This can prevent TypeScript from inferring a too-wide type, such as `string`, and ensure that the code is more precise.

```typescript
function foo(abc: ABC) {
  for (const kStr in abc) {
    const k = kStr as keyof ABC; // Restrict to 'a' | 'b' | 'c'
    const v = abc[k]; 
  }
}
```
x??

---

#### Object.entries for Iteration Over Key-Value Pairs
Background context: The `Object.entries` method returns an array of a given object's own enumerable string-keyed property [key, value] pairs. This can be used to iterate over both the keys and values of an object in a more type-safe manner.

:p How does `Object.entries` help in iterating over objects?
??x
Using `Object.entries` helps because it provides explicit key-value pairs, making the types of the keys and values known at compile time. This avoids the ambiguity that can arise from using `for...in`, which infers a `string` type for the keys.

```typescript
function foo(abc: ABC) {
  for (const [k, v] of Object.entries(abc)) {
    console.log(v);
  }
}
```
x??

---

#### Iteration Over Maps vs. Objects
Background context: While iterating over objects can lead to issues due to prototype pollution and implicit typing, maps provide a safer way to iterate over key-value pairs because they do not inherit properties from their prototypes.

:p What are the benefits of using `Map` for iteration?
??x
Using `Map` is beneficial because it avoids prototype pollution and ensures that only the explicitly added keys and values are considered. This makes the type safety more straightforward and less error-prone compared to objects.

```typescript
const m = new Map([['one', 'uno'], ['two', 'dos']]);
for (const [k, v] of m.entries()) {
  console.log(v);
}
```
x??

---

#### Type Assertions for Immutable Object Iteration
Background context: When iterating over an immutable object, using a type assertion can help ensure that the keys are precise and not too broad. This is useful when you want to cover all keys in a specific type.

:p How does explicitly listing keys provide better control during iteration?
??x
Explicitly listing keys provides better control by ensuring that only those specified keys are considered in the iteration. This helps maintain type safety and avoids issues related to prototype pollution or unexpected properties.

```typescript
function foo(abc: ABC) {
  const keys = ['a', 'b', 'c'] as const;
  for (const k of keys) {
    const v = abc[k];
  }
}
```
x??

---

#### Record Types to Keep Values in Sync
Background context: This concept revolves around managing state changes in UI components, particularly in scenarios where certain properties should trigger updates while others shouldn't. The `Record` type and `keyof` assertion are used to enforce that all necessary properties are accounted for when checking if a component needs an update.

:p How can you ensure that only relevant properties trigger re-renders in a UI component like a scatter plot?
??x
By using the `Record` type along with `keyof`, we can create an object that maps each property to a boolean value indicating whether it should trigger a re-render. This forces any new properties to be explicitly added, preventing silent bugs.

```typescript
const REQUIRES_UPDATE: Record<keyof ScatterProps, boolean> = {
    xs: true,
    ys: true,
    xRange: true,
    yRange: true,
    color: true,
    onClick: false,
};

function shouldUpdate(
    oldProps: ScatterProps,
    newProps: ScatterProps
): boolean {
    for (const kStr in oldProps) {
        const k = kStr as keyof ScatterProps;
        if (oldProps[k] !== newProps[k] && REQUIRES_UPDATE[k]) {
            return true;
        }
    }
    return false;
}
```

x??

---
#### Conservative vs. Fail Closed Approach
Background context: This concept discusses two strategies for determining when a UI component needs to be re-rendered based on property changes. The conservative approach always ensures correct behavior by checking all properties, while the fail closed approach tries to optimize but risks missing necessary updates.

:p Which approach is more likely to miss necessary re-renders?
??x
The fail closed approach might miss necessary re-renders because it only checks specific properties rather than all of them. While this can reduce unnecessary re-renders, it increases the risk that changes in unmonitored properties won't trigger a re-render.

```typescript
function shouldUpdate(
    oldProps: ScatterProps,
    newProps: ScatterProps
): boolean {
    return (
        oldProps.xs === newProps.xs ||
        oldProps.ys === newProps.ys ||
        oldProps.xRange === newProps.xRange ||
        oldProps.yRange === newProps.yRange ||
        oldProps.color === newProps.color
    );
}
```

x??

---
#### Excess Property Checking with Records and Mapped Types
Background context: This concept highlights how TypeScript’s `Record` type can be used to enforce that an object has exactly the same properties as another, ensuring that all necessary updates are accounted for. This is particularly useful in scenarios where you need to keep track of which properties should trigger re-renders.

:p How does using a `Record` with `keyof` help enforce property consistency?
??x
Using a `Record` with `keyof` ensures that any new or deleted properties in the interface are reflected in the record, thus requiring explicit updates. This is achieved by defining a record where each key from the interface maps to a boolean value indicating whether it should trigger an update.

```typescript
const REQUIRES_UPDATE: Record<keyof ScatterProps, boolean> = {
    xs: true,
    ys: true,
    xRange: true,
    yRange: true,
    color: true,
    onClick: false,
};

// Attempting to add a new property without updating the record will result in an error.
interface ScatterProps {
    // ...
    onDoubleClick?: () => void;
}

const REQUIRES_UPDATE: Record<keyof ScatterProps, boolean> = { // Error
    xs: true,
    ys: true,
    xRange: true,
    yRange: true,
    color: true,
    onClick: false,
    onDoubleClick: false, // Property 'onDoubleClick' is missing in type ...
};
```

x??

---

#### Using `Rest Parameters and Tuple Types` to Model Variadic Functions
Background context explaining the concept. When you need a function that can take a different number of arguments based on some TypeScript type, using rest parameters with tuple types provides a powerful solution. This approach ensures safety by aligning the parameter count with the expected input.

If applicable, add code examples with explanations.
:p How can you design a `buildURL` function in TypeScript to handle optional and varying query parameters for different routes?
??x
To design a `buildURL` function that handles optional and varying query parameters for different routes, you need to use type safety and rest parameters. The key idea is to make the second parameter's type depend on the route.

```typescript
interface RouteQueryParams {
    '/': null,
    '/search': { query: string; language?: string },
    // ... other routes
}

function buildURL<Path extends keyof RouteQueryParams>(
    route: Path, 
    params: RouteQueryParams[Path] = null as any // Provide a type-safe default
) {
    return route + (params ? `?${new URLSearchParams(params)}` : '');
}
```

In this example:
- The function is generic over the path.
- The second parameter's type depends on the specific route (`RouteQueryParams[Path]`).
- A default value of `null as any` ensures type safety.

This approach works well for routes with defined query parameters, but it doesn't handle the case where no parameters are required (like `/`). To address this, you can use rest parameters and conditional types to make the function variadic.

```typescript
function buildURL<Path extends keyof RouteQueryParams>(
    route: Path,
    ...args: (RouteQueryParams[Path] extends null ? [] : [params: RouteQueryParams[Path]])
) {
    const params = args ? args[0] : null;
    return route + (params ? `?${new URLSearchParams(params)}` : '');
}
```

This function:
- Uses rest parameters to handle varying numbers of arguments.
- Checks if the query parameter type is `null` and adjusts accordingly.

Using this approach, you can ensure that the function works correctly for all routes without unnecessary null arguments.
x??

---
#### Conditional Types in Function Signatures
Background context explaining the concept. Conditional types are a powerful feature in TypeScript that allow you to define the type of one element based on another's value or type. This is particularly useful when modeling functions whose signatures depend on some input type.

:p How can you use conditional types to model a function with variable parameter count and type?
??x
To use conditional types to model a function with a variable parameter count and type, you need to ensure that the function signature adapts based on the specific route's query parameters. Here’s how:

```typescript
interface RouteQueryParams {
    '/': null,
    '/search': { query: string; language?: string },
    // ... other routes
}

function buildURL<Path extends keyof RouteQueryParams>(
    route: Path,
    ...args: (RouteQueryParams[Path] extends null ? [] : [params: RouteQueryParams[Path]])
) {
    const params = args ? args[0] : null;
    return route + (params ? `?${new URLSearchParams(params)}` : '');
}
```

In this example:
- The function is generic over the path.
- A rest parameter (`...args`) is used to accept a variable number of arguments.
- A conditional type `(RouteQueryParams[Path] extends null ? [] : [params: RouteQueryParams[Path]])` ensures that if the route does not have query parameters, no additional arguments are required. Otherwise, one argument with the appropriate parameter type is expected.

This approach allows you to write a function that behaves differently based on the route's requirements, providing both safety and flexibility.
x??

---
#### Handling Route Query Parameters Safely
Background context explaining the concept. When building routes for web applications, it’s common to have different query parameters depending on the route. Ensuring type safety while allowing flexible argument handling is crucial.

:p How can you ensure that a `buildURL` function handles optional and varying query parameters safely?
??x
To ensure that a `buildURL` function handles optional and varying query parameters safely, you should use TypeScript’s generic types to align the parameter count with the expected input. Here’s how:

```typescript
interface RouteQueryParams {
    '/': null,
    '/search': { query: string; language?: string },
    // ... other routes
}

function buildURL<Path extends keyof RouteQueryParams>(
    route: Path, 
    params: RouteQueryParams[Path] = null as any // Provide a type-safe default
) {
    return route + (params ? `?${new URLSearchParams(params)}` : '');
}
```

In this example:
- The function is generic over the path (`<Path extends keyof RouteQueryParams>`).
- The second parameter's type depends on the specific route (`RouteQueryParams[Path]`).
- A default value of `null as any` ensures type safety.

For routes that don't require query parameters, passing a null value works correctly. However, this can be extended to handle variadic functions using rest parameters and conditional types:

```typescript
function buildURL<Path extends keyof RouteQueryParams>(
    route: Path,
    ...args: (RouteQueryParams[Path] extends null ? [] : [params: RouteQueryParams[Path]])
) {
    const params = args ? args[0] : null;
    return route + (params ? `?${new URLSearchParams(params)}` : '');
}
```

This approach:
- Uses rest parameters to handle varying numbers of arguments.
- Checks if the query parameter type is `null` and adjusts accordingly.

By using these techniques, you can create a function that handles different routes with their respective query parameters safely and efficiently.
x??

---

