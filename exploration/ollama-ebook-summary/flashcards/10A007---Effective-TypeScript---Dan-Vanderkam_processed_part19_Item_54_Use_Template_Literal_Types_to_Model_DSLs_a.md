# Flashcards: 10A007---Effective-TypeScript---Dan-Vanderkam_processed (Part 19)

**Starting Chapter:** Item 54 Use Template Literal Types to Model DSLs and Relationships Between Strings

---

#### Conditional Types and Union Distribution

Background context: Conditional types in TypeScript are powerful tools that allow you to define complex type relationships. One of their most significant features is how they distribute over unions, which can significantly impact your generic types.

:p How does distribution work with conditional types over unions?
??x
Distribution works by applying the conditional type to each member of a union and then combining the results using a union. For example:

```typescript
type AllowIn<T> = AllowIn<T | never>;
// This is equivalent to:
type AllowIn<T> = AllowIn<T> | AllowIn<never>;
// Since `never` in TypeScript has no values, it simplifies to:
type AllowIn<T> = AllowIn<T> | never;
// Which further simplifies back to just:
type AllowIn<T> = AllowIn<T>
```

This means that if you have a generic type and want to apply conditional logic to every member of a union, the resulting type will be a union of those results.

```typescript
type F<never> = never; // F<never> must always be `never` when distribution applies.
```

:p How can one-tuples help manage union distribution in conditional types?
??x
Using one-tuples to wrap conditions can prevent unwanted distribution. For example:

```typescript
type AllowIn<T> = T extends string ? 'string' : never;
// Without a tuple, this would distribute over unions:
type AllowInUnion = AllowIn<'foo' | 'bar'>; // Results in "string" | never

// But by wrapping the condition in a one-tuple, we can disable distribution:
type AllowInTuple<T> = [T] extends [string] ? 'string' : never;
AllowInTuple<'foo' | 'bar'>; // Results in 'string'
```

This demonstrates that using tuples can control whether or not your conditional type distributes over unions.

??x
The answer with detailed explanations.
Using one-tuples to wrap conditions can prevent unwanted distribution. For example, consider the following type:

```typescript
type AllowIn<T> = T extends string ? 'string' : never;
```

Without a tuple, this would distribute over unions:

```typescript
type AllowInUnion = AllowIn<'foo' | 'bar'>; // Results in "string" | never
```

However, by wrapping the condition in a one-tuple, you can disable distribution:

```typescript
type AllowInTuple<T> = [T] extends [string] ? 'string' : never;
AllowInTuple<'foo' | 'bar'>; // Results in 'string'
```

This demonstrates that using tuples can control whether or not your conditional type distributes over unions, giving you more fine-grained control over how your types behave.

---


#### Template Literal Types

Background context: Template literal types allow you to model a subset of strings based on patterns. They are particularly useful when you want to enforce specific string structures in your code without resorting to the generic `string` type.

:p How do template literal types work?
??x
Template literal types let you define a set of strings that match a certain pattern. For example, consider the following:

```typescript
type PseudoString = `pseudo${string}`;
const science: PseudoString = 'pseudoscience';  // ok
const alias: PseudoString = 'pseudonym';       // ok
const physics: PseudoString = 'physics';       //    ~~~~~~~ Type '\"physics\"' is not assignable to type '`pseudo${string}`'.
```

Here, `PseudoString` represents a set of strings that start with the prefix "pseudo". This ensures that only values matching this pattern are valid.

:p What are some practical uses for template literal types?
??x
Template literal types can be used in various scenarios where you want to enforce specific string patterns. For example:

- **DSLs (Domain Specific Languages):** Define a set of allowed strings for a particular domain.
- **Prefix Patterns:** Ensure that strings match certain prefixes or suffixes, such as "pseudo" or "data-".
- **Configuration Keys:** Enforce consistent naming conventions in configuration files.

:p Can you provide an example using template literal types with a prefix pattern?
??x
Certainly! Here’s an example of using template literal types to enforce a specific prefix pattern:

```typescript
type PseudoString = `pseudo${string}`;
const science: PseudoString = 'pseudoscience';  // ok
const alias: PseudoString = 'pseudonym';       // ok
const physics: PseudoString = 'physics';       //    ~~~~~~~ Type '\"physics\"' is not assignable to type '`pseudo${string}`'.
```

In this example, `PseudoString` ensures that only strings starting with "pseudo" are valid.

??x
The answer with detailed explanations.
Template literal types allow you to model a subset of strings based on patterns. For example, in the given code:

```typescript
type PseudoString = `pseudo${string}`;
const science: PseudoString = 'pseudoscience';  // ok
const alias: PseudoString = 'pseudonym';       // ok
const physics: PseudoString = 'physics';       //    ~~~~~~~ Type '\"physics\"' is not assignable to type '`pseudo${string}`'.
```

`PseudoString` represents a set of strings that start with the prefix "pseudo". This ensures that only values matching this pattern are valid, providing a way to enforce specific string structures in your code.

---

#### Template Literal Types for CSS Selectors
Background context: This concept explains how template literal types can be used to create more precise type definitions for DOM queries, specifically for matching CSS selectors. It highlights the limitation of TypeScript's current implementation and how it can be improved with custom type definitions.

:p How can template literal types be used to enhance the precision of element queries in TypeScript?
??x
Template literal types allow us to create more specific type definitions that match the structure of CSS selectors, thus providing better type safety when querying elements from the DOM. By using these types, we can ensure that our queries are as precise as possible.

For example, consider the scenario where you want to query an `<img>` element with a specific ID:

```typescript
type HTMLTag = keyof HTMLElementTagNameMap;
declare global {
    interface ParentNode {
        querySelector<TagName extends HTMLTag>(
            selector: `${TagName}#${string}`
        ): HTMLElementTagNameMap[TagName] | null;
    }
}
```

With this custom type, querying for an `<img>` element with a specific ID will return the more precise `HTMLImageElement` type:

```typescript
const img = document.querySelector('img#spectacular-sunset') as HTMLImageElement | null;
```

This ensures that `img.src` can be accessed without TypeScript complaining about potential undefined properties.

??x

---
#### Handling Element Queries with Specific IDs
Background context: This concept explains how template literal types and type inference can be used to handle element queries based on specific tag names and IDs. It demonstrates the use of template literal types in combination with generic types to achieve more precise typing for DOM elements.

:p How does using a template literal type improve the precision of querying an element by its ID?
??x
Using a template literal type improves the precision of querying an element by its ID because it allows TypeScript to infer the exact type of the queried element based on the tag name. For example, when querying for an `<img>` element with a specific ID:

```typescript
type HTMLTag = keyof HTMLElementTagNameMap;
declare global {
    interface ParentNode {
        querySelector<TagName extends HTMLTag>(
            selector: `${TagName}#${string}`
        ): HTMLElementTagNameMap[TagName] | null;
    }
}
```

This custom type ensures that `document.querySelector('img#spectacular-sunset')` returns an `HTMLImageElement` or `null`, which means you can safely access properties like `src` on the result.

```typescript
const img = document.querySelector('img#spectacular-sunset') as HTMLImageElement | null;
img?.src; // This is now allowed and safe to use.
```

This approach prevents type-related errors by ensuring that only elements of the expected type are returned from queries, making your code more robust.

??x

---
#### Handling Queries with Descendant Selectors
Background context: This concept explains how template literal types can be used to handle CSS descendant selectors in DOM queries. It demonstrates a limitation where spaces in CSS selectors (indicating descendant relationships) cause the query to match the first tag name and ignore subsequent parts of the selector.

:p How does using template literal types affect the handling of descendant selectors in DOM queries?
??x
Using template literal types can lead to unexpected results when dealing with descendant selectors, as they are interpreted literally by TypeScript. For example:

```typescript
const img = document.querySelector('div#container img') as HTMLDivElement | null;
```

In this case, `document.querySelector('div#container img')` returns an `HTMLDivElement` or `null`, because the template literal type matches `"div" # "container img"` separately. The space in the selector is interpreted as a simple string delimiter rather than indicating a descendant relationship.

This means that you cannot directly use template literal types to handle complex CSS selectors like descendants, and you may need additional logic or custom types to achieve more precise typing for such cases.

??x

---

#### Template Literal Types and CSS Selector Parsing
Background context explaining how template literal types can be used to create precise and accurate type definitions for complex string patterns, such as CSS selectors. This is particularly useful when working with the Document Object Model (DOM) in TypeScript.

```typescript
type CSSSpecialChars = ' ' | '>' | '+' | '~' | '|' | ',';
type HTMLTag = keyof HTMLElementTagNameMap;

declare global {
    interface ParentNode {
        // Escape hatch for complex selectors
        querySelector(
            selector: `${HTMLTag}#${string}${CSSSpecialChars}${string}`
        ): Element | null;

        // Standard method for simple selectors
        querySelector<TagName extends HTMLTag>(
            selector: `${TagName}#${string}`
        ): HTMLElementTagNameMap[TagName] | null;
    }
}
```

:p How can template literal types help in creating precise and accurate type definitions for CSS selectors?
??x
Template literal types allow you to model structured subsets of string types, making it possible to define the exact pattern required for a CSS selector. By combining these with other TypeScript features like conditional types and mapped types, you can achieve highly specific and accurate type annotations.

For example:
- `${HTMLTag}#${string}${CSSSpecialChars}${string}` is used to represent a more complex selector structure.
- This approach ensures that the type system accurately reflects the possible structure of selectors, preventing errors where invalid or unexpected characters are used in CSS queries.

```typescript
const img = document.querySelector('img#spectacular-sunset'); //    ^? const img: HTMLImageElement | null
const img2 = document.querySelector('div#container img'); //    ^? const img2: Element | null
```

This ensures safe usage and reduces the likelihood of runtime errors.
x??

---

#### Recursive Type Transformation with `ToCamel`
Background context explaining how to use recursive types in TypeScript to transform snake_case strings into camelCase, which is useful for converting key names from one format to another.

:p How can you create a recursive type that transforms snake_case strings into camelCase?
??x
You can use a combination of conditional types and the `infer` keyword to create a recursive type transformation. This approach involves breaking down the string at each underscore, capitalizing the first letter after an underscore, and combining them back together.

```typescript
type ToCamelOnce<S extends string> = S extends `${infer Head}_${infer Tail}`
    ? `${Head}${Capitalize<Tail>}`
    : S;

type T = ToCamelOnce<'foo_bar'>;  // type is "fooBar"
```

For more complex strings with multiple underscores:
```typescript
type ToCamel<S extends string> = S extends `${infer Head}_${infer Tail}`
    ? `${Head}${Capitalize<ToCamel<Tail>>}`
    : S;

type T0 = ToCamel<'foo'>;  // type is "foo"
type T1 = ToCamel<'foo_bar'>;  // type is "fooBar"
type T2 = ToCamel<'foo_bar_baz'>;  // type is "fooBarBaz"
```

:p How does the `ToCamel` recursive type work to transform strings?
??x
The `ToCamel` type uses a conditional check to split the string at each underscore. If it finds an underscore, it extracts the part before and after the underscore using the `infer` keyword. It then combines these parts into a new string where the first letter after the underscore is capitalized.

This process is recursive, meaning that if there are multiple underscores in the input string, it will continue to break down the string until all parts have been transformed.

Here's an example breakdown:
```typescript
type T2 = ToCamel<'foo_bar_baz'>;
```

1. The type `T2` checks for an underscore and finds one.
2. It extracts "foo" as `Head` and "bar_baz" as `Tail`.
3. It then calls `ToCamel<Tail>` on "bar_baz".
4. The result of the recursive call is `"barBaz"`, so combining it with "foo" gives us "fooBarBaz".

This ensures that all parts of a snake_case string are transformed into camelCase.
x??

---

#### Mapped Types for Object Key Transformation
Background context explaining how mapped types can be used to transform object keys from one format (e.g., snake_case) to another (e.g., camelCase), ensuring precise and accurate type definitions.

:p How can you use TypeScript's mapped types to transform the keys of an object from snake_case to camelCase?
??x
You can use a mapped type in combination with a helper function to transform the keys of an object. The `ToCamel` type is used as a key transformation function within the mapped type.

Here’s how it works:
```typescript
type ObjectToCamel<T extends object> = {
    [K in keyof T as ToCamel<K & string>]: T[K];
};

function objectToCamel<T extends object>(obj: T): ObjectToCamel<T> {
    const out: any = {};
    for (const [k, v] of Object.entries(obj)) {
        out[camelCase(k)] = v;
    }
    return out;
}
```

This mapped type rewrites the keys using the `ToCamel` helper function and assigns the corresponding values.

:p How does the `ObjectToCamel<T>` mapped type work to transform object keys?
??x
The `ObjectToCamel<T>` mapped type uses a mapped type with a key remapping syntax. For each key in the input object, it applies the `ToCamel` helper function to transform the key and maps it to the corresponding value.

Here’s how it works step-by-step:
1. The mapped type iterates over all keys of the input object.
2. It uses the `as` keyword to apply the `ToCamel` transformation to each key.
3. It creates a new property with the transformed key and assigns the original value to it.

For example, if you have an object `{foo_bar: 12}`:
```typescript
const snake = {foo_bar: 12}; //    ^? const snake: { foo_bar: number; }
const camel = objectToCamel(snake); //    ^? const camel: ObjectToCamel<{ foo_bar: number; }> //                    (equivalent to { fooBar: number; })
```

The `camel` object will have the key `fooBar` with a type of `number`, while accessing `foo_bar` would result in an error because it does not exist on the transformed type.

This ensures that your code remains type-safe and provides accurate typing for camelCase keys.
x??

---

#### Conditional Types and Mapped Types
Background context explaining how to use conditional types and mapped types together to achieve precise and accurate type definitions, particularly useful in complex scenarios like transforming key names or parsing domain-specific languages (DSLs).

:p How can you combine conditional types and mapped types to create a more precise and accurate type definition for an object's keys?
??x
You can combine conditional types and mapped types to transform the keys of an object from one format to another, ensuring that the resulting type is highly specific and accurate. This involves using a helper function like `ToCamel` to perform the key transformation.

Here’s how you can do it:

1. Define a helper type for transforming snake_case to camelCase.
2. Use this helper in a mapped type to rewrite object keys.

Example:
```typescript
type ToCamelOnce<S extends string> = S extends `${infer Head}_${infer Tail}`
    ? `${Head}${Capitalize<Tail>}`
    : S;

type ToCamel<S extends string> = S extends `${infer Head}_${infer Tail}`
    ? `${Head}${Capitalize<ToCamel<Tail>>}`
    : S;

function objectToCamel<T extends object>(obj: T): { [K in keyof T as ToCamel<K & string>]: T[K] } {
    const out: any = {};
    for (const [k, v] of Object.entries(obj)) {
        out[camelCase(k)] = v;
    }
    return out;
}
```

This combination ensures that your object keys are transformed accurately while maintaining precise types.

:p How do conditional types and mapped types work together to transform object keys?
??x
Conditional types allow you to create complex type transformations based on conditions, making it possible to split strings at specific delimiters (like underscores) and apply transformations. Mapped types enable you to iterate over the properties of an object and perform these transformations.

Here’s a breakdown:
1. **ToCamelOnce**: This is a simple transformation that handles one level of snake_case to camelCase conversion.
2. **ToCamel**: This uses recursion to handle multiple levels of snake_case keys in complex strings.
3. **ObjectToCamel**: Uses the mapped type syntax with `as` and a helper function like `camelCase` to transform the keys.

By combining these, you can ensure that your object’s keys are transformed accurately while maintaining precise types, providing a more robust type system for your application.

For example:
```typescript
const snake = {foo_bar: 12}; //    ^? const snake: { foo_bar: number; }
const camel = objectToCamel(snake); //    ^? const camel: ObjectToCamel<{ foo_bar: number; }> //                    (equivalent to { fooBar: number; })
```

This ensures that `camel.fooBar` has the correct type, and accessing `foo_bar` would result in an error.
x??

---

#### Template Literal Types Overview
Template literal types allow you to create types from string literals, which can be used to capture nuanced relationships between types. They are particularly useful when working with configuration or settings that should match specific strings.

:p What is a template literal type and how does it help in capturing nuanced relationships?
??x
A template literal type allows the creation of new types based on string literals. It helps ensure that certain properties or parameters must be one of several predefined strings, which can improve the developer experience by reducing errors related to typos or incorrect values.

```ts
type Greeting = `Hello ${string}`;
// Usage: const greeting: Greeting = 'Hello World'; // Valid
const greeting: Greeting = 'Hi there!'; // Error, must start with "Hello"
```
x??

---

#### Mapped and Conditional Types for Complex Type Logic
Mapped types allow you to transform one type into another by applying operations to each property. Combined with conditional types, they enable the creation of complex type systems that reflect nuanced relationships.

:p How do mapped and conditional types work together in TypeScript?
??x
Mapped types iterate over all properties of a type, allowing you to add or modify them based on certain conditions. Conditional types are used to create new types by checking if existing types meet specific criteria. Together, they enable sophisticated type transformations.

```ts
type MappedExample<T> = {
  [K in keyof T]: string;
};

// Usage: 
type Props = { name: string; age: number };
type StringifiedProps = MappedExample<Props>; // { name: string; age: string }

type ConditionalExample<T extends boolean> = T ? "true" : "false";

// Usage:
const value: ConditionalExample<true> = "true"; // Valid
const value2: ConditionalExample<false> = "false"; // Valid
```
x??

---

#### Testing Types in TypeScript
Testing types involves ensuring that your type declarations match the intended logic and behavior. This is particularly important in TypeScript because it allows for complex logic within types, making potential bugs more likely.

:p How can you test types in TypeScript?
??x
You can test types using the type system or tooling outside of the type system. Common techniques include writing tests that check the types directly by assigning values to variables with declared types and asserting their correctness.

```ts
const lengths: number[] = map(['john', 'paul'], name => name.length);
// This checks if `map` returns an array of numbers, as expected.
```

You can also use runtime assertions or unit tests that check the actual behavior:

```ts
test('map function', () => {
  const result = map(['2017', '2018'], v => Number(v));
  expect(result).toEqual([2017, 2018]);
});
```
x??

---

#### Ineffective Ways to Test Types
Testing types can sometimes be done in ways that are not fully effective. For example, just running the function with some inputs and checking if it doesn’t throw an error is insufficient.

:p What is a common but ineffective way to test type declarations?
??x
A common but ineffective way to test type declarations is by simply calling the function and ensuring it does not throw an error. This approach does not verify that the return types or other type-related aspects are correct, as shown in this example:

```ts
test('square a number', () => {
  square(1);
  square(2);
});
```

This test only ensures that `square` doesn’t crash but fails to check if it returns the expected values.

The better approach is to explicitly test the types by assigning values with declared types and asserting their correctness.
x??

---

#### Writing Tests for Type Declarations
Writing comprehensive tests for type declarations involves ensuring both that your type logic works as intended and that the type system correctly enforces the rules you have defined.

:p How should you write tests for a type declaration?
??x
To test type declarations, you should explicitly check that values conform to the declared types. This can be done by assigning known valid and invalid values to variables with specified types and asserting their correctness.

```ts
const lengths: number[] = map(['john', 'paul'], name => name.length);
// Ensure `lengths` is an array of numbers

test('map function', () => {
  const result = map(['2017', '2018'], v => Number(v));
  expect(result).toEqual([2017, 2018]);
});
```

This approach ensures that the type logic is correct and robust.
x??

---

#### Assignability Testing with TypeScript

Background context: When working with TypeScript, you might use assignment for testing type declarations. This method provides some confidence that your type declaration is doing something sensible with types. However, there are limitations and issues associated with this approach.

:p What is a primary issue when using assignment for testing in TypeScript?
??x
A primary issue is that the check uses assignability rather than equality, which can lead to unexpected behavior. For instance, an array of objects might be assignable to an array of simpler objects, even if they don't match exactly.
x??

---
#### Boilerplate and Unused Variable Issues

Background context: To perform assignment-based type checks in TypeScript, you often need to define helper functions, which introduces boilerplate code. Additionally, these helper variables are likely to be unused, leading to linting warnings.

:p How can you work around the issue of unused variables when performing type checks?
??x
You can create a helper function that takes a value and asserts its type without actually using it:

```typescript
function assertType<T>(x: T) {}
```

Then use this helper in your type check:

```ts
assertType<number[]>(map(['john', 'paul'], name => name.length));
```
x??

---
#### Function Type Assignability

Background context: When checking function types with assignment, TypeScript allows functions to be more general than the ones they are assigned to. This means a function that takes fewer parameters can still match a function type that expects more parameters.

:p Why does the following assertion succeed in TypeScript?
??x
```ts
const double = (x: number) => 2 * x;
assertType<(a: number, b: number) => number>(double);  // OK
```

This is because TypeScript's assignability rules allow a function that takes fewer parameters to be considered assignable to a function type that expects more. This behavior reflects the common JavaScript practice where functions can be called with more arguments than declared.
x??

---
#### Object Type Assignability

Background context: When checking object types, assignment-based checks might not reflect true equality but instead check if one type is assignable to another. For example, an array of objects with additional properties might still pass a type check for an array of simpler objects.

:p What issue arises when checking the following type with assignability?
??x
```ts
assertType<{name: string}[]>(map(beatles, name => ({      // OK
  name,
  inYellowSubmarine: name === 'ringo'
})));
```

The map function returns an array of objects with more properties than `{name: string}`, so the assignability check passes. However, this doesn't ensure that all expected properties are present.
x??

---
#### Complex Function Types and `this` Context

Background context: Assignability checks for functions can sometimes lead to unexpected behavior, especially when dealing with complex function types or dynamic contexts like `this`. TypeScript allows functions to be more general than the ones they are assigned to.

:p How does TypeScript handle this issue in Lodash's map function?
??x
Lodash’s map sets the value of `this` for its callback. While TypeScript can model this behavior, it might not always capture all the nuances required by dynamic contexts like `this`.

To properly test such complex functions, you may need to break down the type and check individual components or use advanced type utilities.
x??

---
#### Using Parameters and ReturnType

Background context: When dealing with function types, breaking them down into their component parts using `Parameters` and `ReturnType` can provide a more precise type check.

:p How can you use `Parameters` and `ReturnType` to test the types of functions?
??x
You can use these utility types to check the parameters and return type of a function:

```ts
const double = (x: number) => 2 * x;
declare let p: Parameters<typeof double>;
assertType<[number]>(p); // Argument of type '[number]' is not assignable to parameter of type [number, number]

declare let r: ReturnType<typeof double>;
assertType<number>(r);
```

This approach helps ensure that the function's parameters and return types are correct.
x??

---

#### Testing Map Functionality
Background context explaining that testing a function like `map` requires not only checking the final output but also understanding the intermediate steps. This involves verifying types and values within the callback function used by `map`. 
:p How can we test the details of the intermediate steps when using the `map` function?
??x
To test the details of the intermediate steps, you need to fill out the callback function with assertions for its parameters. For example, in TypeScript, you might use assertType functions to check the types of each parameter passed to the callback.

```typescript
const beatles = ['john', 'paul', 'george', 'ringo'];
assertType<number[]>(map(
    beatles,
    function(name, i, array) {
        assertType<string>(name);
        assertType<number>(i);
        assertType<string[]>(array);
        return name.length;
    }
));
```
In this example, `assertType` is used to ensure that the parameters are of the expected types. This helps in verifying if the callback function is being called correctly with the right arguments.

x??

---

#### Declaration Fixes for Map Function
Background context explaining that initial declarations for functions like `map` might be incomplete and need adjustments to pass stricter type checks.
:p What issues were found with the earlier declaration of `map`, and how did you address them?
??x
The original declaration of `map` had two main issues: it accepted only one parameter in the callback function, and it didn't specify a type for `this`. To fix these, we modified the declaration to include all necessary parameters.

```typescript
declare function map<U, V>(
    array: U[],
    fn: (this: U[], u: U, i: number, array: U[]) => V
): V[];
```

This updated declaration now includes `i` and `array` as parameters in the callback function, making it more accurate.

x??

---

#### Negative Tests with TypeScript
Background context explaining that while strict type checking is important, sometimes there are cases where you might want to introduce intentionally erroneous code to test your understanding.
:p How can negative tests be used to improve type safety when using `map`?
??x
Negative tests in TypeScript involve creating scenarios where the compiler expects an error but does not get one. This can help ensure that certain types of errors or issues are caught. You can use the `@ts-expect-error` comment to achieve this.

```typescript
// @ts-expect-error only takes two parameters
map([1, 2, 3], x => x * x, 'third parameter');
```

In this example, adding a third parameter where none should be expected results in an error. If the code passes without an error, it indicates that the type system is not catching all potential issues.

x??

---

#### Handling Any Types
Background context explaining the issue of `any` types being assigned to modules and how they can undermine type safety.
:p What are the risks of using `any` types in module declarations and how do you mitigate them?
??x
Using `any` types in module declarations can lead to a loss of type safety, as any function call within the module will produce an `any` type. This can spread throughout your codebase, reducing its overall type safety.

To mitigate this, you should avoid assigning `any` types to modules and ensure that all functions have proper type signatures. You can use negative tests with `@ts-expect-error` comments to catch such issues:

```typescript
// @ts-expect-error only takes two parameters
map([1, 2, 3], x => x * x, 'third parameter');
```

By doing so, you ensure that the type checker will flag these as errors, maintaining stricter type safety.

x??

---

#### Detecting Any Types with Type Aliases
Background context explaining how to detect `any` types using type aliases.
:p How can you use a type alias to detect when a function parameter or return type is `any`?
??x
You can create a custom type guard that checks if the type is `any`. For example, you might define a utility type like this:

```typescript
type NeverAny<T> = T extends any ? never : T;
```

This type alias will only match types that are not `any`, making it useful for detecting unexpected `any` types in your code.

:p Can you provide an example of using the `NeverAny` type alias to check function parameters?
??x
Yes, you can use the `NeverAny` type alias to ensure that a function parameter is not of type `any`. Here’s an example:

```typescript
function processElement<T>(element: NeverAny<T>) {
    // The element cannot be 'any' at this point.
}

processElement(42);  // OK
processElement('hello');  // OK
processElement<any>('unexpected any');  // Error: Argument of type 'string' is not assignable to parameter of type 'NeverAny<never>'.
```

In this example, `processElement` ensures that its parameter is not of type `any`, making it easier to catch and correct such issues.

x??

---

#### expect-type Library Overview
Background context: The `expect-type` library is a powerful tool for testing TypeScript types. It integrates well with existing TypeScript test suites and helps catch type mismatches without requiring additional setup. This library leverages TypeScript's type system to ensure that functions, interfaces, and other constructs match the expected types.
:p What is `expect-type` used for?
??x
The `expect-type` library is used to verify that code adheres strictly to its defined types. It ensures that function parameters, return values, and object properties are of the correct type by utilizing TypeScript's structural type system. This helps in maintaining type safety and catching potential bugs early.
```typescript
import { expectTypeOf } from 'expect-type';

const beatles = ['john', 'paul', 'george', 'ringo'];
expectTypeOf(map(
    beatles,
    function(name, i, array) {
        expectTypeOf(name).toEqualTypeOf<string>();
        expectTypeOf(i).toEqualTypeOf<number>();
        expectTypeOf(array).toEqualTypeOf<string[]>();
        return name.length;
    }
)).toEqualTypeOf<number[]>();
```
x??

---

#### Type Mismatch Detection
Background context: The `expect-type` library is adept at catching type mismatches. When used correctly, it can pinpoint exactly where and what the types are not aligning as expected.
:p What happens when a type mismatch occurs in `expect-type`?
??x
When a type mismatch occurs in `expect-type`, the compiler will generate an error indicating that the expected type does not match the actual type. This helps developers to quickly identify and correct any type-related issues in their code.

For example, consider the following incorrect usage:
```typescript
const anyVal: any = 1;
expectTypeOf(anyVal).toEqualTypeOf<number>();
//                                 ~~~~~~
//           Type 'number' does not satisfy the constraint 'never'.
```
The error message `Type 'number' does not satisfy the constraint 'never'.` indicates that TypeScript cannot satisfy the type constraint because it expects a value of type `any`, but provides a number.
x??

---

#### Function Signature Testing
Background context: The `expect-type` library allows for detailed testing of function signatures, ensuring that the provided arguments and return values match the expected types.
:p How can you test function signatures using `expect-type`?
??x
Testing function signatures with `expect-type` involves verifying both the parameters and the return type. You can use the `toEqualTypeOf` method to ensure that each argument and the overall function signature matches the expected types.

Example:
```typescript
const double = (x: number) => 2 * x;
expectTypeOf(double).toEqualTypeOf<(a: number, b: number) => number>();
//                                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//           Type ... does not satisfy '\"Expected: function, Actual: never\"'
```
This test checks that the `double` function has a return type of `(a: number, b: number) => number`, but since it only takes one parameter and returns a single value, it fails to match the expected signature.
x??

---

#### Interface Testing with `expect-type`
Background context: The `expect-type` library can also be used to test interfaces by ensuring that object properties adhere to the defined structure. This is particularly useful for catching errors related to property types or presence/absence of properties.
:p How do you use `expect-type` to test interface types?
??x
Using `expect-type` to test interface types involves comparing an object's actual type with the expected interface type using `toEqualTypeOf`. You can check if all required properties are present and have the correct types.

Example:
```typescript
interface ABReadOnly {
    readonly a: string;
    b: number;
}

declare let ab: {a: string, b: number};
expectTypeOf(ab).toEqualTypeOf<ABReadOnly>();
//               ~~~~~~~~~~~~~
//           Arguments for the rest parameter 'MISMATCH' were not provided.
```
This test verifies that `ab` matches the `ABReadOnly` interface. Since `ab` does not have a `readonly` modifier and all properties are present, it passes.
x??

---

#### Structural Type Testing
Background context: One of the key advantages of using `expect-type` is its ability to perform structural type testing, which means that types are compared based on their structure rather than their exact identity. This allows for more flexible type checking without getting confused by superficial differences like order in unions or tuples.
:p What advantage does structural type testing offer over nominal typing?
??x
Structural type testing offers an advantage over nominal typing because it checks whether the actual structure of a type matches the expected structure, rather than just verifying if the types are identical. This is particularly useful when dealing with TypeScript's union and intersection types.

For example:
```typescript
type Test1 = Expect<Equals<typeof double, (x: number) => number>>;
type Test2 = Expect<Equals<typeof double, (x: string) => number>>;
//                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  Type 'false' does not satisfy the constraint 'true'.
```
In this case, `Test1` correctly identifies that the function matches the expected signature, while `Test2` fails because the parameter type does not match.
x??

#### Function Types and Covariance
Background context: The text discusses function types and their relationship to covariance, particularly how return types of functions are covariant. It also touches on type equality checks within TypeScript and testing types using external tools like dtslint or eslint-plugin-expect-type.

:p What is the significance of function types being covariant with respect to their return types?
??x
Function types in TypeScript are covariant, meaning that if a function returns one type, it can also return a subtype of that type. However, the text highlights an exception where only when `X` equals `Y`, the first conditional type could reliably be assignable to the second.

For example:
```typescript
type Test6 = Expect<Equals<(...args: any[]) => 1 | 2, (...args: any[]) => 2 | 1>>; // This is acceptable due to covariance.
```

x??

---
#### Type Equality in TypeScript
Background context: The text explains how type equality can be checked using `Expect` types in TypeScript. It notes that while this approach provides robustness similar to `expect-type`, the error messages are not particularly illuminating and the concept of type equality is rarely used, leading to unclear semantics.

:p What happens when testing for type equality using `Equals` in TypeScript?
??x
Testing for type equality with `Equals` can produce unexpected results. For instance:
```typescript
type Test3 = Expect<Equals<1 | 2, 2 | 1>>; // Passes as expected.
type Test5 = Expect<Equals<{x: 1} & {y: 2}, {x: 1, y: 2}>>; // Fails with a surprising error message.
```

The failure in `Test5` might be due to the way TypeScript displays type intersections.

x??

---
#### Using dtslint for Type Testing
Background context: The text introduces `dtslint`, an external tool that tests type declarations in DefinitelyTyped repository. It operates through specially formatted comments and inspects types by doing a textual comparison, matching how you'd manually test types in your editor.

:p How does `dtslint` work to test TypeScript types?
??x
`dtslint` works by inspecting the type of each symbol using `$ExpectType` comments. For example:
```typescript
const beatles = ['john', 'paul', 'george', 'ringo'];
map(beatles, function(
    name,  // $ExpectType string
    i,     // $ExpectType number
    array // $ExpectType string[]
) {
    this   // $ExpectType string[]
    return name.length;
});  // $ExpectType number[]
```

This approach is useful but can be sensitive to type display differences.

x??

---
#### Using eslint-plugin-expect-type for Type Testing
Background context: `eslint-plugin-expect-type` is another external tool that works as an ESLint plugin. It checks types using both `$ExpectType` and `///` (TSLINT) comments, making it convenient for testing your own TypeScript code.

:p How does `eslint-plugin-expect-type` differ from `dtslint` in terms of usage?
??x
While both tools use similar mechanisms with `$ExpectType` and `///` (TSLINT) comments to check types, `eslint-plugin-expect-type` integrates better into your development environment. It also offers an auto-fixer feature for easier updates.

For example:
```typescript
const spiceGirls = ['scary', 'sporty', 'baby', 'ginger', 'posh']; //    ^?
const spiceGirls: string[];
```

This is the same syntax used in code samples, making it familiar and straightforward to use.

x??

---
#### Testing Complex Type Scenarios
Background context: The text highlights that testing types using external tools might not cover all edge cases. For instance, TypeScript's autocomplete feature can provide suggestions based on type intersections without fully enforcing strict type checking.

:p How do external tools like `dtslint` or `eslint-plugin-expect-type` handle complex type scenarios?
??x
These tools may miss certain cases where the type system allows for more flexibility than the static checks. For example, in a case like this:
```typescript
type Game = 'wordle' | 'crossword' | (string & {});
const spellingBee: Game = 'spelling bee';
let g: Game = '';
```
The TypeScript playground suggests "wordle" or "crossword", but allows any string to be assigned to `g`.

Such complex scenarios require a hybrid approach combining both static and dynamic type checks.

x??

---

#### Pay Attention to How Types Display
Background context explaining the importance of how types display. The way TypeScript libraries choose to display their types can greatly affect user experience and understanding.

In this example, we are dealing with a `PartiallyPartial` generic type that makes some properties optional while keeping others mandatory. This is implemented using TypeScript's utility types like `Partial`, `Pick`, and `Omit`.

:p How does the initial implementation of `PartiallyPartial` look?

??x
The initial implementation uses utility types to make some properties optional but results in a display that can be less user-friendly.

```typescript
type PartiallyPartial<T, K extends keyof T> = 
  Partial<Pick<T, K>> & Omit<T, K>;

interface BlogComment {
    commentId: number;
    title: string;
    content: string;
}

type PartComment = PartiallyPartial<BlogComment, 'title'>; // { Partial<Pick<BlogComment, "title">> & Omit<BlogComment, "title"> }
```

x??

---
#### Using Resolve to Improve Type Display
Explanation on how `Resolve` works and why it's useful for displaying types more clearly. `Resolve` is a homomorphic mapped type that flattens out the display of object properties.

:p What does the `Resolve` helper do?

??x
The `Resolve` helper uses a homomorphic mapped type to flatten out the display of all properties in an object type, making it clearer for users while hiding implementation details.

```typescript
type Resolve<T> = T extends Function ? T : {[K in keyof T]: T[K]};
```

Using this with the `PartiallyPartial` type makes the resulting type more user-friendly:

```typescript
type PartiallyPartial<T, K extends keyof T> = 
  Resolve<Partial<Pick<T, K>> & Omit<T, K>>;

type PartComment = PartiallyPartial<BlogComment, 'title'>; // { title?: string | undefined; commentId: number; content: string; }
```

x??

---
#### Dealing with Special Cases in Type Display
Explanation on how to handle special cases where the type parameter is `never`.

:p How does handling the case where `K` is `never` affect the display of `PartiallyPartial`?

??x
Handling the case where `K` is `never` allows for a more concise representation. In this scenario, no properties are marked as optional.

```typescript
type PartiallyPartial<T extends object, K extends keyof T> = 
  [K] extends [never]
    ? T // special case
    : T extends unknown
      ? Resolve<Partial<Pick<T, K>> & Omit<T, K>>
      : never;

type FullComment = PartiallyPartial<BlogComment, never>; // BlogComment
```

x??

---
#### Using Other Techniques to Adjust Type Display
Explanation on other techniques like `Exclude<keyof T, never>` and `unknown & T` or `{}` & T.

:p How can the use of `Exclude<keyof T, never>` improve type display?

??x
Using `Exclude<keyof T, never>` can inline `keyof` expressions, making them more readable:

```typescript
type Chan = keyof Color; // keyof Color
type ChanInline = Exclude<keyof Color, never>; // "r" | "g" | "b" | "a"
```

x??

---
#### Handling Classes in Resolve
Explanation on how `Resolve` can sometimes be too aggressive when applied to classes.

:p How does `Resolve` handle the `Date` class?

??x
`Resolve` can make types overly verbose, especially for classes. For example:

```typescript
type D = Resolve<Date>; // { toLocaleString: ...; ... 42 more methods; [Symbol.toPrimitive]: ...; }
```

In this case, it's better to let the type display as `Date`.

x??

---
#### Applying Resolve to Object Types
Explanation on how `Resolve` can inline object types.

:p How does using `unknown & T` or `{}` & T compare with `Resolve` for inlining object types?

??x
Both `unknown & T` and `{}` & T can be used to inline object types, but they are less brittle than `Resolve`.

```typescript
type U = unknown & { r: number; g: number; b: number; a: number }; // { r: number; g: number; b: number; a: number }
```

x??

---
#### Ensuring Legibility Without Sacrificing Clarity
Explanation on the importance of maintaining legibility in type displays.

:p How should you approach changing type displays to ensure clarity without sacrificing readability?

??x
When changing type displays, it's important to maintain legibility. Avoid compromising one aspect for another. For example:

```typescript
type ChanInline = Resolve<keyof Color>; // "r" | "g" | "b" | "a"
```

This improves readability while keeping the display clear.

x??

---

