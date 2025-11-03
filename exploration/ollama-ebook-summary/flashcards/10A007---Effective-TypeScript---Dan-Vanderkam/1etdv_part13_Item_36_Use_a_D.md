# Flashcards: 10A007---Effective-TypeScript---Dan-Vanderkam_processed (Part 13)

**Starting Chapter:** Item 36 Use a Distinct Type for Special Values

---

#### Special Values and Type Safety in TypeScript

Background context: The text discusses a common issue where special values (like -1) are used within functions without proper handling, leading to potential bugs. It highlights how using such values can reduce type safety and how explicitly handling these cases can improve the robustness of code.

:p What is the problem with using `-1` as a return value for `indexOf` in TypeScript?
??x
The problem arises because TypeScript cannot distinguish between a valid index and an invalid one (`-1`). This makes it difficult to handle both cases properly, leading to unexpected behavior when the value `-1` is returned.

For example:
```typescript
function splitAround<T>(vals: readonly T[], val: T): [T[], T[]] {
    const index = vals.indexOf(val);
    return [vals.slice(0, index), vals.slice(index + 1)];
}
```
When `val` is not found in the array, `index` will be `-1`, and this will cause issues with `slice`.

x??

---

#### Safe Index Of Function

Background context: To address the issue mentioned above, a function `safeIndexOf` was introduced that returns either a number or null to clearly distinguish between valid indices and invalid ones.

:p What is the purpose of the `safeIndexOf` function?
??x
The purpose of the `safeIndexOf` function is to return a number if the value is found in the array and null if it is not. This helps TypeScript differentiate between these two cases, improving type safety.

```typescript
function safeIndexOf<T>(vals: readonly T[], val: T): number | null {
    const index = vals.indexOf(val);
    return index === -1 ? null : index;
}
```

x??

---

#### Handling Special Values with `splitAround` Function

Background context: The `safeIndexOf` function was used to rewrite the `splitAround` function, ensuring that it correctly handles cases where the value is not found.

:p How does the revised `splitAround` function handle the case when the element is not present in the array?
??x
The revised `splitAround` function explicitly handles the null case returned by `safeIndexOf`. If the index is null (meaning the value was not found), it returns an empty array and the original array.

```typescript
function splitAround<T>(vals: readonly T[], val: T): [T[], T[]] {
    const index = safeIndexOf(vals, val);
    if (index === null) {
        return [[...vals], []];
    }
    return [vals.slice(0, index), vals.slice(index + 1)];
}
```

This ensures that the function behaves correctly even when the value is not found in the array.

x??

---

#### Using `null` vs. Special Values

Background context: The text emphasizes avoiding special values like -1 or 0 and suggests using null or undefined instead to represent special cases more clearly. It also mentions the importance of using tagged unions for complex state representations.

:p Why is it better to use `null` or `undefined` instead of in-domain special values like `-1`, `0`, or `""`?
??x
Using `null` or `undefined` as special values helps maintain type safety and clarity. These values are explicitly used to represent nullability, making the code more readable and easier to understand.

For example:
```typescript
interface Product {
    title: string;
    /** Price of the product in dollars, or undefined if price is unknown */
    priceDollars?: number | null;
}
```
This approach ensures that the type checker can distinguish between a valid `number` and an undefined value. Using tagged unions (like `ProductStatus`) for complex states can further enhance clarity and robustness.

x??

---

#### Tagged Unions for Complex States

Background context: The text suggests using tagged unions to represent special cases more explicitly, especially when dealing with complex state representations like network request statuses.

:p What is a tagged union, and why might it be preferable over using `null` or `undefined`?
??x
A tagged union (also known as an algebraic data type) is a way to model values that can take on one of several distinct forms. It helps avoid the ambiguity associated with nullables by explicitly representing each state.

For example, in modeling network request statuses:
```typescript
type NetworkStatus = {
    readonly status: 'pending';
} | {
    readonly status: 'error';
    readonly message: string;
} | {
    readonly status: 'success';
    readonly data: any;
};
```
This approach makes the intentions of each case clear and avoids the potential for bugs caused by null or undefined.

x??

---

#### Avoiding Optional Properties
When your application evolves, you might be tempted to add optional properties to existing interfaces. However, this can introduce bugs and make debugging more challenging.
:p What are some reasons to avoid using optional properties?
??x
Using optional properties can lead to several issues:
1. **Consistency Issues**: Missing a required property in multiple places (e.g., forgetting to pass `unitSystem` to `formatValue`).
2. **Debugging Complexity**: Harder to find missing or incorrect defaults, leading to inconsistent behavior.
3. **Type Safety**: Optional properties can introduce unsoundness issues in TypeScript.

For example, if you have a function like:
```typescript
function formatHike({miles, hours}, config: AppConfig) {
    const { unitSystem } = config;
    const distanceDisplay = formatValue({
        value: miles,
        units: 'miles',
        unitSystem // Only passed here
    });
    const paceDisplay = formatValue({
        value: miles / hours,
        units: 'mph'  // Missing unitSystem, leading to inconsistent display.
    });
    return `${distanceDisplay} at ${paceDisplay}`;
}
```
This can result in incorrect unit conversions and a confusing user experience.

x??

---

#### Type Splitting for Configurations
To handle evolving configurations while maintaining backward compatibility, you might want to split the type into two: one for un-normalized input and another for normalized use within the application.
:p How can you split AppConfig types to maintain compatibility?
??x
You can split `AppConfig` into two types: `InputAppConfig`, which allows optional properties, and a more specific `AppConfig` that requires all properties. This approach centralizes default value handling and ensures consistency.

For example:
```typescript
interface InputAppConfig {
    darkMode: boolean;
    // ... other settings ...
    /** default is imperial */
    unitSystem?: UnitSystem;
}

interface AppConfig extends InputAppConfig {
    unitSystem: UnitSystem;  // required
}
```

Additionally, you can provide normalization logic to ensure consistency:
```typescript
function normalizeAppConfig(inputConfig: InputAppConfig): AppConfig {
    return {
        ...inputConfig,
        unitSystem: inputConfig.unitSystem ?? 'imperial',
    };
}
```

This approach helps maintain backward compatibility while ensuring type safety and consistency.

x??

---

#### Optional Properties in Large Interfaces
Adding optional properties to large interfaces can lead to a combinatorial explosion of possible combinations, making it difficult to test all valid states.
:p Why is managing optional properties in large interfaces problematic?
??x
Managing optional properties in large interfaces can become overwhelming due to the exponential increase in potential combinations. For instance, if you have 10 optional properties, there are \(2^{10} = 1024\) possible combinations.

Testing all these combinations is impractical, and it's hard to ensure that all combinations make sense or are valid states for your application. This can lead to inconsistencies and bugs in your codebase.

For example:
```typescript
interface LargeInterface {
    prop1?: boolean;
    prop2?: number;
    // ... up to 10 properties ...
}
```

With 10 optional properties, you have \(2^{10} = 1024\) combinations. Testing all these states can be challenging and error-prone.

x??

---

#### Optional Properties and Backward Compatibility
Optional properties are often used when evolving APIs while maintaining backward compatibility with existing code.
:p When is it appropriate to use optional properties in evolving interfaces?
??x
Optional properties are necessary when:
1. **Describing Existing APIs**: You need to maintain compatibility with older versions of your API.
2. **Evolving APIs Gradually**: You want to introduce new features without breaking existing implementations.

However, they can lead to issues such as inconsistency and debugging complexity if not handled carefully. Therefore, it's important to weigh the benefits against potential drawbacks.

For example:
```typescript
interface OldInterface {
    prop1: boolean;
    prop2: number; // Required in old code
}
```

When evolving `OldInterface`, you might introduce optional properties like this:
```typescript
interface NewInterface extends OldInterface {
    prop3?: string;  // Optional property for new features
}
```

This approach allows for gradual evolution while maintaining compatibility with older implementations.

x??

---

#### Optional Properties in TypeScript
Background context explaining optional properties and their impact on type checking. Discuss how they can lead to repeated and possibly inconsistent code for default values. Explain that using required properties or creating distinct types can mitigate some of these issues.
:p What are the drawbacks of using optional properties in TypeScript?
??x
Optional properties can prevent the type checker from finding bugs and can lead to repeated and possibly inconsistent code for filling in default values. They should be used cautiously, considering if there's a valid alternative like making the property required or creating distinct types.
x??

---

#### DrawRect Function Example
Background context explaining how functions with multiple parameters of the same type can be error-prone. Discuss the importance of clear function invocations and how TypeScript can help catch errors. Provide an example of a problematic `drawRect` function call.
:p What does the given `drawRect` function call imply about the potential issues in function signatures?
??x
The given `drawRect` function call with parameters (25, 50, 75, 100, 1) is ambiguous without knowing the parameter types. This can lead to errors such as passing coordinates and dimensions out of order or incorrectly.
x??

---

#### Refactoring Functions for Clarity
Background context explaining why refactoring functions with multiple parameters into a single object can improve clarity and reduce errors. Discuss the use of `Point` and `Dimension` interfaces in TypeScript.
:p How does refactoring the `drawRect` function to take an object parameter help?
??x
Refactoring the `drawRect` function to take an object parameter, like this:
```typescript
interface Point { x: number; y: number; }
interface Dimension { width: number; height: number; }
function drawRect(params: {topLeft: Point, size: Dimension, opacity: number}) {
    // implementation
}
```
helps clarify the purpose of each parameter and improves type safety. The type checker can now distinguish between different types of parameters.
x??

---

#### Commutative Functions
Background context explaining when functions with commutative arguments are unambiguous. Provide examples of commutative and non-commutative functions.
:p Which functions are considered commutative, and why?
??x
Commutative functions like `max(a, b)` and `isEqual(a, b)` are unambiguous because the order of their parameters does not matter. For example, `max(5, 10)` and `max(10, 5)` both return the same result.
x??

---

#### Natural Order in Functions
Background context explaining how a natural order can reduce confusion but also noting that this might vary among developers. Provide examples of functions with a clear natural order.
:p How can the concept of "natural order" be applied to function parameters?
??x
The concept of "natural order" can apply to functions like `array.slice(start, stop)` where the order makes more sense (e.g., start before stop). However, this might vary among developers; for example, a date could be represented as `(year, month, day)`, `(month, day, year)`, or `(day, month, year)`. It's important to choose an order that best fits the context and ensure it is well-documented.
x??

---

#### Interface Design Principles
Background context explaining how interface design can improve function clarity and reduce errors. Discuss the importance of making interfaces easy to use correctly and hard to use incorrectly.
:p Why is it beneficial to use distinct types for parameters in functions?
??x
Using distinct types for parameters in functions, such as `Point` and `Dimension`, improves code clarity and helps the type checker catch incorrect invocations. This makes the interface easier to use correctly and harder to use incorrectly, leading to fewer bugs.
x??

---

