# Flashcards: 10A007---Effective-TypeScript---Dan-Vanderkam_processed (Part 14)

**Starting Chapter:** Item 39 Prefer Unifying Types to Modeling Differences

---

#### Prefer Unifying Types to Modeling Small Differences

Background context explaining the concept. TypeScript's type system allows for powerful transformations between types, but sometimes modeling these differences can lead to cognitive overhead and additional code. The recommendation is to unify types when possible, reducing complexity.

If applicable, add code examples with explanations.
:p What is the main idea of unifying types in TypeScript?
??x
The main idea is to prefer unification over modeling small differences between types. By unifying types, you reduce cognitive overhead and avoid errors that can occur from passing one version of a type to a function expecting another.

For example, consider the `Student` interface with camelCase names:
```typescript
interface Student {
    firstName: string;
    lastName: string;
    birthDate: string;
}
```
And the `StudentTable` interface using snake_case:
```typescript
interface StudentTable {
    first_name: string;
    last_name: string;
    birth_date: string;
}
```

If you write a function to convert between these types, like this:
```typescript
type ObjectToCamel<T> = { [K in keyof T as资本 : pascalCase<K>]: T[K] };

async function writeStudentToDb(student: Student) {
    await writeRowToDb(db, 'students', student);
    //                                     ~~~~~~~
    // Type 'Student' is not assignable to parameter of type 'StudentTable'.
}
```
You may encounter errors if you forget to convert between the types. It's simpler to have a single version of the `Student` type throughout your code.

x??

---
#### Using Template Literal Types for Conversion

Background context explaining the concept. TypeScript allows using template literal types to model transformations between different types, such as converting snake_case to camelCase and vice versa.

:p How can you use template literal types to convert between `StudentTable` and `Student` interfaces?
??x
You can use template literal types to generate one type from another. For instance, you can create a utility type like `ObjectToCamel<T>` that converts snake_case keys to camelCase:

```typescript
type ObjectToCamel<T> = { [K in keyof T as资本 : pascalCase<K>]: T[K] };
```

This way, you can define:
```typescript
type Student = ObjectToCamel<StudentTable>;
//   ^?
type Student = {
    firstName: string;
    lastName: string;
    birthDate: string;
};
```
With this utility type, TypeScript will automatically handle the conversion when necessary. This avoids the need to manually convert types every time you pass data between functions.

x??

---
#### Cognitive Overhead and Runtime Adjustments

Background context explaining the concept. Unifying types can simplify your codebase but might require adjustments in runtime logic. The goal is to reduce cognitive overhead by having a single consistent type across your application.

:p Why might unification of types be required?
??x
Unification of types may be required when you want to reduce cognitive overhead and ensure consistency throughout your codebase. For instance, if you have an interface like `StudentTable` that uses snake_case for database columns:
```typescript
interface StudentTable {
    first_name: string;
    last_name: string;
    birth_date: string;
}
```
And a type in TypeScript that typically uses camelCase property names:
```typescript
interface Student {
    firstName: string;
    lastName: string;
    birthDate: string;
}
```

Unifying these types can reduce the need for explicit conversion code and make your code easier to understand. However, this might require adjustments in runtime logic where you interact with databases or APIs.

x??

---
#### Handling Different Versions of Types

Background context explaining the concept. When different versions of a type are needed, you must ensure proper conversions between them. This can be error-prone if not managed correctly.

:p How do you handle conversion between `Student` and `StudentTable` interfaces?
??x
Handling conversion between `Student` and `StudentTable` requires writing functions to convert one type to the other. For example, you might define a function like this:
```typescript
function objectToSnake<T>(obj: T): { [K in keyof T as snakeCase<K>]: T[K] } {
    return Object.fromEntries(
        Object.entries(obj).map(([key, value]) => [snakeCase(key), value])
    );
}

async function writeStudentToDb(student: Student) {
    await writeRowToDb(db, 'students', objectToSnake(student));
}
```
This ensures that when you pass a `Student` type to the database, it is properly converted to the required `StudentTable` format. Without this conversion, TypeScript would raise an error due to type mismatches.

x??

---
#### Avoiding Unification When Necessary

Background context explaining the concept. There are cases where unifying types may not be feasible or beneficial. For example, when working with tagged unions or stateful data that should remain distinct.

:p In what scenarios might you not want to unify types?
??x
You might not want to unify types in scenarios where the types represent fundamentally different states or entities. For instance, using a tagged union for enum-like behavior:

```typescript
type Action =
    | { type: 'LOGIN'; username: string }
    | { type: 'LOGOUT' };
```

Unifying such distinct states could lead to confusion and potential bugs in your application logic.

x??

---

#### Prefer Imprecise Types to Inaccurate Types
Background context: When writing type declarations, you often have a choice between being precise and being imprecise. Being precise means using more specific types that better capture the exact nature of the data or functionality. However, precision comes with risks—increased chances of mistakes and potential inaccuracy if real-world use cases exceed the assumptions made during type declaration.
:p Why is it advised to prefer imprecise types over inaccurate ones?
??x
It's recommended to err on the side of imprecision because inaccurate types can lead to incorrect behavior and require users to work around your type system. For example, if you define a coordinate as `number[]` but later find out that coordinates can also include an elevation (a third value), using `GeoPosition = [number, number]` would be inaccurate. Users may have to use type assertions or disable the type checker entirely.
```typescript
// Example of inaccurate types causing issues
interface Point {
    type: 'Point';
    coordinates: GeoPosition; // Inaccurate as it doesn't account for elevation
}

type GeoPosition = [number, number];
```
x??

---
#### Geographical Data Types in TypeScript
Background context: When dealing with geographical data formats like GeoJSON, you often have to define the types of coordinate arrays. The provided example uses `number[]` for coordinates but suggests using a tuple type `[number, number]` to be more precise.
:p How can precision in type declarations affect user experience?
??x
Precision in type declarations is beneficial because it helps catch bugs and takes advantage of TypeScript's tooling. However, overly precise types that are inaccurate can cause issues. For instance, if you define `GeoPosition = [number, number]` but GeoJSON allows an elevation as a third value, your type declaration becomes inaccurate.
```typescript
// Example of inaccurate tuple type for coordinates
type GeoPosition = [number, number]; // Inaccurate as it doesn't account for elevation

interface Point {
    type: 'Point';
    coordinates: GeoPosition;
}
```
x??

---
#### Precision vs. Accuracy in Type Declarations
Background context: When defining types for complex data structures like Lisp-like expressions used by the Mapbox library, there's a spectrum of precision you can choose from. The more precise your types are, the harder it is to make mistakes but also the higher the chance that they might be inaccurate.
:p What are the trade-offs between precise and imprecise type declarations?
??x
The trade-off lies in the balance between accuracy and usability. Precise types provide better validation and tooling support but can introduce errors if real-world data does not fully align with your expectations. In contrast, imprecise types like `any` or broader unions (`number | string | any[]`) are more flexible and less likely to cause issues at the cost of potentially losing some benefits of static typing.
```typescript
// Example of imprecise type declaration
type Expression1 = any;
```
x??

---
#### Mapbox Expressions Typing in TypeScript
Background context: The Mapbox library uses JSON to define expressions, which can include various data types and function calls. When writing TypeScript declarations for such expressions, you have several levels of precision to choose from.
:p What are the different levels of precision when typing Lisp-like expressions?
??x
The different levels of precision for typing Lisp-like expressions in Mapbox are as follows:

1. **Allow anything**: `type Expression = any;`
2. **Allow strings, numbers, and arrays**: `type Expression = number | string | any[];`
3. **Allow strings, numbers, and arrays starting with known function names**: This requires more complex type definitions.
4. **Ensure each function gets the correct number of arguments**.
5. **Ensure each function gets the correct type of arguments**.

Each level increases in complexity but also potential accuracy.
```typescript
// Example of a precise but potentially inaccurate type declaration
type Expression2 = (fn: string, ...args: any[]) => any;
```
x??

---

#### Improved Type Precision for Expressions
Background context: The original types were too broad, leading to false negatives where valid expressions were not flagged correctly. We aim to improve precision by making our type system more specific while ensuring that we don't introduce regressions (i.e., stop working on existing valid expressions).

:p How can you define a more precise `Expression` type in TypeScript?
??x
To define a more precise `Expression` type, we use a combination of unions and recursive types. Initially, we define the possible function names (`FnName`) and then create an array that only allows these function names as the first element.

```typescript
type FnName = '+' | '-' | '*' | '/' | '>' | '<' | 'case' | 'rgb';
type CallExpression = [FnName, ...any[]];
type Expression3 = number | string | CallExpression;
```

This type ensures that only valid function names are used and any number of arguments can be passed. However, to enforce the correct number of arguments for specific functions like `case`, we use a recursive approach with interfaces.

Here’s how you can define more precise types:

```typescript
type Expression4 = number | string | CallExpression;
type CallExpression = MathCall | CaseCall | RGBCall;

type MathCall = [
  '+' | '-' | '/' | '*' | '>' | '<',
  Expression4,
  Expression4,
];

interface CaseCall {
  0: 'case';
  [n: number]: Expression4;
  length: 4 | 6 | 8 | 10 | 12 | 14 | 16; // etc.
}

type RGBCall = ['rgb', Expression4, Expression4, Expression4];
```

With these types, the compiler can catch errors such as invalid function names or incorrect argument counts.

??x
The answer with detailed explanations:
This approach ensures that each `CallExpression` has a fixed number of arguments and enforces correct usage. For example, `CaseCall` uses an interface to ensure it always contains exactly four values: `'case'`, followed by two expressions for the condition and true value, then two more for the false value.

Recursive types are used here because we need to handle nested function calls correctly. The `length` property in `CaseCall` enforces that only specific lengths of arrays are valid, preventing incorrect argument counts.

```typescript
const okExpressions: Expression4[] = [
  10,
  "red",
  ["+", 10, 5],
  ["rgb", 255, 128, 64],
  ["case", [">", 20, 10], "red", "blue"],
];

const invalidExpressions: Expression4[] = [
  true,
  // Error: Type 'boolean' is not assignable to type 'Expression4'
  ["**", 2, 31], // Error: Type '\"**\"' is not assignable to type '\"+\" | \"-\" | \"/\" | ...'
  ["rgb", 255, 0, 127, 0],
  // Error: Type 'number' is not assignable to type 'undefined'.
  ["case", [">", 20, 10], "red", "blue", "green"],
];
```

??x
---

#### Recursive Types for Function Arguments
Background context: To ensure each function call has the correct number of arguments, we need to create recursive types that can handle nested calls.

:p How do you enforce the correct number of arguments in a `CaseCall`?
??x
To enforce the correct number of arguments in a `CaseCall`, we use an interface with a `length` property. This ensures that the array always has a specific length, which corresponds to the valid argument counts for the function.

```typescript
interface CaseCall {
  0: 'case';
  [n: number]: Expression4;
  length: 4 | 6 | 8 | 10 | 12 | 14 | 16; // etc.
}
```

This interface states that `CaseCall` must have the first element as `'case'`, followed by any number of elements (since `[n: number]: Expression4`), and a fixed length.

??x
The answer with detailed explanations:
Using an interface for `CaseCall` allows TypeScript to enforce specific constraints on the array. The `0` index ensures that the function name is always `'case'`, while the generic type `Expression4` allows any valid expression as arguments. The `length` property enforces a fixed number of elements, ensuring that only arrays with the correct argument count are allowed.

For example:

```typescript
const okExpressions: Expression4[] = [
  ["case", [">", 20, 10], "red", "blue"],
];

const invalidExpressions: Expression4[] = [
  ["case", [">", 20, 10]], // Error: length is too short
  ["case", [">", 20, 10], "red", "blue", "green"], // Error: length is too long
];
```

??x
---

#### Recursive Types and Function Calls
Background context: To handle nested function calls correctly, we need to ensure that our types are recursive. This allows us to validate not just top-level expressions but also those within deeper levels of nesting.

:p How can you create a type that handles nested `CallExpression`?
??x
To create a type that handles nested `CallExpression`, we define the base type and then recursively allow for further calls.

```typescript
type CallExpression = MathCall | CaseCall | RGBCall;

type MathCall = [
  '+' | '-' | '/' | '*' | '>' | '<',
  Expression4,
  Expression4,
];

interface CaseCall {
  0: 'case';
  [n: number]: Expression4;
  length: 4 | 6 | 8 | 10 | 12 | 14 | 16; // etc.
}

type RGBCall = ['rgb', Expression4, Expression4, Expression4];
```

This type system ensures that each `CallExpression` can be a `MathCall`, `CaseCall`, or `RGBCall`. Each of these types enforces specific constraints on the number and type of arguments.

??x
The answer with detailed explanations:
By defining recursive types like this, we ensure that nested function calls are also correctly typed. For example:

- `MathCall` defines a simple arithmetic operation with exactly two operands.
- `CaseCall` ensures that the `case` function is called with specific numbers of arguments (4 or 6).
- `RGBCall` enforces the correct number of color values.

This approach allows us to validate deeply nested expressions without making assumptions about their structure. The recursive nature of the types means that any valid call can be checked and validated, even if it's part of a larger expression.

```typescript
const okExpressions: Expression4[] = [
  ["+"], // Error: Needs more arguments
  [">", 20, 10],
];

const invalidExpressions: Expression4[] = [
  ["**", 2, 31], // Error: '**' is not a valid function name
  ["rgb", 255, 0, 127, 0, 64], // Error: Invalid length for RGB call
];
```

??x
---

#### TypeScript Improvements and Challenges
Background context: The provided text discusses improvements in TypeScript's type checking, highlighting both benefits and challenges. It focuses on a specific `Expression` type declaration and its errors, emphasizing precision versus accuracy in type declarations.

:p What is an example of a complex but inaccurate type declaration mentioned in the text?
??x
An example given is the `Expression4` type declaration, which is meant to be precise about math operations but ends up being overly strict. For instance:
```typescript
const moreOkExpressions: Expression4[] = [
    ['-', 12], // Incorrectly flagged as an error because it only has one parameter.
    ['+', 1, 2, 3], // Incorrectly flagged as an error because a number is not assignable to undefined.
    ['*', 2, 3, 4] // Incorrectly flagged as an error for the same reason as above.
];
```
x??

---

#### Uncanny Valley of Type Safety
Background context: The text uses the "uncanny valley" metaphor to describe how precise types improve type safety but can also introduce inaccuracies if not carefully managed. It emphasizes that overly complex and inaccurate types can lead to user frustration.

:p What is the uncanny valley metaphor in the context of TypeScript?
??x
The uncanny valley metaphor refers to the point where very detailed and realistic models become too accurate, leading users to focus on minor inaccuracies. In TypeScript, this means that while precise type declarations enhance safety, overly complex types can introduce errors that are harder to spot and fix.

For example, in `Expression4`, the type declaration is too strict because it requires all math operators (`+`, `-`, `*`) to take exactly three parameters, even though the actual specification allows some flexibility:
```typescript
const moreOkExpressions: Expression4[] = [
    ['-', 12], // Incorrectly flagged as an error.
    ['+', 1, 2, 3], // Incorrectly flagged as an error.
    ['*', 2, 3, 4] // Incorrectly flagged as an error.
];
```
x??

---

#### Importance of Testing Precise Types
Background context: The text stresses the need for thorough testing when refining precise types to ensure that they remain accurate and useful. It highlights how complex code requires more tests.

:p Why is expanding your test set important when refining TypeScript types?
??x
Expanding your test set is crucial because it helps ensure that the refined, precise types are both accurate and robust. For example, in `Expression4`, while making the type more precise by specifying exact parameter counts for operators like `+`, `-`, and `*` can improve accuracy, it might also introduce errors if not thoroughly tested.

You should test a wide range of valid expressions to confirm that your types correctly identify only invalid ones. This ensures that your improved type declarations do not create false positives or negatives.
x??

---

#### Balancing Precision and Accuracy
Background context: The text discusses how complex but inaccurate type declarations can be worse than simpler, less precise ones. It advises against modeling inaccurately if a precise model cannot be achieved.

:p What advice is given regarding the balance between precision and accuracy in TypeScript types?
??x
The advice is to avoid creating complex and inaccurate type declarations when they do not fully capture the intended behavior. If you can't accurately model something, use `any` or `unknown` instead of an inaccurate representation. This approach maintains higher overall type safety while avoiding misleading errors.

For example, in `Expression4`, rather than over-constraining the number of parameters for operators like `+`, `-`, and `*`, it might be better to use a more flexible but less precise type that doesn't introduce false positives.
x??

---

#### Ambiguous Type Definitions
Background context explaining why ambiguous type definitions can lead to confusion and incorrect mental models. The example provided illustrates how a general term like `name` is not specific enough, while fields like `endangered` are ambiguous without proper context.

:p What issues arise from using general or ambiguous terms in type definitions?
??x
General and ambiguous terms such as `name`, `endangered`, and `habitat` can lead to unclear intentions and make it difficult for developers to understand the exact meaning of these fields. For example, the term `name` could refer to a scientific name or a common name, while the `endangered` field might mean "endangered or worse" or literally "endangered." This ambiguity can cause confusion and incorrect assumptions.

```typescript
interface Animal {
    name: string; // General term with multiple possible meanings
    endangered: boolean; // Ambiguous - could mean "endangered or worse"
    habitat: string; // Broad term that needs more specificity
}
```
x??

---

#### Precise Type Definitions
Background context explaining the importance of precise and domain-specific type definitions. The example shows how renaming fields to more specific terms improves clarity and provides a standard classification system.

:p How can you improve the precision and clarity of type definitions?
??x
By using more specific terms that better describe the intended data, such as `commonName`, `genus`, `species` for names, and `status` with a `ConservationStatus` enum to represent conservation levels. Additionally, renaming `habitat` to `climates` with a standard taxonomy like the Köppen climate classification enhances clarity.

```typescript
interface Animal {
    commonName: string; // Specific term for common name
    genus: string;      // Specific term for genus
    species: string;    // Specific term for species
    status: ConservationStatus; // Standard classification system
    climates: KoppenClimate[];   // Standard climate classification
}

type ConservationStatus = 'EX' | 'EW' | 'CR' | 'EN' | 'VU' | 'NT' | 'LC'; 
type KoppenClimate = 
    'Af' | 'Am' | 'As' | 'Aw' | 
    'BSh' | 'BSk' | 'BWh' | 'BWk' |
    'Cfa' | 'Cfb' | 'Cfc' | 'Csa' | 'Csb' | 'Csc' | 'Cwa' | 'Cwb' | 'Cwc' | 
    'Dfa' | 'Dfb' | 'Dfc' | 'Dfd' |
    'Dsa' | 'Dsb' | 'Dsc' | 'Dwa' | 'Dwb' | 'Dwc' | 'Dwd' |
    'EF' | 'ET';

const snowLeopard: Animal = {
    commonName: 'Snow Leopard', 
    genus: 'Panthera', 
    species: 'Uncia', 
    status: 'VU',  // Vulnerable
    climates: ['ET', 'EF', 'Dfd'],  // Alpine or subalpine
};
```
x??

---

#### Reusing Domain Vocabulary
Background context explaining the importance of reusing domain-specific vocabulary to enhance clarity and reduce ambiguity. The example highlights how using standardized terms like `ConservationStatus` and `KoppenClimate` can make the code more understandable.

:p Why is it important to use domain-specific vocabulary in type definitions?
??x
Using domain-specific vocabulary helps maintain consistency and clarity, making the code easier to understand for both current and future developers. By reusing standardized terms from the problem domain, you avoid ambiguity and provide clear meanings. For example, using `ConservationStatus` with a standard classification system ensures everyone understands the exact meaning of "endangered," while `KoppenClimate` provides a consistent way to describe habitats.

```typescript
type ConservationStatus = 'EX' | 'EW' | 'CR' | 'EN' | 'VU' | 'NT' | 'LC'; 
type KoppenClimate = 
    'Af' | 'Am' | 'As' | 'Aw' | 
    'BSh' | 'BSk' | 'BWh' | 'BWk' |
    'Cfa' | 'Cfb' | 'Cfc' | 'Csa' | 'Csb' | 'Csc' | 'Cwa' | 'Cwb' | 'Cwc' | 
    'Dfa' | 'Dfb' | 'Dfc' | 'Dfd' |
    'Dsa' | 'Dsb' | 'Dsc' | 'Dwa' | 'Dwb' | 'Dwc' | 'Dwd' |
    'EF' | 'ET';
```
x??

---

#### Naming Conventions
Background context explaining the importance of clear and meaningful names in type definitions to improve code clarity. The example contrasts ambiguous terms with specific, domain-relevant terms.

:p What is the benefit of using more specific names for fields?
??x
Using more specific names helps clarify the purpose and meaning of each field, reducing ambiguity and making the code easier to understand. For instance, `commonName` clearly indicates that it refers to a common name, while `genus` and `species` provide precise taxonomic information. This clarity ensures that developers can quickly grasp the intent behind each field without needing external documentation.

```typescript
interface Animal {
    commonName: string; // Specific term for common name
    genus: string;      // Specific term for genus
    species: string;    // Specific term for species
}
```
x??

---

