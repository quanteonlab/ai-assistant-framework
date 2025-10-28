# Flashcards: 10A007---Effective-TypeScript---Dan-Vanderkam_processed (Part 15)

**Starting Chapter:** Item 42 Avoid Types Based on Anecdotal Data

---

#### Reuse Domain Terms for Readability and Clarity
Properly naming variables, functions, and types according to domain terms can enhance readability and maintainability. Vague or generic names like "data," "info," or "entity" should be avoided if they do not convey specific meaning.
:p How does using meaningful names in code improve the overall quality of the program?
??x
Using meaningful names improves code comprehension by making it easier for other developers (and yourself) to understand what each piece of code represents. This reduces the risk of errors and makes refactoring and maintenance more straightforward.

For example, consider the following two snippets:
```typescript
// Vague naming
function processData(data: any[]) { ... }

// Meaningful naming
function analyzeSalesData(salesData: any[]) { ... }
```
In the second snippet, "analyzeSalesData" clearly indicates the purpose of the function, making it easier to understand and maintain.

x??

---
#### Avoid Using Different Names for the Same Thing
To avoid confusion and ensure clarity, names should be distinct and meaningful. If multiple terms are used interchangeably for the same concept, this can lead to inconsistencies and misunderstandings.
:p Why is using different names for the same thing problematic in coding?
??x
Using different names for the same thing can lead to confusion and inconsistencies within a codebase. This makes it harder to understand the intent of the code and increases the risk of bugs.

For example:
```typescript
// Problematic usage
function calculateBoundingBox(feature: GeoJSONFeature): BoundingBox | null { ... }

interface GeoJSONFeature {
    type: 'Feature';
    geometry: IGeoNodeList | null; // This is different from coordinates in bounding box logic
    properties: unknown;
}

type IGeoNodeList = any[][];
```
Here, `IGeoNodeList` might be confusingly used for `coordinates`, leading to potential bugs or misunderstandings.

x??

---
#### Import Types from Official Sources
Using officially defined types can help ensure that your code adheres to established standards and avoid unintended behavior. Writing custom types based on anecdotal data may miss important edge cases.
:p Why is it better to import official types rather than defining them yourself?
??x
Importing official types ensures that your code aligns with standardized specifications, which can reduce the risk of errors and make your code more predictable.

For example:
```typescript
import { GeoJSONFeature } from 'geojson-types';

// Instead of:
interface GeoJSONFeature {
    type: 'Feature';
    geometry: IGeoNodeList | null; // Potentially incorrect or incomplete definition
    properties: unknown;
}
```
By using the official `GeoJSONFeature` type, you leverage existing and well-tested definitions.

x??

---
#### Define Types Based on Official Specifications
When defining types based on external specifications, it is crucial to follow those definitions accurately. This helps ensure that your code aligns with expected data structures.
:p How should you define a GeoJSONFeature type according to official specifications?
??x
To define a `GeoJSONFeature` type according to official specifications, you should closely adhere to the structure defined by the GeoJSON standard.

```typescript
import { GeoJSONFeature } from 'geojson-types';

// Alternatively:
interface GeoJSONFeature {
    type: 'Feature';
    geometry: GeoJSONGeometry | null;
    properties: Record<string, unknown>;
}

interface GeoJSONGeometry {
    type: 'Point' | 'LineString' | 'Polygon' | 'MultiPolygon';
    coordinates: number[] | number[][] | number[][][] | number[][][][];
}
```
This ensures that your `GeoJSONFeature` and its components follow the correct structure, reducing the risk of mismatches.

x??

---

#### GeoJSON Type Declaration Importance
Background context: The provided text discusses the importance of using formal type declarations from a specification, such as the GeoJSON spec, to avoid bugs and ensure robustness. TypeScript is used here for type safety.

:p Why are community-sourced TypeScript type declarations important in this scenario?
??x
Using community-sourced TypeScript type declarations, like those available on DefinitelyTyped for GeoJSON, ensures that your code adheres strictly to the formal specification. This reduces the likelihood of bugs and increases code reliability by catching issues early during development.

For example, consider the following code snippet:
```typescript
import { Feature } from 'geojson';
function calculateBoundingBox(f: Feature): BoundingBox | null {
    let box: BoundingBox | null = null;
    const helper = (coords: any[]) => { /* ... */ };
    const { geometry } = f;
    if (geometry) {
        helper(geometry.coordinates);
    }
    return box;
}
```
This code might work fine for most geometries, but it fails when `geometry` is a `GeometryCollection`, because such an object does not have the `coordinates` property. This is caught by TypeScript due to the accurate type definitions.

x??

---

#### Handling GeometryCollections in GeoJSON
Background context: The text explains that a GeoJSON geometry can be of various types, including `GeometryCollection`. The latter does not have a `coordinates` property, which can lead to runtime errors if not handled correctly. The solution is to either disallow `GeometryCollection` or handle it properly.

:p How do you ensure your code handles `GeometryCollection`s in GeoJSON?
??x
To handle `GeometryCollection`s in GeoJSON, you need to check the type of each geometry and process them accordingly. For example, if a feature's geometry is a `GeometryCollection`, you can throw an error or recursively call a helper function on each individual geometry within the collection.

Here’s how you can implement this logic:
```typescript
const { geometry } = f;
if (geometry) {
    if (geometry.type === 'GeometryCollection') {
        throw new Error('GeometryCollections are not supported.');
    }
    helper(geometry.coordinates); // OK for other geometry types
}
```

Alternatively, you can use a recursive function to handle `GeometryCollection` properly:
```typescript
const geometryHelper = (g: Geometry) => {
    if (g.type === 'GeometryCollection') {
        g.geometries.forEach(geometryHelper);
    } else {
        helper(g.coordinates); // OK for other geometry types
    }
};

const { geometry } = f;
if (geometry) {
    geometryHelper(geometry);
}
```

These approaches ensure that your code can correctly process all valid GeoJSON geometries and avoid runtime errors.

x??

---

#### API Client Libraries in TypeScript
Background context: The text suggests using official TypeScript client libraries for APIs, or generating types from an official source if such a library is not available. This is especially useful with GraphQL APIs, which provide schemas that can be used to generate type definitions.

:p Why should you use official TypeScript clients or generate types from the API documentation?
??x
Using official TypeScript clients or generating types from the API documentation ensures that your code is aligned with the API's specifications and reduces potential bugs. Official libraries are maintained by the API providers and thus are up-to-date, accurate, and well-documented.

For example, in a GraphQL scenario, you can use tools like `graphql-codegen` to generate TypeScript type definitions from the schema provided by the API. This helps in creating robust and type-safe client code that adheres to the expected data structures.

Here’s an example of how you might set up `graphql-codegen`:
```yaml
# .codegen.yml configuration file
schema: "http://example.com/graphql"
generates:
  src/types.ts:
    plugins:
      - 'typescript'
```
Running `npx graphql-codegen` will generate the TypeScript types based on the provided schema.

x??

---

#### OpenAPI Schema and Type Safety
Background context: The provided text discusses using an OpenAPI schema to generate type-safe interfaces for interacting with REST APIs. An OpenAPI schema is a JSON file that describes endpoints, HTTP verbs, and request/response types.

:p What is the purpose of generating types from an OpenAPI schema?
??x
The purpose is to ensure type safety when working with API requests and responses in TypeScript or other languages that support strong typing. By using generated interfaces, you can avoid runtime errors related to incorrect data types and improve code maintainability.
??x

---

#### Extracting Schemas Using JSON Schema Tools
Background context: The text explains how to use tools like `jq` and `json-schema-to-typescript` to extract schema definitions from an OpenAPI JSON file.

:p How would you extract a specific schema definition using `jq`?
??x
To extract the schema definition for "CreateCommentRequest" from an OpenAPI JSON file named `schema.json`, you can use the following command:
```bash
$ jq '.components.schemas.CreateCommentRequest' schema.json > comment.json
```
This command uses `jq` to filter and output only the relevant schema section.
??x

---

#### Generating TypeScript Interfaces From Schemas
Background context: The text demonstrates how to generate a TypeScript interface for an API request using `json-schema-to-typescript`.

:p How can you generate a TypeScript interface from a JSON schema file?
??x
You can use the `json-schema-to-typescript` tool to generate a TypeScript interface. For example, if your JSON schema is in `comment.json`, you would run:
```bash
$ npx json-schema-to-typescript comment.json > comment.ts
```
This command generates a TypeScript file named `comment.ts` that contains the appropriate interfaces.
??x

---

#### Ensuring Type Synchronicity with APIs
Background context: The text discusses the importance of keeping generated types in sync with API schema changes.

:p Why is it important to keep your generated types up-to-date with the API schema?
??x
It's crucial because any changes in the API, such as adding or removing fields, can lead to type mismatches if not reflected in your local code. Keeping the types in sync ensures that your application remains robust and error-free.
??x

---

#### Handling Missing Specifications
Background context: The text mentions scenarios where official schemas might not be available.

:p What should you do if there is no official schema or specification for an API?
??x
If there's no official schema, you can use tools like `quicktype` to generate types from the data. However, it’s important to verify the generated types against real usage scenarios to ensure they accurately represent the expected input and output.
??x

---

#### Auto-Generated TypeScript Declarations for DOM API
Background context: The text explains how TypeScript's type declarations for the browser DOM API are auto-generated.

:p How do TypeScript's type declarations for the browser DOM API benefit developers?
??x
TypeScript’s type declarations for the DOM API, generated from MDN documentation, provide a clear and accurate model of web APIs. This ensures that developers can write more reliable and type-safe code by leveraging these pre-defined types.
??x

---

#### Strategies for Keeping Types in Sync
Background context: The text suggests strategies to handle maintaining sync between your locally generated types and the API schema.

:p What are some strategies for keeping your generated types up-to-date with an API's evolving schema?
??x
Strategies include regularly updating the schema file from the official source, using version control systems, and implementing automated checks or scripts that compare the local schemas against the latest API documentation. This ensures that your application remains aligned with the API’s specifications.
??x

---

#### Using any Wisely in TypeScript
Background context: In TypeScript, the `any` type is a powerful but potentially dangerous tool. It can be used to bypass type checking for parts of your code, making it easier to work with legacy or dynamically typed JavaScript codebases. However, improper use can lead to runtime errors and loss of type safety.

:p How does the scoped `any` type differ from globally using `any` in terms of impact on type safety?
??x
The scoped `any` type is limited to a single expression within a function argument or statement, ensuring that other parts of the code maintain their original types. In contrast, using `any` globally can cause "contagious" effects where an `any` type might spread throughout your codebase.

For example:
```typescript
function eatDinner1() {
    const pizza: any = getPizza();
    eatSalad(pizza); // ok but unsafe
    pizza.slice();  // This call is completely unchecked.
}

function spiceItUp() {
    const pizza = eatDinner1(); // const pizza: any
    pizza.addRedPepperFlakes(); // This call is also unchecked.
}
```
In the above example, `eatDinner1` uses an `any` type for the local variable `pizza`, making subsequent calls to `slice()` or methods on `pizza` unchecked. However, returning this `any` value from `eatDinner1` and using it in another function (`spiceItUp`) can lead to unexpected `any` types spreading across your codebase.

In contrast:
```typescript
function eatDinner2() {
    const pizza = getPizza();
    eatSalad(pizza as any); // This is preferable because the type checking remains localized.
    pizza.slice();  // This call is safe and checked.
}
```
Here, `as any` only affects the specific call to `eatSalad`, keeping the rest of the code with a `Pizza` type.

x??

---

#### Narrow Scope for any Types
Background context: The use of `any` should be restricted to the smallest possible scope to minimize potential errors and maintain overall type safety. This is particularly important when dealing with legacy or dynamically typed JavaScript codebases, where you might need to bypass type checking temporarily but want to avoid losing the benefits of static typing elsewhere.

:p Why is it better to use `eatSalad(pizza as any)` over declaring a variable with `any` type?
??x
Using `eatSalad(pizza as any)` ensures that only the specific call to `eatSalad` is unchecked, while maintaining the original type for `pizza` in other parts of the function. This approach minimizes the impact on your code's type safety and reduces the risk of introducing bugs.

For example:
```typescript
function eatDinner2() {
    const pizza = getPizza();
    eatSalad(pizza as any); // This is preferable.
    pizza.slice();  // This call remains checked, ensuring type safety.
}
```
In this case, `pizza` retains its original type of `Pizza`, and the only unchecked operation is the call to `eatSalad`. If you were to use a variable declared with `any`, like so:
```typescript
function eatDinner1() {
    const pizza: any = getPizza();
    eatSalad(pizza);  // This is unchecked.
    pizza.slice();   // Also unchecked, which can be dangerous.
}
```
This would make the entire function less type-safe and harder to maintain.

x??

---

#### Contagious any Types
Background context: An `any` return type in a function can spread throughout your codebase if not properly managed. This is because TypeScript does not track the lifetime of types declared with `any` beyond their initial declaration, leading to potential runtime errors and loss of type safety.

:p What are the implications of returning an `any` type from a function?
??x
Returning an `any` type from a function can lead to "contagious" effects where the `any` type might spread throughout your codebase. This is because TypeScript does not enforce or track the type after it leaves the function scope, potentially leading to runtime errors and loss of type safety.

For example:
```typescript
function eatDinner1() {
    const pizza: any = getPizza();
    eatSalad(pizza);  // This call is unchecked.
    pizza.slice();   // Also unchecked, which can be dangerous.
    return pizza;    // Now `pizza` has an `any` type outside the function.
}

function spiceItUp() {
    const pizza = eatDinner1();  // const pizza: any
    pizza.addRedPepperFlakes();  // This call is also unchecked, leading to potential errors.
}
```
In this scenario, returning a variable declared with `any` type can lead to unexpected and unchecked operations in other parts of your codebase.

Using the more narrowly scoped approach with `as any` within function arguments or expressions can help limit the impact:
```typescript
function eatDinner2() {
    const pizza = getPizza();
    eatSalad(pizza as any);  // This call is checked.
    pizza.slice();          // Also checked, ensuring type safety.
}
```
Here, only the specific call to `eatSalad` is unchecked, while maintaining type safety elsewhere in the function.

x??

---

#### Suppressing Type Errors with @ts-ignore and @ts-expect-error
Background context: In some cases, you might need to suppress a type error temporarily. TypeScript provides two directives for this purpose: `@ts-ignore` and `@ts-expect-error`. These allow you to silence errors on specific lines of code without globally affecting the type safety of your codebase.

:p What is the difference between using @ts-ignore and @ts-expect-error?
??x
The primary difference between `@ts-ignore` and `@ts-expect-error` lies in how they are scoped:

- **@ts-ignore**: This directive silences an error on a specific line of code but does not provide any additional information if the error is resolved later. It is scoped to a single line and does not propagate beyond that point.

```typescript
function eatDinner1() {
    const pizza = getPizza();  // @ts-ignore
    eatSalad(pizza);           // Error suppressed for this line only.
    pizza.slice();             // Still checked.
}
```

- **@ts-expect-error**: This directive also silences an error but provides more context. If the error is resolved later, TypeScript will issue a new warning, allowing you to remove the directive and ensuring that your code remains type-safe.

```typescript
function eatDinner2() {
    const pizza = getPizza();  // @ts-expect-error
    eatSalad(pizza);           // Error suppressed for this line only.
    pizza.slice();             // Still checked, but you will be notified if the error is resolved.
}
```

Using `@ts-expect-error` can help you catch and resolve type issues later while maintaining the benefits of static typing. Conversely, using `@ts-ignore` without further review might leave unresolved issues in your code.

x??

---

#### Controlling Type Safety with as any
Background context: When dealing with complex objects or properties that are dynamically typed, you might need to silence specific type errors on individual properties within a larger object. Using `as any` around the entire object can be risky and may disable type checking for all properties. Instead, using `as any` on individual properties can limit the scope of the impact.

:p How does using as any on an individual property differ from using it on the whole object?
??x
Using `as any` on an individual property within a larger object limits the scope of type errors to that specific property only, while maintaining type safety for other properties in the same object. This approach is more controlled and less likely to introduce global type issues.

For example:
```typescript
const config: Config = {
    a: 1,
    b: 2,
    c: { // Type error here
        key: value // ~~~ Property 'key' is missing in type 'Bar'
    }
};
```
Using `as any` on the entire object can silence all errors but might also disable type checking for other properties:
```typescript
const config: Config = {
    a: 1,
    b: 2,
    c: { // Using as any here silences all errors in this object.
        key: value
    } as any; // Don't do this.
};
```

In contrast, using `as any` only on the problematic property:
```typescript
const config: Config = {
    a: 1,
    b: 2,
    c: { // Only the 'key' error is silenced.
        key: value as any
    }
};
```
This approach ensures that other properties (`a` and `b`) remain checked, maintaining overall type safety.

x??

---

#### Prefer More Precise Variants of any
Background context: In TypeScript, using `any` can lead to a loss of type safety and make debugging harder. It's generally better to use more specific types when possible to ensure that your code is as robust as it could be.

:p What are the disadvantages of using `any` in TypeScript functions?
??x
Using `any` can lead to several issues:
- Loss of type checking, meaning you won't get compile-time errors for invalid operations.
- The function return type will also be inferred as `any`, making the function less useful when calling it or receiving its result.

Example code where using any leads to problems:
```typescript
function getLengthBad(array: any) {
    // Don't do this.
    return array.length;
}

getLengthBad(/123/);  // No error, returns undefined
```
x??

---

#### Using Specific Types Instead of any[]
Background context: When you expect a specific type such as an array or object, it's better to use the appropriate specific type instead of `any`. This improves code safety and readability.

:p What are the benefits of using more specific types like `any[]` over `any` in TypeScript?
??x
Using more specific types like `any[]` offers several advantages:
- Type checking is performed on array operations, ensuring that only valid array methods can be called.
- The function return type is inferred as `number`, making it clearer what the function returns.
- Call checks ensure that parameters are of the expected type.

Example code comparing using `any` and `any[]`:
```typescript
function getLengthBad(array: any) {
    // Don't do this.
    return array.length;
}

function getLength(array: any[]) {
    // This is better
    return array.length;
}

getLength(/123/);  // Error: Argument of type 'RegExp' is not assignable to parameter of type 'any[]'.
```
x??

---

#### Using {[key: string]: any} for Dynamic Objects
Background context: When you expect an object that might have varying keys, using `{[key: string]: any}` or `Record<string, any>` allows for flexibility in the object's structure.

:p What is the difference between using a specific object type and `{[key: string]: any}` when expecting dynamic objects?
??x
Using a specific object type like `object` vs. `{[key: string]: any}` (or `Record<string, any>`) has different implications:
- Using `object` implies that the object can have properties of any non-primitive types.
- You can still iterate over keys but cannot access values directly without type assertion.

Example code showing iteration with specific and dynamic object types:
```typescript
function hasAKeyThatEndsWithZ(o: Record<string, any>) {
    for (const key in o) {
        if (key.endsWith('z')) {
            console.log(key, o[key]);
            return true;
        }
    }
    return false;
}

// Using `object` type:
function hasAKeyThatEndsWithZObject(o: object) {
    for (const key in o) {  // Error: Element implicitly has an 'any' type
        if (key.endsWith('z')) {
            console.log(key, o[key]);
            return true;
        }
    }
    return false;
}
```
x??

---

#### Using void with Functions
Background context: When you expect a function that returns `void` or any specific function signature, it's better to define the type explicitly instead of using `any`.

:p What are some ways to handle different function signatures when not caring about their return values?
??x
Handling function signatures explicitly can help maintain type safety and clarity:
- Use `() => any` for functions with no parameters.
- Use `(arg: any) => any` for functions with one parameter.
- Use `(...args: any[]) => any` for functions with multiple parameters.

Example code showing different function types:
```typescript
type Fn0 = () => any;  // Any function callable with no params
type Fn1 = (arg: any) => any;  // With one param
type FnN = (...args: any[]) => any;  // With any number of params, same as "Function"
```
x??

---

