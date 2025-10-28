# Flashcards: 10A007---Effective-TypeScript---Dan-Vanderkam_processed (Part 9)

**Starting Chapter:** Item 24 Understand How Context Is Used in Type Inference

---

#### Context in Type Inference

TypeScript uses context to infer types, which can sometimes lead to unexpected results. Understanding how context is used helps in identifying and resolving these issues.

:p How does TypeScript use context for type inference?
??x
TypeScript infers types based on both the values assigned and the context within which they are used. This means that the same value may have a different inferred type depending on where it's being used.
For example, consider:
```typescript
function setLanguage(language: string) { /* ... */ }
setLanguage('JavaScript');  // OK
```
In this case, 'JavaScript' is inferred as `string` because of the function context.

But when you factor out a variable like so:
```typescript
let language = 'JavaScript'; 
setLanguage(language); // Error: Argument of type 'string' is not assignable to parameter of type 'Language'
```
Here, `language` is inferred as `string`, even though it should be inferred as `Language`.

This discrepancy arises because TypeScript infers the type at the time of assignment without considering the function's context.
x??

---

#### Tuple Types and Context

Tuple types can also lead to issues when a value is factored out from its original context.

:p How does the context of a tuple affect type inference in TypeScript?
??x
In TypeScript, tuples are used to represent ordered collections with fixed element types. When a tuple value is factored out into a local variable or constant, the type inference can be different from when it's inline.

Consider this example:
```typescript
function panTo(where: [number, number]) { /* ... */ }
panTo([10, 20]);  // OK

const loc = [10, 20];
//    ^? const loc: number[]
panTo(loc); // Error: Argument of type 'number[]' is not assignable to parameter of type '[number, number]'
```
In the first instance, `[10, 20]` is inferred as a tuple. However, when assigned to `loc`, it becomes an array (`number[]`) due to context loss.

To fix this:
- Use a type annotation: 
```typescript
const loc: [number, number] = [10, 20];
```
- Or use const assertion: 
```typescript
const loc = [10, 20] as const; // const loc: readonly [10, 20]
```
x??

---

#### Using `as const` for Const Context

The `as const` operator can help maintain the context and type precision of a value.

:p What does the `const` assertion (`as const`) do in TypeScript?
??x
The `const` assertion in TypeScript allows you to make the compiler treat an object literal as read-only, thereby preserving its structure and types. 

For example:
```typescript
const loc = [10, 20] as const; // const loc: readonly [10, 20]
```
Here, `loc` is inferred with a precise tuple type `readonly [10, 20]`.

This can be particularly useful to prevent accidental modifications and maintain the original types of properties.
x??

---

#### String Literal Types and Context

String literal types can also cause issues when values are factored out.

:p How does context affect the inference of string literals in TypeScript?
??x
When using string literal types, context is crucial for accurate type inference. If a value is factored out from its original context, it may lose its precise type information and be inferred as just `string`.

For instance:
```typescript
type Language = 'JavaScript' | 'TypeScript' | 'Python';

function setLanguage(language: Language) { /* ... */ }
setLanguage('JavaScript'); // OK

let language = 'JavaScript'; 
//          ~~~~~~~~ Argument of type 'string' is not assignable to parameter of type 'Language'
```
Here, `language` is inferred as a general `string`, but it should be of type `Language`.

To solve this:
- Add a type annotation: 
```typescript
let language: Language = 'JavaScript'; setLanguage(language); // OK
```
- Or use const assertion: 
```typescript
const language = 'JavaScript' as const; // const language: "JavaScript"
setLanguage(language); // OK
```
x??

---

#### Objects and Context

Objects containing string literals or tuples can also lose their precise types when factored out.

:p How does context affect the inference of object properties in TypeScript?
??x
When an object contains string literals, tuple elements, or other complex structures, factoring it into a local variable can lead to loss of type precision. 

For example:
```typescript
type Language = 'JavaScript' | 'TypeScript' | 'Python';

interface GovernedLanguage {
    language: Language;
    organization: string;
}

function complain(language: GovernedLanguage) { /* ... */ }
complain({ language: 'TypeScript', organization: 'Microsoft' }); // OK

const ts = { 
    language: 'TypeScript', 
    organization: 'Microsoft'
};
//       ~~ Argument of type '{ language: string; organization: string; }' is not assignable to parameter of type 'GovernedLanguage'.
```
Here, `ts` loses its precise type and `language` is inferred as a general `string`.

To fix this:
- Add a type annotation: 
```typescript
const ts: GovernedLanguage = { language: 'TypeScript', organization: 'Microsoft' };
```
x??

---

#### Callbacks and Context

When passing callbacks, context helps in inferring the parameter types correctly.

:p How does context affect the inference of function parameters in TypeScript?
??x
In TypeScript, when you pass a callback to another function, the type inference is based on the expected signature of the callback. If the callback is factored out into a constant without proper type annotations, it can lead to implicit `any` types.

For example:
```typescript
function callWithRandomNumbers(fn: (n1: number, n2: number) => void) {
    fn(Math.random(), Math.random());
}

callWithRandomNumbers((a, b) => { // Parameter 'a' implicitly has an 'any' type 
    console.log(a + b);            // Parameter 'b' implicitly has an 'any' type
});
```
Here, `a` and `b` are inferred as `any`.

To fix this:
- Add explicit type annotations: 
```typescript
const fn = (a: number, b: number) => {
    console.log(a + b);
};

callWithRandomNumbers(fn);
```
Or use a type declaration if the function is only used in one place.

x??

---

#### Evolving Types in TypeScript
Evolving types are a unique behavior in TypeScript where a variable's type can change based on assignments or operations performed during runtime. This is different from narrowing, which involves refining a type to be more specific within conditional checks.

:p How does evolving type work in TypeScript?
??x
In TypeScript, when a variable like `nums` is declared with an array but initialized as an empty array (`const nums = []`), its initial type is `any[]`. However, once elements of the correct type (e.g., numbers) are pushed into it, TypeScript infers the type to be `number[]`.

```typescript
function range(start: number, limit: number): number[] {
    const nums = [];
    for (let i = start; i < limit; i++) {
        nums.push(i);
    }
    return nums;
}
```
Here, even though `nums` is initially an empty array (`any[]`), after pushing numbers into it, the type evolves to `number[]`.

The key point is that this behavior only happens when values are written to a variable. If you try to read from such a variable before it's assigned anything meaningful, TypeScript will infer its type as `any`.

```typescript
let value;
// ^? let value: any
if (Math.random() < 0.5) {
    value = /hello/; // ^? let value: RegExp
} else {
    value = 12;     // ^? let value: number
}
value            // ^? let value: number | RegExp
```
x??

---

#### Initializing Variables with Evolving Types
When a variable is initialized with an empty array or `null`, its type can evolve to become more specific as values of the correct type are assigned.

:p How does TypeScript determine the evolving type of a variable?
??x
TypeScript infers the initial type based on the initialization. If a variable is declared and initialized with an empty array (`const nums = []`), it starts with a type that allows any elements (i.e., `any[]`). As elements are added to the array, TypeScript narrows down the type based on the values pushed into it.

```typescript
let result = [];
// ^? let result: any[]
result.push('a');
// ^? let result: string[]
result.push(1);
// ^? let result: (string | number)[]
```
Similarly, if a variable is initialized to `null` or `undefined`, its type can evolve based on the values assigned during runtime.

```typescript
let value = null;
// ^? let value: any
try {
    value = doSomethingRiskyAndReturnANumber();
    // ^? let value: number
} catch (e) {
    console.warn('alas.');
}
value            // ^? let value: number | null
```
x??

---

#### Evolving Types vs. Narrowing
Evolving types and narrowing are related but distinct concepts in TypeScript.

:p What is the difference between evolving types and narrowing?
??x
Evolving types refer to a variable's type changing during runtime as values of specific types are assigned or pushed into it. For example, if `nums` starts as an empty array (`any[]`), pushing numbers into it makes its inferred type evolve to `number[]`.

Narrowing, on the other hand, is about refining a variable's type within conditional checks. For instance:

```typescript
let value: any;
if (typeof value === 'string') {
    // Here TypeScript knows `value` must be a string
}
```
In this case, TypeScript narrows down the type of `value` to be `string`, but only in the context of the conditional block.

Evolving types are more about runtime changes and don't have fixed points where the type is determined; they keep evolving as the program runs. Narrowing, however, happens at specific points (like within an `if` statement) and is a way to make TypeScript aware that certain variables cannot hold values of other types in certain contexts.

```typescript
let value: any;
if (typeof value === 'string') {
    // ^? let value: string
}
```
x??

---

#### Implications for Function Calls with Evolving Types
Function calls can sometimes affect the type inference of evolving variables, especially when using array methods like `forEach`.

:p How does function call affect evolving types in an array?
??x
When you use a method like `forEach` on an evolving array and map its return value to another variable, TypeScript might not be able to infer the correct type due to the initial empty state of the array.

Consider this example:

```typescript
function makeSquares(start: number, limit: number): number[] {
    const nums = [];
    // ^~~~ Variable 'nums' implicitly has type 'any[]'
    range(start, limit).forEach(i => {
        nums.push(i * i);
    });
    return nums;  // ^~~~ Variable 'nums' implicitly has an 'any[]' type
}
```

Even though `range` returns a number array, the initial state of `nums` as an empty array (`any[]`) means TypeScript cannot accurately infer its final type within the function scope. To avoid this issue, it's better to use the built-in array method `map`, which can handle the transformation in one go.

```typescript
function makeSquares(start: number, limit: number): number[] {
    return range(start, limit).map(i => i * i);
}
```
x??

---

#### Evolving Types and Functional Constructs
Background context: The passage explains how using functional constructs like `map`, `reduce`, etc., can help types flow through your TypeScript code, reducing the need for explicit type annotations. It mentions that these constructs come with their own advantages but also caveats related to incorrect use or overreliance on them.
If applicable, add code examples with explanations:
```typescript
const csvData = "...");
const rawRows = csvData.split(' ');
const headers = rawRows[0].split(',');
const rows = rawRows.slice(1)
  .map((rowStr) => rowStr.split(",").reduce((row, val, i) => ((row[headers[i]] = val), row), {}));
```
:p What are evolving types in TypeScript?
??x
Evolving types allow the type system to infer the structure of objects or arrays as you build them. For example, when using `map` or `reduce`, the initial empty object `{}` can evolve into a specific shape based on the keys and values being assigned.
x??

---

#### Using Lodash for Type Safety
Background context: The passage highlights how using libraries like Lodash can improve type safety in TypeScript by ensuring that types flow through functional constructs. It provides examples of how vanilla JavaScript approaches might require explicit type annotations, whereas Lodash solutions pass the type checker without modification.
If applicable, add code examples with explanations:
```typescript
import _ from 'lodash';

const rows = rawRows.slice(1)
  .map(rowStr => _.zipObject(headers, rowStr.split(',')));
```
:p How do functional constructs and libraries like Lodash help in maintaining typesafety?
??x
Functional constructs and libraries like Lodash help maintain typesafety by providing strongly typed methods that ensure the correct type flow. This means you can avoid or reduce the need for explicit type annotations, making your code more concise while keeping it well-typed.
x??

---

#### Example with Hand-rolled Loops
Background context: The text provides an example of parsing CSV data using hand-rolled loops in TypeScript, which require explicit type annotations to work correctly. This contrasts with functional approaches that do not.
If applicable, add code examples with explanations:
```typescript
const rowsImperative = rawRows.slice(1).map(rowStr => {
  const row = {};
  rowStr.split(',').forEach((val, j) => { // Error: No index signature
    row[headers[j]] = val;
  });
  return row;
});
```
:p Why do hand-rolled loops require type annotations in TypeScript?
??x
Hand-rolled loops often need explicit type annotations because the initial object `{}` does not have a predefined shape. TypeScript needs to be informed about the structure of the resulting object, especially when properties are being added dynamically.
x??

---

#### Flat and Concise Array Operations
Background context: The passage demonstrates how using methods like `flat` can make code more concise and type-safe without requiring additional type annotations. This is particularly useful in complex data munging scenarios.
If applicable, add code examples with explanations:
```typescript
const allPlayers = Object.values(rosters).flat(); // No need for a type annotation here
```
:p How does the `flat` method help in reducing type annotations?
??x
The `flat` method helps reduce type annotations by inferring the correct array structure directly. Since it returns an array of the appropriate depth, TypeScript can infer the final type without needing explicit annotations.
x??

---

#### Team Rosters and Best Paid Players
Background context: The example shows how to build a flat list of players from nested data structures using `flat` and how to find the best-paid players on each team. It contrasts traditional loops with functional approaches that require fewer type annotations.
If applicable, add code examples with explanations:
```typescript
const bestPaid = _(allPlayers)
  .groupBy(player => player.team)
  .mapValues(players => _.maxBy(players, p => p.salary))
  .values()
  .sortBy(p => -p.salary)
  .value();
```
:p How does using `flat` and functional constructs simplify the code for building a flat list of players?
??x
Using `flat` simplifies the code by flattening the multidimensional array directly, resulting in a single-dimensional array. This makes it easier to work with and reduces the need for explicit type annotations.
x??

---

#### Using Functional Constructs in Complex Operations
Background context: The text provides an example of how functional constructs can be used in complex operations like filtering and sorting players based on their salaries. It shows that these constructs provide a more concise and readable solution while maintaining typesafety.
If applicable, add code examples with explanations:
```typescript
const teamToPlayers = {};
for (const player of allPlayers) {
  const {team} = player;
  teamToPlayers[team] = teamToPlayers[team] || [];
  teamToPlayers[team].push(player);
}
```
:p How do functional constructs like `groupBy` and `maxBy` help in managing complex data operations?
??x
Functional constructs like `groupBy` and `maxBy` help manage complex data operations by providing clear, readable code that leverages the type system effectively. They ensure types flow correctly, reducing the need for explicit type annotations and making the code easier to understand.
x??

---

