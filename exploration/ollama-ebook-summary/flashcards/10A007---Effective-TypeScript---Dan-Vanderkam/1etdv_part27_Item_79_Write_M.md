# Flashcards: 10A007---Effective-TypeScript---Dan-Vanderkam_processed (Part 27)

**Starting Chapter:** Item 79 Write Modern JavaScript

---

#### Background on Modernizing JavaScript to TypeScript

Modernizing JavaScript to TypeScript is a process aimed at improving code quality, reducing bugs, and leveraging type safety. The transition involves converting existing JavaScript codebases into TypeScript, a strongly typed superset of JavaScript.

Relevant background information:
- A 2017 study found that 15% of bugs fixed in JavaScript projects on GitHub could have been prevented with TypeScript.
- A survey at Airbnb over six months indicated that 38% of postmortems could have been prevented by using TypeScript.

:p What is the primary benefit of migrating a large JavaScript project to TypeScript according to the study?
??x
The primary benefit, as highlighted in the study, is reducing bugs. The 2017 study showed that 15% of bugs fixed in JavaScript projects on GitHub could have been prevented with TypeScript.
x??

---

#### Importance of Gradual Migration

Gradual migration is crucial for large projects to avoid overwhelming changes and maintain productivity during the transition process.

:p Why is gradual migration important when migrating a large project?
??x
Gradual migration is important because it helps manage the complexity of converting a large codebase, reducing the risk of introducing new bugs. It allows developers to adapt gradually without overwhelming them.
x??

---

#### Tracking Progress During Migration

Tracking progress and maintaining momentum are essential during long migrations to ensure no backsliding.

:p Why is tracking your progress critical during a TypeScript migration?
??x
Tracking progress is critical because it helps maintain momentum and ensures that the project remains on schedule. It allows you to identify issues early and address them, preventing backsliding.
x??

---

#### Benefits of Modern JavaScript for Transition

Adopting modern JavaScript features can simplify transitioning to TypeScript due to its compatibility with the latest JavaScript.

:p Why is adopting modern JavaScript a good first step before migrating to TypeScript?
??x
Adopting modern JavaScript is beneficial because it prepares the codebase for TypeScript. TypeScript is designed to work well with modern JavaScript, so updating to newer features makes the transition smoother.
x??

---

#### Example of Legacy Code

The dygraphs charting library serves as an example of legacy code that needs modernization and migration.

:p What does the author use dygraphs as an example in this context?
??x
The author uses dygraphs as an example to illustrate how a legacy JavaScript library can benefit from modernization and migration to TypeScript.
x??

---

#### Writing Modern JavaScript

Writing modern JavaScript, defined here as everything introduced in ES2015 (ES6) and after, is essential for preparing your codebase for TypeScript.

:p What does the author define as "modern JavaScript"?
??x
The author defines "modern JavaScript" as everything introduced in ECMAScript 2015 (ES6) and later versions.
x??

---

#### Compiling with TypeScript

TypeScript compiles to any version of JavaScript, from ES2015 back to older ES5. This feature allows you to use the latest JavaScript features while maintaining compatibility.

:p How does TypeScript handle compiling modern JavaScript code?
??x
TypeScript handles compiling modern JavaScript by using its transpiler capability, which can convert new JavaScript features into older, more widely supported JavaScript versions.
x??

---

#### Transitioning with TypeScript

Transitioning an existing JavaScript project to TypeScript involves gradually updating the codebase and leveraging TypeScript's strong typing.

:p What is the process of transitioning a large JavaScript project to TypeScript?
??x
The process involves gradually updating the codebase by writing parts in TypeScript, compiling it to older JavaScript versions using TypeScript's transpiler capabilities, and ensuring compatibility with existing systems.
x??

---

#### ECMAScript Modules
ECMAScript modules provide a standard way to break JavaScript code into separate modules. Prior to ES2015, there was no standard module system, and developers had to use various methods like manual concatenation, Node.js require statements, AMD define callbacks, or TypeScript's own module system.

:p What is the advantage of using ECMAScript modules over other module systems?
??x
ECMAScript modules offer a standardized approach that simplifies module management in JavaScript projects. They improve code organization and maintainability by allowing developers to split large scripts into smaller, reusable pieces. This standardization also enhances compatibility across different environments.

```typescript
// Example of ES Module in TypeScript
import * as b from './b';
console.log(b.name);

// Corresponding file (./b.ts)
export const name = 'Module B';
```
x??

---

#### Classes vs Prototypes
JavaScript traditionally uses a prototype-based object model, but many developers prefer the class-based approach due to its simplicity and readability. ES2015 introduced the `class` keyword, making it easier for developers to write code that is more familiar to those coming from other languages like C++ or Java.

:p How does using classes in JavaScript differ from using prototypes?
??x
Using classes in JavaScript provides a clearer structure and syntax compared to traditional prototype-based inheritance. Classes encapsulate properties and methods within their scope, making the code easier to understand and maintain. Additionally, TypeScript can better handle class definitions compared to prototype chains.

```typescript
// Prototype version
function Person(first, last) {
  this.first = first;
  this.last = last;
}
Person.prototype.getName = function() {
  return this.first + ' ' + this.last;
}

const marie = new Person('Marie', 'Curie');
console.log(marie.getName());

// Class-based version
class Person {
  constructor(first, last) {
    this.first = first;
    this.last = last;
  }
  
  getName() {
    return this.first + ' ' + this.last;
  }
}

const marie = new Person('Marie', 'Curie');
console.log(marie.getName());
```
x??

---

#### Migration to ES Modules
If your project is still using a single file or concatenation for managing modules, it's time to transition to ECMAScript modules. This change can simplify your build process and improve the maintainability of your codebase.

:p Why should you use ES modules in TypeScript projects?
??x
Using ES modules in TypeScript simplifies module management by providing a standardized approach that enhances compatibility across different environments. It allows for better organization of code, easier reuse, and improved readability. Transitioning to ES modules can also make the migration process smoother when moving from one JavaScript project setup to another.

```typescript
// CommonJS version
const b = require('./b');
console.log(b.name);

// ES Module version in TypeScript
import * as b from './b';
console.log(b.name);
```
x??

---

#### TypeScript with Classes
TypeScript can handle class definitions more effectively than it does prototype-based inheritance. By adopting classes, you can leverage the strong typing and static analysis features of TypeScript to ensure your code is correct.

:p How does TypeScript handle class definitions?
??x
TypeScript understands class-based constructs better than prototype chains. It provides type safety and enables easier refactoring through tools like the `--emitDecoratorFromClassExpression` flag in the compiler options. This ensures that your classes are compiled correctly into JavaScript, maintaining their structure and functionality.

```typescript
// Example of a class in TypeScript
class Person {
  constructor(first: string, last: string) {
    this.first = first;
    this.last = last;
  }
  
  getName(): string {
    return `${this.first}${this.last}`;
  }
}

const marie = new Person('Marie', 'Curie');
console.log(marie.getName());
```
x??

---

---
#### Use let and const Instead of var
Background context: The `var` keyword in JavaScript has some quirky scoping rules, leading to potential bugs. On the other hand, `let` and `const` provide block scope, making code more predictable and easier to reason about.

:p What are the differences between `var`, `let`, and `const` in terms of scope?
??x
The key difference lies in their scoping rules:
- `var`: Function-scoped (function-level variables) or globally scoped.
- `let` and `const`: Block-scoped (block, statement, or expression level).

For example:
```javascript
for (var i = 0; i < 5; i++) {
    console.log(i); // Logs from 0 to 4 (hoisted but not initialized yet)
}

console.log(i); // Logs 5 because `i` is still in scope

// Using let and const avoids this issue:
for (let j = 0; j < 5; j++) {
    console.log(j); // Logs from 0 to 4
}
```

Using block scoping helps prevent accidental variable hoisting, which can lead to bugs.
x??

---
#### Use for-of or Array Methods Instead of C-Style For Loops
Background context: The traditional C-style `for(;;)` loop introduces an index variable that might not be needed. It also becomes tricky when working with iterators like those found in ES2015.

:p Why is using `for-of` preferred over the traditional C-style for loops?
??x
The `for-of` loop simplifies iteration by directly accessing each value from the iterable without needing an index variable. This makes the code more readable and easier to maintain, especially when working with arrays or other iterables like strings.

For example:
```javascript
// Traditional C-style for loop
let array = [1, 2, 3];
for (var i = 0; i < array.length; i++) {
    console.log(array[i]);
}

// Using `for-of` loop
array = [1, 2, 3];
for (const value of array) {
    console.log(value);
}
```

The `for-of` loop avoids the need to manually increment and decrement an index variable, making the code cleaner.
x??

---
#### Use Async and Await for Asynchronous Functions Instead of Callbacks or Raw Promises
Background context: JavaScript's asynchronous programming can be complex with callbacks or raw promises. The `async` and `await` keywords simplify this by providing a more readable syntax.

:p How do `async` and `await` make asynchronous functions easier to understand?
??x
The `async` keyword transforms a function into an asynchronous one, which returns a promise automatically. The `await` keyword can be used inside the async function to wait for a promise to resolve or reject, making it more readable.

Example:
```javascript
// Using callbacks and promises
function getUserData(userId) {
    return fetch(`https://api.example.com/users/${userId}`)
        .then(response => response.json())
        .then(data => console.log(data));
}

getUserData(123);

// Using async/await
async function getUserDataAsync(userId) {
    const response = await fetch(`https://api.example.com/users/${userId}`);
    const data = await response.json();
    console.log(data);
}

getUserDataAsync(123);
```

Using `async` and `await` makes the code more synchronous in nature, reducing callback hell and making it easier to understand.
x??

---
#### Use Arrow Functions for Better Readability
Background context: Arrow functions provide a more concise syntax compared to traditional function expressions. Additionally, they preserve the lexical `this`, which can be particularly useful when dealing with objects.

:p Why should arrow functions be preferred over traditional function expressions?
??x
Arrow functions are shorter and often more readable due to their syntax. They also bind `this` lexically from the surrounding context, avoiding issues related to `this` in methods or callbacks.

Example:
```javascript
// Traditional function expression
function foo(param) {
    console.log(this.param);
}

// Arrow function
const foo = (param) => {
    console.log(this.param); // `this` is bound to the lexical scope where it's defined
};
```

Using arrow functions can make your code cleaner, and they automatically bind `this`, which simplifies handling in callbacks or within objects.
x??

---
#### Use Default Parameter Values for Function Parameters
Background context: In modern JavaScript, you can provide default values directly in function parameter declarations. This is more concise and allows the type of a parameter to be inferred from its default value.

:p How do you declare functions with default parameters?
??x
You can define a function with default parameters by providing default values inside the parameter list:

Example:
```javascript
function foo(param = 123) {
    console.log(param);
}

// Calling without arguments uses the default value
foo(); // Logs 123

// Calling with an argument overrides the default
foo(456); // Logs 456
```

This makes function calls more explicit and easier to understand, as well as allowing TypeScript to infer the type of `param` from its default value.
x??

---
#### Use Compact Object Literals and Destructuring Assignment for Consistent Naming
Background context: Instead of writing `{ x: x }`, you can simply write `{ x }`. This is more concise and encourages consistent naming. Similarly, destructuring assignment allows you to unpack values from an array or object in a clear way.

:p How do compact object literals and destructuring work?
??x
Compact object literals allow you to omit the `:` when the key and value are identical:

Example:
```javascript
const x = 10;
// Traditional: { x: x }
// Compact: { x }

// Destructuring allows you to unpack values from an array or object.
let pair = [42, "answer"];
let [x, y] = pair; // Unpacks the first element into `x` and second into `y`
```

Destructuring is particularly useful when working with objects or arrays in TypeScript.

Example:
```typescript
type Person = { name: string, age: number };
const person: Person = { name: "Alice", age: 30 };

// Destructuring object properties
const { name } = person; // `name` will be "Alice"
```

These techniques improve code readability and consistency.
x??

---
#### Use Map and Set for Associative Arrays
Background context: The ES2015 `Map` and `Set` types provide a more robust way to handle associative arrays compared to objects in JavaScript. They avoid issues stemming from the conflation of objects and associative arrays.

:p Why should you use `Map` and `Set` over plain objects?
??x
Using `Map` and `Set` provides several benefits:
- `Map` allows keys of any type, including non-primitive values like objects.
- `Set` ensures that each value is unique.
- They are designed to handle iterable data more efficiently.

Example with `Map`:
```javascript
const map = new Map();
map.set("key", "value");
console.log(map.get("key")); // Logs "value"
```

Example with `Set`:
```javascript
const set = new Set();
set.add(1);
set.add(2);
console.log(set.has(1)); // Logs true
```

These data structures provide a safer and more reliable way to manage complex associative arrays.
x??

---
#### Use Optional Chaining for Nullable Values
Background context: Optional chaining (`?.`) allows you to safely access deeply nested properties without the risk of `TypeError` if any intermediate value is null or undefined.

:p What does optional chaining do?
??x
Optional chaining checks each property in a chain before accessing it, allowing you to skip over null or undefined values. This avoids errors and makes your code more robust.

Example:
```javascript
const obj = { prop: "value" };
console.log(obj.prop); // Logs "value"

// Using optional chaining
const value1 = obj?.prop; // Works fine if `obj` is not null/undefined
const value2 = obj?.deeply?.nested?.property; // Returns undefined if any part of the chain is null or undefined

if (obj?.deepProperty) {
    console.log("Accessed deeply without error");
}
```

Optional chaining helps prevent runtime errors by short-circuiting at the first `null` or `undefined`.
x??

---
#### Use Nullish Coalescing for Default Values
Background context: The nullish coalescing operator (`??`) provides a way to return a fallback value when dealing with `null` or `undefined`. This avoids conflating truthy and nullish values, which can lead to bugs.

:p How does the nullish coalescing operator work?
??x
The `??` operator returns the right-hand operand if the left-hand side is either `null` or `undefined`, otherwise it returns the left-hand value. This helps avoid issues where a falsy value (like 0) could be treated as false.

Example:
```javascript
const x = null;
console.log(x ?? "default"); // Logs "default"

const y = 0; // Falsy but not null or undefined
console.log(y ?? "default"); // Logs "0" because it's truthy

const z = null;
console.log(z || "default"); // Also logs "default"
```

Using `??` is safer and more explicit about handling `null` and `undefined`.
x??

---

#### Use @ts-check and JSDoc to Experiment with TypeScript

Background context: This concept introduces the use of `@ts-check` directive and JSDoc comments for type checking in JavaScript files before converting them fully to TypeScript. It helps identify issues early on, making the migration process smoother.

:p What is the purpose of using `@ts-check`?
??x
The purpose of using `@ts-check` is to enable a form of type checking in JavaScript code without fully converting it to TypeScript. This helps developers catch potential type-related errors and gain familiarity with TypeScript's type system before making a full migration.

```typescript
// Example: Using @ts-check for type checking
// @ts-check
const person = {first: 'Grace', last: 'Hopper'};
2 * person.first; // Error: The right-hand side of an arithmetic operation must be of type 'any', 'number', or an enum.
```
x??

---

#### Undeclared Globals

Background context: Identifies symbols that are not declared in the code but used, requiring proper declaration to resolve errors. This is crucial for maintaining correct and consistent type definitions.

:p What does `@ts-check` report when encountering undeclared global variables?
??x
When using `@ts-check`, it reports errors related to undeclared global variables. These can be either local symbols that need `let` or `const` declarations, or "ambient" symbols defined elsewhere (e.g., in HTML `<script>` tags) that require a type declaration file.

```typescript
// Example: Error due to undeclared global variable 'user'
// @ts-check
console.log(user.firstName); // Error: Cannot find name 'user'
```
x??

---

#### Unknown Libraries

Background context: When using third-party libraries, TypeScript needs specific type declarations. `@ts-check` helps identify these missing declarations and provides guidance on how to resolve them.

:p What does `@ts-check` report when encountering an unknown library function?
??x
When using `@ts-check`, it reports errors for functions or properties that are not recognized by the compiler due to missing type definitions. For example, using a jQuery method like `.style()` without the appropriate types can result in errors.

```typescript
// Example: Error due to missing TypeScript declaration for jQuery
// @ts-check
$('#graph').style({'width': '100px', 'height': '100px'}); // Error: Cannot find name '$'
```
x??

---

#### DOM Issues

Background context: When working with the Document Object Model (DOM) in web applications, TypeScript may report errors due to type mismatches between methods and the `HTMLElement` interface. This requires careful handling of element types.

:p What error might `@ts-check` generate when manipulating DOM elements?
??x
`@ts-check` can generate an error related to property access on `HTMLElement`. For example, attempting to set `.value` on a generic `HTMLElement` will result in a type error because only specific input types have this property.

```typescript
// Example: Error due to incorrect property access on HTMLElement
// @ts-check
const ageEl = document.getElementById('age');
ageEl.value = '12'; // Error: Property 'value' does not exist on type 'HTMLElement'
```
x??

---

#### Inaccurate JSDoc

Background context: Existing JSDoc comments in the codebase can be misleading or incomplete. `@ts-check` checks these annotations for accuracy and enforces strict typing, leading to better code quality.

:p What issues might `@ts-check` identify with inaccurate JSDoc?
??x
`@ts-check` may identify two main types of inaccuracies: (1) Mismatches between the actual implementation and documented return or parameter types; (2) Incorrect assumptions about DOM methods or properties. These issues need to be addressed for smooth TypeScript integration.

```typescript
// Example: Error due to inaccurate JSDoc comments
// @ts-check
/** 
 * Gets the size (in pixels) of an element.
 * @param {Node} el The element
 * @return {{w: number, h: number}} The size
 */
function getSize(el) {
    const bounds = el.getBoundingClientRect();
    return {width: bounds.width, height: bounds.height};
}
```
x??

---

#### Quick Fixes in JSDoc

Background context: `@ts-check` can suggest quick fixes to add type annotations based on usage. These suggestions help developers incrementally improve their codebase while working with JavaScript files.

:p How does the TypeScript language service provide quick fixes?
??x
The TypeScript language service provides quick fixes for adding type annotations based on the actual usage of variables or parameters in the code. For example, it can infer types from function arguments and suggest JSDoc comments to document these types accurately.

```typescript
// Example: Quick fix suggestion by TypeScript language service
function double(val) {
    return 2 * val;
}
```
x??

---

#### Structural Typing Mismatch

Background context: While `@ts-check` can suggest type annotations, the results might not always be ideal. Structural typing in JavaScript allows for flexible object shapes, but `@ts-check` may sometimes infer overly generic types.

:p What is a potential drawback of using quick fixes with `@ts-check`?
??x
A potential drawback of using quick fixes with `@ts-check` is that they might result in overly generic or structurally mismatched type annotations. For instance, inferring `{files: {forEach: (arg0: (file: any) => Promise<void>) => void;}}` for an array of files may not accurately reflect the intended structure.

```typescript
// Example: Incorrect quick fix suggestion by TypeScript language service
function loadData(data) {
    data.files.forEach(async file => {
        // ...
    });
}
```
x??

---

#### Enable Type Checking Without Converting to TypeScript
Background context: To enable type checking without converting a JavaScript file to TypeScript, you can add `// @ts-check` at the top of your JavaScript file. This allows for basic type checking using JSDoc annotations and third-party type definitions.

:p How do you enable type checking in a JavaScript file without converting it to TypeScript?
??x
By adding `// @ts-check` at the top of the JavaScript file, you can use JSDoc-style type comments and have better type inference. This allows for basic static typing checks while keeping your code as JavaScript.
```javascript
// @ts-check

function add(a: number, b: number): number {
    return a + b;
}
```
x??

---

#### Declare Globals and Add Type Declarations for Third-Party Libraries
Background context: When working with third-party libraries in TypeScript, you often need to declare global types or use existing type definitions. This can be done by creating custom `.d.ts` files or using pre-existing declaration files from npm.

:p How do you add type declarations for a third-party library in TypeScript?
??x
You can create a custom `index.d.ts` file and declare the types, interfaces, and functions of the third-party library. Alternatively, you can install an existing `.d.ts` file via npm.

For example:
```typescript
// index.d.ts

declare global {
    interface Window {
        myLibrary: any;
    }
}

window.myLibrary = { // initialization code here };
```
x??

---

#### Use `allowJs` to Mix TypeScript and JavaScript
Background context: The `allowJs` compiler option in TypeScript allows you to work with both TypeScript and JavaScript files within the same project. This is particularly useful when gradually transitioning from JavaScript to TypeScript.

:p What does the `allowJs` compiler option enable?
??x
The `allowJs` compiler option enables the coexistence of TypeScript and JavaScript files in a single project, allowing for gradual migration. When using `allowJs`, you can import JavaScript modules into TypeScript code and vice versa, with limited type checking unless you use `@ts-check`.

Example:
```typescript
// main.ts

import { myFunction } from './myScript.js';

console.log(myFunction());
```

```javascript
// myScript.js
export function myFunction() {
    return "Hello, TypeScript!";
}
```
x??

---

#### Transition Gradually to TypeScript with `allowJs`
Background context: To gradually transition a JavaScript project to TypeScript, you can use the `allowJs` compiler option. This allows you to introduce TypeScript into your build chain and run tests while converting individual modules.

:p How do you start introducing TypeScript into an existing JavaScript project?
??x
You can enable `allowJs` in your `tsconfig.json` to allow JavaScript files to coexist with TypeScript. You can then use tools like Jest or Webpack to integrate TypeScript sources, ensuring that your tests run correctly during the transition.

Example `tsconfig.json`:
```json
{
    "compilerOptions": {
        "allowJs": true,
        "outDir": "./dist",
        "rootDir": "./src"
    },
    "include": ["src/**/*.ts", "src/**/*.js"]
}
```

Example `jest.config.js` for Jest integration:
```javascript
module.exports = {
    transform: { '^.+\\.tsx?$': 'ts-jest' }
};
```
x??

---

#### Configure Webpack with TypeScript Support
Background context: To use TypeScript in a Webpack project, you need to configure Webpack and install necessary loaders like `ts-loader`.

:p How do you configure Webpack to support TypeScript?
??x
You can install `ts-loader` via npm and then configure your `webpack.config.js` to include rules for `.tsx` and `.ts` files.

Example `webpack.config.js`:
```javascript
module.exports = {
    module: {
        rules: [
            { test: /\.tsx?$/, use: 'ts-loader', exclude: /node_modules/ }
        ]
    },
    resolve: {
        extensions: ['.tsx', '.ts', '.js']
    }
};
```
x??

---

#### Configure Jest for TypeScript
Background context: To run tests in a TypeScript project using Jest, you need to configure Jest to handle `.ts` and `.tsx` files.

:p How do you configure Jest to work with TypeScript?
??x
You can install `ts-jest` via npm and then configure Jest's transform options to include TypeScript sources.

Example `jest.config.js`:
```javascript
module.exports = {
    transform: { '^.+\\.tsx?$': 'ts-jest' }
};
```
x??

---

#### Use `ts-node` for Node.js Integration
Background context: To run a TypeScript file with Node.js, you can use the `ts-node` package as a drop-in replacement or configure it to understand TypeScript.

:p How do you use `ts-node` to run TypeScript files?
??x
You can install `ts-node` via npm and then register it with Node.js so that it understands TypeScript syntax. Alternatively, you can replace the Node command with `ts-node`.

Example:
```bash
$ node -r ts-node/register main.ts
```
x??

---

#### Generate Pure JavaScript Sources with `outDir`
Background context: When using TypeScript, you can specify an output directory (`outDir`) that generates pure JavaScript sources. This allows your existing build chain to process the generated JavaScript files.

:p How do you configure TypeScript to generate JavaScript files in a specific directory?
??x
You can set the `outDir` option in your `tsconfig.json` to specify the output directory for generated JavaScript files.

Example `tsconfig.json`:
```json
{
    "compilerOptions": {
        "outDir": "./dist"
    }
}
```
x??

---

