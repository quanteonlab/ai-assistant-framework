# Flashcards: 10A007---Effective-TypeScript---Dan-Vanderkam_processed (Part 28)

**Starting Chapter:** Item 82 Convert Module by Module Up Your Dependency Graph

---

#### AllowJs Compiler Option
Background context: When integrating TypeScript into your project, you might have both JavaScript and TypeScript files. The `allowJs` compiler option allows you to include JavaScript files in your build process alongside TypeScript files.

:p What is the purpose of using the `allowJs` compiler option?
??x
The `allowJs` compiler option enables the TypeScript compiler to handle JavaScript files as well, allowing a mixed environment during the transition from JavaScript to TypeScript. This helps ensure that existing JavaScript codebase can coexist with new TypeScript code without requiring immediate refactoring.
x??

---

#### Starting Module Conversion at Leaves
Background context: When converting your project to TypeScript, it's important to start with modules that have no dependencies on other modules (leaves in the dependency graph) and move up towards the root of the dependency tree.

:p How should you approach module conversion during the transition from JavaScript to TypeScript?
??x
You should begin by converting the leaves of your dependency graph—those modules which do not import any other modules. This ensures that you don't encounter type errors in dependent modules prematurely, making the process more manageable and less error-prone.
x??

---

#### Migrating Third-Party Dependencies
Background context: To leverage TypeScript with third-party libraries, you often need to install corresponding `@types` packages which provide TypeScript definitions for those libraries. This helps in ensuring that types flow through your code seamlessly.

:p How do you handle third-party dependencies when migrating to TypeScript?
??x
For third-party dependencies, it's important to install the appropriate `@types` package. For example, if you are using lodash, you would run:
```bash
npm install --save-dev @types/lodash
```
This ensures that your project has type definitions for these libraries, allowing TypeScript to understand how they should be used and catch potential misuse.
x??

---

#### Adding Type Definitions for External APIs
Background context: When your codebase interacts with external APIs, it's crucial to add type definitions for these interactions. This helps in ensuring that the types flow correctly throughout your application.

:p How do you handle external API calls during TypeScript migration?
??x
You should start by adding type definitions for external API calls early on. For instance, if you have an API call like:
```typescript
async function fetchTable() {
    const response = await fetch('/data');
    if (response.ok) throw new Error('Failed to fetch.');
    return response.json();
}
```
You can refactor it to include a type definition for the expected data structure:
```typescript
interface TabularData {
    columns: string[];
    rows: number[][];
}

async function fetchTable(): Promise<TabularData> {
    const response = await fetch('/data');
    if (response.ok) throw new Error('Failed to fetch.');
    return response.json();
}
```
This ensures that types will flow from all calls to `fetchTable` and helps catch potential misuse of the API.
x??

---

#### Visualizing Dependency Graphs
Background context: Visualizing the dependency graph can help in understanding how modules interact, especially when converting a large codebase. Tools like madge can be used to generate these visualizations.

:p How do you visualize your project's dependency graph?
??x
To visualize your project’s dependency graph, you can use tools like `madge`. For example, running:
```bash
madge --output dot .
```
will generate a `.dot` file that represents the dependency graph of your project. This tool helps in identifying circular dependencies and visualizing how different modules interact.
x??

---

#### Circular Imports in Dependency Graphs
Background context explaining the concept. Circular imports occur when two or more modules depend on each other directly or indirectly, creating a loop that cannot be resolved by the dependency resolver.

:p What are circular imports and how do they affect project dependency graphs?
??x
Circular imports happen when module A depends on module B, and module B also depends on module A. This creates a cycle in the dependency graph that can lead to issues during compilation or runtime, such as undefined symbols or incorrect type checks. To avoid these problems, it's often necessary to refactor code to remove circular dependencies.

For example:
- ModuleA.js imports ModuleB.
- ModuleB.js also imports ModuleA.

This results in a circular import issue, which can be resolved by reorganizing the modules or using proper design patterns like dependency injection.

---
#### Topological Sorting for Dependency Resolution
Background context explaining the concept. A topological sort of a directed acyclic graph (DAG) can help determine an ordering for tasks that have dependencies on each other. In software projects, this is used to determine which files/modules should be compiled first based on their import relationships.

:p How does topological sorting help in managing dependency graphs?
??x
Topological sorting helps ensure that modules are processed and compiled in the correct order, respecting their dependencies. By running a topological sort on the dependency graph of your project, you can identify the proper sequence for compiling or executing tasks to avoid circular imports.

Example:
Consider these two modules:
- ModuleA depends on ModuleB.
- ModuleB depends on ModuleC.

The topologically sorted order might be: ModuleC -> ModuleB -> ModuleA. This ensures that all dependencies are resolved before a module is processed.

```typescript
// Pseudocode for simple topological sort
function topologicalSort(graph) {
    let visited = new Set();
    let result = [];
    
    function visit(node) {
        if (visited.has(node)) return;
        
        for (let neighbor of graph[node]) {
            visit(neighbor);
        }
        
        result.push(node);
        visited.add(node);
    }

    // Traverse all nodes
    for (let node in graph) {
        visit(node);
    }

    // The result is in reverse order, so reverse it
    return result.reverse();
}
```

x??

---
#### Migrating to TypeScript: Adding Types vs Refactoring
Background context explaining the concept. When converting a JavaScript project to TypeScript, the immediate goal should be to add type annotations and convert the codebase without major refactoring efforts until the initial conversion is complete.

:p What are the priorities when migrating an existing JavaScript project to TypeScript?
??x
When migrating an existing JavaScript project to TypeScript, focus on adding types rather than performing large-scale refactors. The primary objective is to ensure that your project starts leveraging type safety and other features of TypeScript without introducing unnecessary changes.

For example:
- Convert variable declarations with `let` or `const` and add explicit types.
- Use interfaces for complex objects.
- Add type annotations where necessary, but avoid making substantial design changes unless absolutely required.

Example:
```typescript
// Before migration (JavaScript)
let greeting: string;
greeting = 'Hello';

// After migration (TypeScript)
let greeting: string = 'Hello';
```

x??

---
#### Undeclared Class Members in TypeScript
Background context explaining the concept. In JavaScript, class members do not need to be declared before use, but in TypeScript, they must be explicitly declared with their types.

:p What is an issue you might encounter when converting a JavaScript class to TypeScript?
??x
When converting a JavaScript class to TypeScript, one common issue is undeclared class members. TypeScript requires that all properties and methods of a class be properly typed before use, even if the class in JavaScript does not have explicit type declarations.

For example:
```typescript
class Greeting {
    constructor(name) {
        this.greeting = 'Hello';  // Error: Property 'greeting' does not exist on type 'Greeting'
        this.name = name;
    }
    greet() {
        return `${this.greeting}${this.name}`;  // Error: Property 'greeting' does not exist
    }
}
```

The quick fix for this is to add the missing types:
```typescript
class Greeting {
    greeting: string;
    name: any;
    
    constructor(name) {
        this.greeting = 'Hello';
        this.name = name;
    }
    greet() {
        return `${this.greeting}${this.name}`;
    }
}
```

x??

---
#### Values with Changing Types in TypeScript
Background context explaining the concept. Sometimes, values may change their type during the execution of a program, leading to issues when converting JavaScript code to TypeScript.

:p What is an example of an issue related to changing types that you might encounter while migrating JavaScript to TypeScript?
??x
When migrating JavaScript to TypeScript, one common issue is dealing with values whose types can change dynamically. TypeScript needs explicit type declarations for all variables and properties at the point they are declared or referenced, which may conflict with how those values behave in runtime.

For example:
```typescript
let value: any = 'Hello';
value = 42; // This works fine in JavaScript but causes issues when trying to use strict types

// In TypeScript, you might need to redeclare the type based on its usage or use `any` if dynamic typing is required.
```

x??

---

#### Migrating to TypeScript
Background context: When migrating from JavaScript (or another language) to TypeScript, developers often encounter issues that were previously implicit. This process can reveal bad design decisions and enforce type safety, making it harder to justify poor designs.
:p What are some common challenges you might face when migrating code to TypeScript?
??x
When migrating code to TypeScript, you may face several challenges such as encountering errors where properties do not exist on an object, losing type safety with JSDoc annotations, or having issues with tests that were implicitly passing due to the dynamic nature of JavaScript. These challenges arise because TypeScript enforces stricter typing and can surface issues that were previously ignored.
```typescript
const state = {};
state.name = 'New York';  // Error: Property 'name' does not exist on type '{}'
state.capital = 'Albany'; // Error: Property 'capital' does not exist on type '{}'

interface State {
    name: string;
    capital: string;
}
const state = {} as State; // Type assertion
state.name = 'New York';
state.capital = 'Albany';

// @ts-check
/**
 * @param {number} num
 */
function double(num) {
    return 2 * num;
}

double('trouble'); // Error: Argument of type 'string' is not assignable to parameter of type 'number'
```
x??

---
#### Fixing Property Type Issues in TypeScript
Background context: When migrating JavaScript objects with dynamically added properties, TypeScript may complain about missing properties. This can be resolved by defining the object's structure using interfaces or by initializing it all at once.
:p How do you fix TypeScript errors related to properties that don't exist on an object?
??x
To fix these issues, you can either initialize the object with all its properties at once or define an interface for the object. Here are both approaches:
1. Initialize the object with all its properties:
```typescript
const state = {
    name: 'New York',
    capital: 'Albany'
};
```
2. Define an interface and type assert the object:
```typescript
interface State {
    name: string;
    capital: string;
}
const state = {} as State; // Type assertion
state.name = 'New York';
state.capital = 'Albany';
```
Using interfaces or type assertions can help avoid errors, but remember to refactor eventually for cleaner code.
x??

---
#### Using Type Assertions in TypeScript
Background context: Sometimes, you might need to use type assertions when the fix is not trivial. However, these should be used sparingly as they bypass static typing checks and can lead to runtime errors.
:p When is it appropriate to use type assertions in TypeScript?
??x
TypeScript allows the use of type assertions in cases where initializing an object with all its properties at once is not feasible. This often happens when you're dealing with dynamic objects or data structures that are hard-coded elsewhere in your application. While using type assertions can help keep your migration process moving, it's important to refactor eventually for better code maintainability.

Example of a situation where a type assertion might be used:
```typescript
interface State {
    name: string;
    capital: string;
}
const state = {} as State; // Type assertion
state.name = 'New York';
state.capital = 'Albany';

// A good practice is to leave a TODO or file a bug to clean this up later.
```
x??

---
#### Maintaining Type Safety with JSDoc and @ts-check
Background context: When using JSDoc comments, they are not enforced in TypeScript. This can lead to issues where types are implicitly `any`, causing type safety concerns.
:p What happens when you convert JavaScript code to TypeScript and use JSDoc annotations?
??x
When converting JavaScript code that uses JSDoc annotations to TypeScript with the `@ts-check` directive, type safety can be compromised because JSDoc comments stop being enforced. This means that types are implicitly set to `any`, leading to potential runtime errors or warnings. 

Example of a problem:
```typescript
// @ts-check
/**
 * @param {number} num
 */
function double(num) {
    return 2 * num;
}

double('trouble'); // No error in TypeScript, but should be an error due to wrong type
```
To maintain type safety, you need to manually add the types using `type` or `interface`. Here's how you can fix it:
```typescript
function double(num: number) {
    return 2 * num;
}

double('trouble'); // Error: Argument of type 'string' is not assignable to parameter of type 'number'
```
Once you've added the types, you should remove them from JSDoc comments to avoid redundancy.
x??

---
#### Migrating Tests Last
Background context: During a TypeScript migration, tests should be the last part of your dependency graph. This ensures that changes in production code do not affect test results, providing reassurance during the transition.
:p Why should you migrate tests last when converting JavaScript code to TypeScript?
??x
Migrating tests last helps maintain consistency and ensures that your tests continue to pass without having been changed. This is important because it provides confidence that the behavior of your application remains intact during the migration process.

Additionally, since production code does not import test files, migrating them later simplifies the transition by reducing dependencies and making the process more straightforward.

Example of a test file:
```typescript
// Test file (should be last to migrate)
describe('double function', () => {
    it('should double the number correctly', () => {
        expect(double(2)).toBe(4);
    });
});
```
By keeping tests as the final part, you ensure that your application's behavior remains consistent during the migration.
x??

#### Starting Migration by Adding @types

Background context: The first step in migrating a project to TypeScript is ensuring that all third-party modules and external API calls have proper type definitions. This often involves adding `@types` packages if they are not already available.

:p How do you start the migration process for third-party modules and external APIs?
??x
The process begins by installing any necessary `@types` packages using npm or yarn. For example:
```sh
npm install --save-dev @types/node
```
or
```sh
yarn add --dev @types/react
```

You should also consider the dependency graph of your project, starting with utility code at the bottom and moving upwards.

??x

---

#### Migrating Own Modules

Background context: After ensuring third-party modules are correctly typed, you proceed to migrate your own custom modules. This typically starts from the bottom of the dependency graph, focusing on utility or helper functions before moving towards more complex parts of the codebase.

:p What is the recommended order for migrating custom modules?
??x
Start with the lowest-level utility and helper functions in your project. These are usually found at the base of the dependency tree and do not rely heavily on other components.

Example:
If you have a `config.ts` file that contains global configuration values, start by converting this before moving to more complex modules like services or controllers.

??x

---

#### Visualizing Dependency Graphs

Background context: To help manage the migration process, it is useful to visualize the dependency graph of your project. Tools such as `ts-node-dev` with a custom script can be used to explore and map out dependencies.

:p Why is visualizing the dependency graph important during TypeScript migration?
??x
Visualizing the dependency graph helps in understanding which parts of the codebase depend on each other, allowing you to prioritize migrations effectively. It also aids in tracking progress and ensures that no critical modules are overlooked.

For example, using `ts-node-dev` with a custom script can help:
```sh
ts-node-dev --transpile-only ./node_modules/ts-node-digraph/cli.js > dependency_graph.dot
```
This generates a DOT file which can be visualized with tools like Graphviz to get an overview of the project's architecture.

??x

---

#### Refactoring Considerations During Migration

Background context: While migrating, it is important to avoid refactoring code unless absolutely necessary. Instead, keep track of ideas for future refactorings and focus on type conversion tasks only.

:p What should you avoid doing while converting your codebase to TypeScript?
??x
Avoid refactoring or changing the design of your code as you migrate from JavaScript to TypeScript. Focus solely on adding types where needed and fixing any type-related errors. Keep a list of potential future refactorings but do not implement them yet.

For example, if you come across an odd or poorly designed part of the code:
```typescript
class OddClass {
  // some logic here
}
```
Note it down for later refactoring, but keep it as is during the initial migration phase.

??x

---

#### Enabling noImplicitAny

Background context: Once you have added types to your codebase and are ready to enforce stricter type checking, enable `noImplicitAny`. This setting ensures that all variables must be explicitly typed, preventing loose type checks from masking real errors.

:p What does enabling `noImplicitAny` do?
??x
Enabling `noImplicitAny` forces the TypeScript compiler to treat any untyped variables as having an implicit `any` type. This helps in catching potential type-related issues early during development.

For example:
Before:
```typescript
class Chart {
  indices: any;
  
  getRanges() {
    for (const r of this.indices) {
      const low = r[0];
      const high = r[1];
    }
  }
}
```
After enabling `noImplicitAny` and fixing types:
```typescript
class Chart {
  indices: number[][];
  
  getRanges() {
    for (const r of this.indices) {
      const low = r[0]; // Error: Element implicitly has an 'any' type because...
      const high = r[1]; // Error: Element implicitly has an 'any' type because...
    }
  }
}
```

??x

---

#### Tracking Progress During Migration

Background context: To ensure you are making progress during the migration, track your type coverage and monitor the number of `noImplicitAny` errors. This helps in measuring the effectiveness of the changes made.

:p How can you track progress during the TypeScript migration?
??x
You can use tools like the `--watch` option with `tsc` to continuously check for new type issues as you make changes:
```sh
npx tsc --watch
```
Additionally, tracking your type coverage (using tools like Istanbul) gives a good sense of how much of your codebase is properly typed.

For example:
Use the `--project` flag with `istanbul` to generate a report on type coverage:
```sh
npx istanbul cover npx tsc -- --project tsconfig.json
```

??x

---

#### Committing Changes During Migration

Background context: As you make changes during the migration, commit your progress frequently. This helps in maintaining a history of changes and makes it easier to revert if necessary.

:p What is recommended practice for committing changes during TypeScript migration?
??x
Commit your type corrections and other changes regularly. This ensures that you have a clear record of your progress and can easily backtrack if needed.

Example:
Instead of making all changes at once, commit after fixing each file or module:
```sh
git add .
git commit -m "Fix types in Chart class"
```

??x

---

#### Prioritizing Type Safety

Background context: Depending on the criticality of your codebase, you might choose to prioritize type safety in different parts. For instance, production code might need stricter type checks than test code.

:p How can you handle type safety priorities differently in various parts of a project?
??x
You can have distinct `tsconfig.json` files with different settings for different parts of your project. For example, you can have one configuration file that is more permissive and another that enforces stricter types.

Example:
```json
// tsconfig.prod.json
{
  "compilerOptions": {
    "noImplicitAny": true,
    "strictNullChecks": true
  }
}

// tsconfig.unit.json
{
  "compilerOptions": {
    "noImplicitAny": false
  }
}
```

You can use `ts-node-dev` with project references to manage these configurations.

??x

---

#### Enforcing Strict Type Checking

Background context: After enabling `noImplicitAny`, you can further increase the strictness of type checking by turning on other options like `strictNullChecks`. These settings help in catching more potential issues early in the development process.

:p What are some additional ways to enforce stricter type checking?
??x
In addition to `noImplicitAny`, you can enable `strictNullChecks` and other `compilerOptions` to increase the strictness of your code. For example:
```json
{
  "compilerOptions": {
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "alwaysStrict": true
  }
}
```

These settings help in ensuring that your TypeScript code is robust and type-safe.

??x

---

---
#### Understand the Relationship Between TypeScript and JavaScript
Background context: TypeScript is a superset of JavaScript that adds static typing to help catch errors during development. Understanding this relationship is crucial for effective use of TypeScript.

:p What is the relationship between TypeScript and JavaScript?
??x
TypeScript is a language built on top of JavaScript, adding type checking and other features while still compiling down to standard JavaScript. This means all valid JavaScript code is also valid TypeScript, but not vice versa.
```typescript
// Example in TypeScript
function sayHello(name: string): void {
    console.log(`Hello, ${name}!`);
}

// The same function in JavaScript
function sayHello(name) {
    console.log(`Hello, ${name}!`);
}
```
x??

---
#### Know Which TypeScript Options You’re Using
Background context: TypeScript has various options that control how the language behaves. Being aware of these can help you configure your development environment and avoid common pitfalls.

:p What are some key TypeScript compiler options?
??x
Key TypeScript compiler options include `noImplicitAny`, `strictNullChecks`, and `checkJs`. These options affect type inference, null checks, and JavaScript file handling respectively.
```bash
tsc --init  # Initializes a tsconfig.json file with default settings
```
x??

---
#### Understand That Code Generation Is Independent of Types
Background context: TypeScript’s static typing is purely for development-time checking. The generated JavaScript does not include these types; they are used during compilation to catch errors.

:p How does code generation differ from type checking in TypeScript?
??x
Code generation in TypeScript is independent of its type system. While TypeScript uses the type information at compile time to generate strongly-typed JavaScript, this typing information is discarded after compilation. The generated JavaScript will not include any type annotations.
```typescript
// TypeScript
function add(a: number, b: number): number {
    return a + b;
}

// Generated JavaScript (by TypeScript compiler)
function add(a, b) {
    return a + b;
}
```
x??

---
#### Get Comfortable with Structural Typing
Background context: Structural typing in TypeScript allows for type compatibility based on structure rather than explicit inheritance. This feature is powerful but can lead to unexpected results if not used carefully.

:p What is structural typing in TypeScript?
??x
Structural typing in TypeScript determines type compatibility based on the structure of objects, not their class hierarchy. For example, an object with the same shape as a defined interface or class is considered compatible.
```typescript
interface Point { x: number; y: number; }

const pointLike = { x: 0, y: 0 };
let p: Point = pointLike; // Valid due to structural typing
```
x??

---
#### Limit Use of the any Type
Background context: The `any` type in TypeScript is a wildcard that bypasses most type checking. Overusing it can lead to runtime errors and defeat the purpose of static typing.

:p Why should you limit the use of the `any` type?
??x
Limiting the use of the `any` type helps maintain the benefits of static typing by reducing the risk of runtime errors and making code more predictable. It is generally advisable to use more precise types where possible.
```typescript
// Overuse of any can lead to issues
let value: any = 5;
value = "hello"; // No error, but could cause runtime issues

// Using a specific type is better practice
let value2: number = 5;
value2 = "hello"; // Error at compile time
```
x??

---

