# Flashcards: 10A007---Effective-TypeScript---Dan-Vanderkam_processed (Part 26)

**Starting Chapter:** Item 76 Create an Accurate Model of Your Environment

---

#### Understanding DOM Types in TypeScript
Background context: In JavaScript, the Document Object Model (DOM) has a type hierarchy that is often ignored. However, this becomes crucial when using TypeScript due to its static typing system. Understanding the differences between `Node`, `Element`, `HTMLElement`, and `EventTarget` will help you write more robust and type-safe code.
:p What are the main types in the DOM that developers should be familiar with in TypeScript?
??x
The main types in the DOM relevant for TypeScript include:
- **Node**: The base class representing nodes in the document tree. This can be a text node, element, comment, etc.
- **Element**: A more specific type of `Node` representing HTML elements (e.g., `<div>`, `<span>`).
- **HTMLElement**: An even more specialized type of `Element` that represents HTML elements and provides additional properties and methods.
- **EventTarget**: The base interface for objects that can emit events.

These types are particularly important when dealing with DOM operations in TypeScript, as they help you define the correct type annotations for variables or function parameters.

```typescript
// Example usage
const element: HTMLElement = document.querySelector('div');
element.addEventListener('click', (event: MouseEvent) => {
  console.log(event);
});
```
x??

---
#### Using lib Setting in tsconfig.json
Background context: The `lib` setting in your TypeScript configuration file (`tsconfig.json`) is used to specify the set of libraries and language features that TypeScript should assume are available. This helps TypeScript perform type checking based on a specific runtime environment, such as a browser.
:p How do you include browser-specific types using the `lib` setting?
??x
You can include browser-specific types in your `tsconfig.json` file by adding `"dom"` to the `lib` array:
```json
{
  "compilerOptions": {
    "lib": ["dom", "es2018"]
  }
}
```
This tells TypeScript that it should use type declarations for the DOM and assume ES2018 features are supported. By including `"dom"`, you ensure that TypeScript can correctly handle types related to HTML elements, events, etc.

If you want more granular control over which features are included, you can specify multiple libraries or even individual modules.
x??

---
#### Modeling Global Variables in TypeScript
Background context: When your script interacts with global variables defined by other scripts on the page (e.g., `window.userInfo`), it’s important to declare these globally available types so that TypeScript doesn’t complain about them during type checking. This ensures accurate and consistent type checking across all parts of your application.
:p How can you model a global variable in TypeScript?
??x
You can model global variables by using the `declare global` syntax. For instance, if there is a `window.userInfo` object:
```typescript
// user-info-global.d.ts
interface UserInfo {
    name: string;
    accountId: string;
}

declare global {
    interface Window {
        userInfo: UserInfo;
    }
}
```
This tells TypeScript that the `Window` interface has an additional property called `userInfo`. You can then use this type safely in your code without any issues.

```typescript
// main.ts
const user = window.userInfo;
console.log(user.name);  // No errors!
```
x??

---
#### Installing @types Packages for Libraries
Background context: If you are using third-party libraries like jQuery or Google Analytics, it’s a good practice to install their type definitions via `@types` packages. This ensures that TypeScript has the correct type information and can provide better intellisense and error checking.
:p How do you model types available in a library using @types?
??x
You can model types for libraries by installing their corresponding `@types` package. For example, to get type definitions for jQuery:
```sh
npm install --save-dev @types/jquery
```
Similarly, for Google Analytics (or other third-party services):
```sh
npm install --save-dev @types/google.analytics
```
By doing this, you ensure that TypeScript can correctly check the types and methods available in these libraries.

Example of using jQuery with its type definitions:
```typescript
import $from 'jquery';$(document).ready(() => {$('#myElement').click(() => {
        console.log('Clicked!');
    });
});
```
x??

---
#### Handling Module Imports with webpack
Background context: When you use tools like webpack to import non-JavaScript files (like images or CSS) directly in your JavaScript code, these imports are part of the environment and should be modeled appropriately so that TypeScript can handle them without errors.
:p How do you model type definitions for file imports using webpack?
??x
You can model types for specific file imports by creating a custom declaration file. For instance, if you import an image:
```typescript
// webpack-imports.d.ts
declare module '*.jpg' {
    const src: string;
    export default src;
}
```
This tells TypeScript that any `*.jpg` file imported will be treated as a string.

Example of importing and using the image in TypeScript code:
```typescript
import sunrisePath from './images/beautiful-sunrise.jpg';

console.log(sunrisePath);  // No errors!
```
x??

---

#### Environment Modeling for TypeScript Applications

Background context: When developing applications using TypeScript, especially when dealing with complex projects that run across different environments (e.g., browser, Node.js, and test environments), it's crucial to accurately model each environment. This involves creating distinct `tsconfig.json` files for each environment and ensuring type declarations match the actual runtime versions.

:p How can you ensure accurate modeling of different environments in a TypeScript project?
??x
To ensure accurate modeling of different environments, use multiple `tsconfig.json` files with specific settings for each environment (e.g., browser, Node.js). Ensure that the types used in your code are aligned with the actual runtime versions. For example, if you're using Node.js version 20, install `@types/node` from that version to match the type declarations.

```json
// Example tsconfig.json for Node.js
{
    "compilerOptions": {
        "target": "es2020",
        "module": "commonjs",
        "typeRoots": ["./node_modules/@types"]
    }
}
```
x??

---

#### TypeScript and Unit Testing

Background context: While TypeScript can catch certain errors through static type checking, it is not a substitute for unit testing. Both serve different purposes in ensuring code correctness.

:p How do type annotations help in finding bugs in the `add` function example?
??x
Type annotations help by catching certain bugs that are hard to detect with just unit tests. For instance, in the `add` function:

```typescript
function add(a: number, b: number): number {
    if (isNaN(a) || isNaN(b)) {
        return 'Not a number.'; // ~~~ Type 'string' is not assignable to type 'number'.
    }
    return (a | 0) + (b | 0);
}
```

The type checker flags the error when trying to return a string from a function that returns a `number`, ensuring that such bugs are caught early.

```typescript
test('add with NaN', () => {
    expect(() => add(NaN, 1)).toThrow(); // This will pass because TypeScript catches it.
});
```
x??

---

#### Complementary Processes of Unit Testing and Type Checking

Background context: Unit tests and type checking both serve as forms of verification but cover different aspects. Unit tests provide a lower bound on correctness by demonstrating expected behavior in some cases, while type checking provides an upper bound on incorrectness by catching certain types of mistakes.

:p How do unit tests and type checking complement each other in the `add` function example?
??x
Unit tests and type checking complement each other as follows:

- **Type Checking:** Catches static errors such as returning the wrong type or performing invalid operations.
  ```typescript
  function add(a: number, b: number): number {
      if (isNaN(a) || isNaN(b)) {
          return 'Not a number.'; // ~~~ Type 'string' is not assignable to type 'number'.
      }
      return (a | 0) + (b | 0);
  }
  ```

- **Unit Testing:** Demonstrates expected behavior in specific cases.
  ```typescript
  test('add', () => {
      expect(add(0, 0)).toEqual(0);
      expect(add(123, 456)).toEqual(579);
      expect(add(-100, 90)).toEqual(-10);
  });
  ```

Together, they ensure that the function is robust and behaves correctly across a wide range of inputs.

```typescript
test('add with NaN', () => {
    expect(() => add(NaN, 1)).toThrow(); // This will pass because TypeScript catches it.
});
```
x??

---

#### Handling Out-of-Domain Calls

Background context: When working with functions that can accept any type of argument (e.g., `add` function), you should rely on the type checker to prevent invalid calls. Unit tests are used for demonstrating expected behavior, not for all possible inputs.

:p How do unit tests and type checking handle out-of-domain calls in TypeScript?
??x
Unit tests and type checking handle out-of-domain calls as follows:

- **Type Checking:** Ensures that functions are called with the correct types.
  ```typescript
  test('out-of-domain add', () => {
      expect(add(null, null)).toEqual(0); // ~~~~ Type 'null' is not assignable to parameter of type 'number'.
      expect(add(null, 12)).toEqual(12); // ~~~~ Type 'null' is not assignable to parameter of type 'number'.
      expect(add(undefined, null)).toBe(NaN); // ~~~~~~~~~ Type 'undefined' is not assignable to parameter of ...
      expect(add('ab', 'cd')).toEqual('abcd'); // ~~~~ Type 'string' is not assignable to parameter of type  'number'.
  });
  ```

- **Unit Testing:** Should focus on demonstrating expected behavior for valid inputs.
  ```typescript
  test('add with valid numbers', () => {
      expect(add(0, 0)).toEqual(0);
      expect(add(123, 456)).toEqual(579);
      expect(add(-100, 90)).toEqual(-10);
  });
  ```

This ensures that the function is used correctly and that invalid inputs are caught early.

```typescript
test('add with valid numbers', () => {
    expect(add(0, 0)).toEqual(0);
    expect(add(123, 456)).toEqual(579);
    expect(add(-100, 90)).toEqual(-10);
});
```
x??

---

#### Side Effects and Type Safety

Background context: Functions that have side effects (e.g., updating a database) should be carefully tested to ensure they handle unexpected inputs correctly. Type checking alone may not catch all issues.

:p How can you test the `updateUserById` function with TypeScript?
??x
You can use an `@ts-expect-error` directive in your tests to assert that certain calls are type errors, ensuring that side effects are handled correctly:

```typescript
test('invalid update', () => {
    // @ts-expect-error Can't call updateUserById to update an ID.
    expect(() => updateUserById('123', { id: '234' })).toReject(); // This will pass because TypeScript catches it.
});
```

This ensures that the function behaves as expected when given unexpected inputs.

```typescript
test('invalid update', () => {
    // @ts-expect-error Can't call updateUserById to update an ID.
    expect(() => updateUserById('123', { id: '234' })).toReject();
});
```
x??

---

#### Type Checking and Unit Testing Complementarity
Type checking helps catch incorrect behaviors by ensuring type safety, while unit tests ensure correct behavior on specific inputs. Both are important for program correctness but serve different purposes.
:p How do type checking and unit testing complement each other?
??x
Both type checking and unit testing are crucial for ensuring program correctness from different perspectives. Type checking ensures that types align correctly, preventing many errors at compile time. Unit tests, on the other hand, verify specific behaviors of your code with particular inputs. Relying on both helps catch a wide range of bugs.
??x

---

#### Separating Type Checking from Building
In TypeScript, you can separate type checking from building to improve performance. This reduces build times by skipping the type checking step if it's not needed.
:p How does separating type checking from building help with performance?
??x
Separating type checking from building allows you to disable the type checking process during builds, which can be resource-intensive. This speeds up the build process since TypeScript types are erased at runtime anyway (Item 78). You can configure your build tooling to only emit JavaScript files without running the type checker.
??x

---

#### Compiler Performance Impact
TypeScript has minimal impact on runtime performance but can affect developer tooling performance, particularly with `tsc` and `tsserver`.
:p How does TypeScript impact compiler performance?
??x
`tsc`, the TypeScript compiler, impacts build performance by checking your code for type errors. Slower `tsc` means longer compile times and slower production of JavaScript files. `tsserver`, the language service, can make editors feel sluggish if it takes a long time to update error messages or suggest completions.
??x

---

#### Editor Performance Considerations
`tsserver` performance directly affects your editor experience, causing delays in error detection and response times.
:p How does `tsserver` affect developer tooling?
??x
`tsserver` performance is critical for a smooth editing experience. If it's slow, you might experience long wait times for errors to appear or disappear after code changes, making development less efficient. Optimizing `tsserver` can significantly improve the responsiveness of your editor.
??x

---

#### Avoiding Redundant Checks
Avoid testing inputs that would be type errors in unit tests unless there are specific security or data corruption concerns.
:p How should you handle type error-prone inputs in unit tests?
??x
In unit tests, focus on verifying behaviors rather than catching type errors. If an input is a type error and it should never occur in your application, avoid testing these inputs explicitly. Instead, ensure that the types themselves are correct through TypeScript annotations. Only test inputs that make sense from a business logic standpoint.
??x

---

#### Optimizing Build Performance
You can use various techniques to optimize build performance when separating type checking from building.
:p What techniques can be used to improve build performance?
??x
To improve build performance, you might consider the following strategies:
1. **Incremental Builds**: Use tools like `tsc --build` which runs only necessary checks.
2. **Parallel Build Execution**: Run multiple builds in parallel if you have a multi-core system.
3. **Selective Compilation**: Compile only changed files or specific modules.
4. **Tooling Configuration**: Optimize your build tool configuration to reduce unnecessary steps.

These techniques can help balance between thorough type checking and fast build times.
??x

#### Transpile Only Mode
Background context: The text discusses how tools like ts-node and bundlers can perform type checking and then bundle or run the generated JavaScript. However, they do not necessarily need to perform these steps, especially if you are only interested in transpiling TypeScript code into JavaScript.
:p What is the purpose of running a tool like ts-node in "transpile only" mode?
??x
Running ts-node with `--transpileOnly` will tell it to skip type checking and simply transform your TypeScript code into JavaScript. This can significantly speed up development time, especially for small or trivial programs where the time taken by type checking is noticeable.
```bash
$time ts-node --transpileOnly hello.ts  # Runs faster as it skips type checking
```
x??

---

#### Impact on Iteration Cycle
Background context: The text highlights how turning off type checking can improve developer experience (DX) in a development environment. Tools like ts-node or bundlers that perform this task can be set to transpile code only, which is beneficial for quick iterations.
:p How does running TypeScript with `--transpileOnly` affect the iteration cycle?
??x
Running TypeScript with `--transpileOnly` significantly reduces the time it takes to run a program by skipping type checking. For example, a trivial TypeScript program runs much faster without type checking:
```bash$ time ts-node --transpileOnly hello.ts  # Quick execution due to no type checking
```
This can greatly enhance the developer's iteration cycle and overall experience.
x??

---

#### Pruning Unused Dependencies
Background context: The text suggests that dead code elimination can help reduce the size of TypeScript projects, leading to faster compilation times. By setting flags like `noUnusedLocals`, TypeScript can detect and report unused symbols.
:p What is the purpose of pruning unused dependencies in a TypeScript project?
??x
Pruning unused dependencies helps minimize the amount of code processed by the TypeScript compiler, thereby reducing build times and improving editor responsiveness. It involves identifying and removing unused local variables, functions, and types from your codebase to make it more efficient.
```typescript
function foo() {} //       ~~~ 'foo' is declared but its value is never read.
export function bar() {}
```
x??

---

#### Dead Code Elimination with knip
Background context: The text mentions that using a tool like `knip` can help identify and remove unused third-party dependencies, not just local code. This can significantly reduce the overall size of your project by pruning unnecessary type declarations.
:p How does `knip` assist in managing TypeScript projects?
??x
`knip` is a tool designed to detect and report both local and third-party unused symbols and dependencies. By using `knip`, you can identify unused third-party modules, which often contain extensive type declarations that are not used in your codebase. This helps reduce the overall size of your project and improve performance.
```bash
$tsc --listFiles  # Lists all sources included in the TypeScript project
```
x??

---

#### Visualizing Dependencies with Treemaps
Background context: The text concludes by suggesting a visual approach to understanding dependencies using treemaps. This visualization can help developers see how one dependency might bring in many others, making it easier to identify unnecessary or redundant modules.
:p What is the purpose of using treemaps for managing TypeScript projects?
??x
Using treemaps to visualize dependencies provides a graphical representation of how different modules and their sub-modules relate to each other. This can help developers understand the complexity of their project's dependency tree, making it easier to identify and remove unused or redundant dependencies.
```bash$ tsc --listFiles  # Produces output showing all included sources
```
By visualizing these relationships, you can better manage your project's dependencies and improve overall performance.
x??

#### Treemap Visualization for File Size Analysis
Background context: The author discusses using a treemap visualization to understand the file size of TypeScript files during compilation. This tool helps identify large dependencies and optimize project size.

:p How can you visualize the file sizes of TypeScript files being compiled?
??x
You can use the following command:
```sh
tsc --noEmit --listFiles | xargs stat -f "percentz percentN" | npx webtreemap-cli
```
This command sequence compiles TypeScript files without emitting output, lists all file paths to be processed, uses `stat` (with platform-specific options) to get file sizes, and finally visualizes these as a treemap using `webtreemap-cli`.

x??

---

#### Large Dependency Management with Googleapis Example
Background context: The author mentions that large dependencies like Google APIs can significantly impact project size. In one example, the entire set of Google APIs (300+) was bundled together in one package, which weighed 80.5 MB.

:p Why did the author consider updating to a newer version of googleapis?
??x
The author considered updating because the newer version likely provides better support for selectively including only necessary APIs, reducing the overall bundle size and improving performance.

x??

---

#### Incremental Builds in TypeScript
Background context: Incremental builds allow TypeScript to save work during compilation. This can significantly reduce build times by reusing information from previous compilations.

:p How does incremental builds work in TypeScript?
??x
When you run `tsc --incremental` (or simply `tsc -b`), it writes a `.tsbuildinfo` file that saves the state of the compilation. On subsequent runs, `tsc` reads this file and uses it to perform type checks more quickly.

x??

---

#### Project References in TypeScript
Background context: Project references allow distinct parts of a project to be compiled independently, reducing redundant work during development.

:p How do you set up project references in a TypeScript project?
??x
You create separate `tsconfig.json` files for each part of your repository. These files specify which other parts they can reference. You also have a top-level `tsconfig.json` for shared configuration. Here is an example setup:

```json
// tsconfig-base.json
{
  "compilerOptions": {
    // other settings
    "declaration": true,
    "composite": true
  }
}

// tsconfig.json
{
  "files": [],
  "references": [
    { "path": "./src" },
    { "path": "./test" }
  ]
}

// src/tsconfig.json
{
  "extends": "../tsconfig-base.json",
  "compilerOptions": {
    "outDir": "../dist/src",
    "rootDir": "."
  }
}
```

```json
// test/tsconfig.json
{
  "extends": "../tsconfig-base.json",
  "compilerOptions": {
    "outDir": "../dist/test",
    "rootDir": "."
  },
  "references": [
    { "path": "../src" }
  ]
}
```

With this setup, running `tsc -b` will coordinate the build process and only recompile parts that have changed.

x??

---

#### General Rule of Thumb for Project Structure
Background context: The general rule suggests that TypeScript is more beneficial when you have a higher proportion of your own first-party code compared to third-party libraries. This often applies to large corporations with extensive custom codebases, but may not hold true for smaller projects.

:p What does the general rule suggest about project structure in relation to TypeScript?
??x
The rule suggests that TypeScript is most beneficial when you have a significant amount of your own first-party code relative to third-party dependencies. For large corporations with substantial custom code, using TypeScript can significantly aid in maintaining and scaling their applications.
x??

---

#### Project Organization for Large Applications
Background context: Proper organization of projects can enhance development speed and performance. Grouping related functionalities into distinct but manageable chunks is recommended.

:p How should you structure your projects for larger applications?
??x
Projects for large applications should be organized by grouping related functionalities into separate, well-defined projects. For example, distinguishing between `src` and `test` directories or separating client-side from server-side code can improve manageability.
x??

---

#### Simplifying Types to Improve Performance
Background context: Overly complex types can significantly degrade TypeScript performance due to the increased number of checks required during compilation.

:p What is a potential consequence of using overly complex union types?
??x
Using overly complex union types, like a `Year` type that represents every possible year from 2000 to 2999, can lead to performance issues. TypeScript has to perform extensive type checking for each operation involving such a type, which can slow down both the build process (`tsc`) and editor latency (`tsserver`).
x??

---

#### Efficient Type Design
Background context: Choosing simpler types over complex ones can improve both build and editor performance.

:p How should you design your types to optimize performance?
??x
To optimize performance, use simpler types like `string` or `number` instead of large unions. For instance, using a string for the `Year` type would be more efficient than defining it as a union of all possible years.
x??

---

#### Extending Interfaces vs Intersecting Type Aliases
Background context: Interface extension is generally more efficient for subtyping compared to intersecting type aliases.

:p Why should you prefer interfaces over types when implementing subtypes?
??x
You should prefer extending interfaces over intersecting type aliases because TypeScript can handle subtyping operations more efficiently with interfaces. This means that using `class` or `interface` inheritance will be processed faster by the TypeScript compiler.
x??

---

#### Annotating Return Types
Background context: Explicitly annotating return types in functions can help the TypeScript compiler infer types more accurately and reduce unnecessary work.

:p When should you consider adding type annotations to function return types?
??x
You should consider adding explicit type annotations to function return types, especially when dealing with complex operations or recursive types. This helps the TypeScript compiler understand the expected output without having to do extensive inference.
x??

---

#### Managing Large Unions
Background context: Avoiding large unions can prevent performance issues in TypeScript.

:p What is a common issue with using large unions?
??x
A common issue with using large unions is that they can significantly impact performance. Each operation involving these types requires checking against all possible union members, which can be slow.
x??

---

#### Recognizing Performance Issues
Background context: Understanding the difference between build performance and editor latency is crucial for effective optimization.

:p How do you differentiate between build performance issues and editor latency?
??x
Build performance issues refer to problems encountered during the TypeScript compiler (`tsc`) process, while editor latency refers to delays experienced in real-time editing and code suggestions by `tsserver`. Recognizing these differences helps target optimizations more effectively.
x??

---

#### Simplifying Types Further
Background context: Additional strategies for optimizing types include using branded types or extending existing interfaces.

:p What are some ways to simplify your types further?
??x
To simplify your types, you can use branded types if needed (as discussed in Item 64) or extend existing interfaces instead of intersecting type aliases. These approaches help maintain simpler and more efficient type definitions.
x??

---

#### Visualizing Compilation with Treemaps
Background context: Using treemaps provides a visual representation of what TypeScript is compiling, helping you identify unnecessary dependencies.

:p How can you use treemaps to optimize your project?
??x
You can use treemaps to visualize the compilation process and identify unnecessary or bloated code. By analyzing these maps, you can pinpoint areas where dead code or redundant type dependencies exist, allowing for targeted optimization.
x??

---

#### Incremental Builds with Project References
Background context: Using incremental builds and project references can reduce the work `tsc` does between builds.

:p How do incremental builds help in optimizing TypeScript performance?
??x
Incremental builds allow `tsc` to only recompile changed files, reducing the overall compilation time. By leveraging project references, you can further minimize the work required by including only necessary dependencies during each build.
x??

---

#### Summary of Key Points
Background context: The text covers several key points about optimizing TypeScript performance through efficient type design and project organization.

:p What are some key strategies for improving TypeScript performance?
??x
Key strategies include simplifying types to avoid large unions, using interfaces instead of intersecting type aliases for subtyping, annotating function return types, organizing projects into manageable chunks, and utilizing tools like treemaps and incremental builds.
x??
---

