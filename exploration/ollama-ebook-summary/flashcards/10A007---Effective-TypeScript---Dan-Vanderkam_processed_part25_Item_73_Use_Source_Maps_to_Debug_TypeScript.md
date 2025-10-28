# Flashcards: 10A007---Effective-TypeScript---Dan-Vanderkam_processed (Part 25)

**Starting Chapter:** Item 73 Use Source Maps to Debug TypeScript

---

#### Source Maps in TypeScript Debugging
Background context explaining source maps and their role in debugging TypeScript. Explain how TypeScript compiles to JavaScript, and why direct debugging of generated JavaScript can be challenging.

:p What are source maps, and why are they important for debugging TypeScript code?
??x
Source maps are mappings between a script’s original source code (like TypeScript) and the compiled version (JavaScript). They enable debuggers to show you the original source code even when working with the generated JavaScript. This is crucial because direct debugging of generated JavaScript can be difficult due to its complex structure, especially in cases involving asynchronous operations or minification.

For example, consider a simple TypeScript function:

```typescript
// index.ts
function addCounter(el: HTMLElement) {
    let clickCount = 0;
    const button = document.createElement('button');
    button.textContent = 'Click me';
    button.addEventListener('click', () => {
        clickCount++;
        button.textContent = `Click me (${clickCount})`;
    });
    el.appendChild(button);
}
addCounter(document.body);
```

When this is compiled to ES5 JavaScript, the code can become quite different and harder to debug:

```javascript
// index.js
var __extends = (this && this.__extends) || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
};
var addCounter = function (el) {
    var clickCount = 0;
    var button = document.createElement('button');
    button.textContent = 'Click me';
    button.addEventListener('click', (function (_this) { return function () { _this.clickCount++; _this.button.textContent = "Click me (" + _this.clickCount + ")"; }; })(this));
    el.appendChild(button);
};
addCounter(document.body);
```

To debug this in the browser, you would use a source map to view and interact with the original TypeScript code.

x??

---
#### Enabling Source Maps
Explanation of how to enable source maps in a TypeScript project. Discuss modifying `tsconfig.json` and the resulting output files.

:p How do you enable source maps for debugging in your TypeScript project?
??x
To enable source maps in your TypeScript project, you need to modify the `tsconfig.json` file. Specifically, you should add or update the `"compilerOptions"` section with the `"sourceMap": true` option:

```json
{
    "compilerOptions": {
        "sourceMap": true
    }
}
```

When you run `tsc`, it will generate both a `.js` file and a corresponding `.js.map` file for each TypeScript file. The `.js.map` file contains the mapping information that allows your debugger to translate the JavaScript code back into the original TypeScript source.

For example, if you have an `index.ts` file:

```typescript
// index.ts
function addCounter(el: HTMLElement) {
    let clickCount = 0;
    const button = document.createElement('button');
    button.textContent = 'Click me';
    button.addEventListener('click', () => {
        clickCount++;
        button.textContent = `Click me (${clickCount})`;
    });
    el.appendChild(button);
}
addCounter(document.body);
```

After compiling with the source map option enabled, you will get an `index.js` file and an `index.js.map` file:

```shell
$ tsc index.ts
```

These files can be used by your debugger to show you the original TypeScript code during debugging sessions.

x??

---
#### Debugging Generated JavaScript
Explanation of how source maps affect debugging in a browser environment. Discuss setting breakpoints and inspecting variables using the debugger.

:p How does enabling source maps help with debugging generated JavaScript in a browser?
??x
Enabling source maps helps by allowing you to debug your TypeScript code as if it were still the original source, even though the browser is running the compiled JavaScript.

When you enable source maps in your `tsconfig.json`, the TypeScript compiler generates both `.js` and `.js.map` files. When you open the debugger (e.g., Chrome DevTools), you can see a file named after your original TypeScript file (`index.ts`) with an italics font, indicating that this is not a real file but one included via the source map.

For example, in Chrome DevTools:

1. Load your application.
2. Open the Sources panel.
3. You should see `index.ts` listed as if it were part of your project files.
4. Set breakpoints on lines of code in `index.ts`.
5. Interact with your application to hit those breakpoints.

The debugger will pause execution at the correct line of TypeScript, showing you the original source code and allowing you to inspect variables.

Here’s an example setup:

```shell
$ tsc index.ts  # Generates index.js and index.js.map
```

In Chrome DevTools:

1. Open `index.ts` in the Sources panel.
2. Set a breakpoint on line 3 of `index.ts`.
3. Refresh the page or trigger some actions to hit the breakpoint.

The debugger will pause at the correct location, showing you the original TypeScript code and allowing you to inspect variables and step through the code.

x??

---
#### Debugging Node.js Programs
Explanation of how source maps can be used for debugging in Node.js applications. Discuss using `--inspect-brk` flag with Node.

:p How do you use source maps to debug Node.js programs?
??x
To debug Node.js programs using source maps, you first need to ensure that your TypeScript project is set up correctly with source map generation enabled. Then, you can compile and run the program in a way that supports debugging.

Here’s an example of how to do this:

1. **Compile the TypeScript code:**

   ```shell
   $ tsc bedtime.ts  # Generates bedtime.js and bedtime.js.map
   ```

2. **Run the compiled JavaScript with Node's `--inspect-brk` flag:**

   ```shell
   $ node --inspect-brk bedtime.js
   Debugger listening on ws://127.0.0.1:9229/587c380b-fdb4-48df-8c09-a83f36d8a2e7
   For help, see: https://nodejs.org/en/docs/inspector
   ```

3. **Open your browser and navigate to `chrome://inspect`** (or a similar interface in other browsers).

4. **Select the remote target to inspect**:

   - In Chrome, you might see something like "Node.js Runtime".

5. **Set breakpoints and interact with the program**:

   - The debugger will pause at the specified line when the script reaches it.

For example, consider the `bedtime.ts` file:

```typescript
// bedtime.ts
async function sleep(ms: number) {
    return new Promise<void>((resolve) => setTimeout(resolve, ms));
}

async function main() {
    console.log('Good night.');
    await sleep(1000);
    console.log('Morning already.?');
}

main();
```

After compiling and running with `--inspect-brk`, you can set a breakpoint in the TypeScript source code within your browser’s debugger:

```shell
$ tsc bedtime.ts  # Generates bedtime.js and bedtime.js.map
$ node --inspect-brk bedtime.js  # Debugger starts
```

In Chrome DevTools:

1. Go to `chrome://inspect`.
2. Select "Node.js Runtime" (or similar).
3. Set a breakpoint on line 4 of `bedtime.ts`:
   ```typescript
   async function main() {
   ```

When you run the application, it will pause at this point, and you can inspect variables and step through the code.

x??

---

#### Source Maps and Debugging Node.js TypeScript Code
Background context: When working with Node.js applications written in TypeScript, understanding how to properly configure and use source maps is crucial. Source maps help maintain the connection between compiled JavaScript files and their original TypeScript code during debugging sessions. In some cases, especially when using tools like `ts-node`, the source maps might contain an inline copy of your original code, which can be useful for debugging but should not be published.
:p How do you ensure that source maps are correctly configured in a Node.js application written in TypeScript?
??x
To ensure that source maps are correctly configured, you need to configure your `tsconfig.json` and `jsconfig.json` (if using Visual Studio Code) files properly. For instance:
```json
// tsconfig.json
{
  "compilerOptions": {
    "sourceMap": true,
    "inlineSources": true // This ensures the source maps contain inline copies of code.
  }
}
```
You also need to ensure that your development environment is set up to utilize these source maps, such as using `ts-node` with appropriate flags:
```shell
ts-node --project . --inspect
```
This setup allows you to debug TypeScript code by stepping through the original source files. However, it's important not to publish the generated JavaScript code and associated source maps unless necessary.
x??

---

#### Reconstructing Types at Runtime in TypeScript
Background context: While TypeScript types are erased during runtime, there might be situations where you need to access these types dynamically. This can happen when implementing complex APIs or middleware that needs to validate input against defined interfaces.
:p How do you address the problem of validating request bodies in a web server using TypeScript types at runtime?
??x
To address this issue, one approach is to generate validation logic from your TypeScript interfaces. Here’s an example:
```typescript
interface CreateComment {
    postId: string;
    title: string;
    body: string;
}

function validateCreateComment(body: any): boolean {
    const comment = body as CreateComment;
    return (
        comment.postId &&
        typeof comment.postId === 'string' &&
        comment.title &&
        typeof comment.title === 'string' &&
        comment.body &&
        typeof comment.body === 'string'
    );
}

app.post('/comment', (request, response) => {
    const { body } = request;
    if (!validateCreateComment(body)) {
        return response.status(400).send('Invalid request');
    }
    // Continue with application validation and logic
    return response.status(200).send('ok');
});
```
This function `validateCreateComment` ensures that the request body matches the `CreateComment` interface. By generating this kind of validation code, you avoid duplicating type checks in your application logic.
x??

---

#### Generating TypeScript Types from OpenAPI Schema
Background context: When working with APIs, it is common to define types and validation logic. For APIs that use OpenAPI specs, you can leverage tools like `json-schema-to-typescript` for generating TypeScript types and a JSON schema validator like Ajv for request validation.
:p What tool would you use to generate TypeScript types from an OpenAPI spec?
??x
You would use the `json-schema-to-typescript` tool. This tool converts your OpenAPI schema into corresponding TypeScript interfaces, making it easier to validate requests against the defined schema.

Example command:
```sh
npx json-to-ts <path/to/openapi.json>
```

This command generates TypeScript types based on the provided JSON Schema from an OpenAPI spec.
x??

---

#### Using Zod for Runtime Validation in TypeScript
Background context: When working with TypeScript, you may want to define your types using runtime constructs and derive static types from those. Libraries like Zod facilitate this by allowing you to create schemas at runtime that can be used both for validation and type inference.
:p How does Zod help in defining request validation logic?
??x
Zod helps by allowing you to define a schema at runtime, which can then be used for validation as well as inferring static types. This approach eliminates the need for separate type definitions.

Example code:
```ts
import { z } from 'zod';

const createCommentSchema = z.object({
    postId: z.string(),
    title: z.string(),
    body: z.string(),
});

type CreateComment = z.infer<typeof createCommentSchema>;
```

In this example, `createCommentSchema` is a runtime value that can be used to validate requests. The type `CreateComment` is derived from the schema and can be used in your TypeScript code.
x??

---

#### Interoperability Issues with Runtime Types
Background context: While using runtime libraries like Zod for validation has many benefits, it also introduces some challenges. One of these challenges is ensuring that all types are defined consistently, especially if you need to reference external types or generate types from other sources.

:p What potential issue arises when using Zod and TypeScript together?
??x
When using Zod alongside TypeScript, there can be interoperability issues. For example, if a schema created with Zod needs to reference another type, that type will also have to be defined in the runtime system (using Zod), which may conflict with existing static types defined in TypeScript.

This can complicate integration with other parts of your codebase or external libraries.
x??

---

#### Benefits and Downsides of Using Runtime Validation Libraries
Background context: While using a runtime validation library like Zod has advantages, such as reducing redundancy between type definitions and schema validation logic, it also introduces some downsides. These include the need to learn new syntax, potential conflicts with existing TypeScript types, and difficulties in interoperating with other sources of truth.

:p What are the main benefits of using Zod for request validation?
??x
The main benefits of using Zod for request validation include:

1. **Reduced Duplication**: The schema defined in Zod can serve as both a static type and a runtime validator, eliminating redundancy.
2. **Easier Maintenance**: A single source of truth (the schema) simplifies maintenance since the same logic is used for both type inference and validation.

Example:
```ts
const createCommentSchema = z.object({
    postId: z.string(),
    title: z.string(),
    body: z.string(),
});

type CreateComment = z.infer<typeof createCommentSchema>;
```

This code snippet shows how Zod's schema can be used to infer static types and validate requests.
x??

---

#### Additional Considerations for Runtime Validation
Background context: When deciding between using a runtime validation library like Zod or defining types statically with TypeScript, there are several factors to consider. These include the complexity introduced by additional build steps, the need for learning new tools, and potential issues when integrating with other parts of your codebase.

:p What are some downsides of using a runtime validation library like Zod?
??x
Some downsides of using a runtime validation library like Zod include:

1. **Additional Complexity**: Adding another tool to the build process can increase complexity.
2. **Learning Curve**: Developers need to learn and understand both TypeScript type definitions and Zod schema definitions.
3. **Interoperability Issues**: If you need to reference external types or generate types from other sources, it may be challenging to integrate with existing systems.

Example:
```ts
const createCommentSchema = z.object({
    postId: z.string(),
    title: z.string(),
    body: z.string(),
});

type CreateComment = z.infer<typeof createCommentSchema>;
```

This example demonstrates the redundancy in defining types and validation logic when using Zod, which can complicate the codebase.
x??

---

---
#### Using Zod for Runtime Validation
Background context: A distinct runtime type validation system like Zod can express constraints that are hard to capture with TypeScript types, such as "a valid email address" or "an integer." This approach requires no additional build step and leverages TypeScript for all validations.
:p What is the advantage of using a tool like Zod for runtime validation?
??x
Using Zod for runtime validation allows you to express complex constraints that are difficult to capture with TypeScript types. It eliminates the need for an additional build step since everything can be done through TypeScript. Additionally, it tightens your iteration cycle if you expect frequent changes in schema.
```typescript
import z from 'zod';

const createCommentSchema = z.object({
  postId: z.string(),
  title: z.string(),
  body: z.string()
});

app.post('/comment', (request, response) => {
  const {body} = request;
  if (!createCommentSchema.safeParse(body).success) {
    return response.status(400).send('Invalid request');
  }
  // Proceed with the valid data
});
```
x?
---

---
#### JSON Schema Generation from TypeScript Types
Background context: You can generate runtime values from your TypeScript types by using a tool like `typescript-json-schema`. This approach allows you to reverse the previous section's approach, generating JSON Schema for your API types.
:p How can you use `typescript-json-schema` to generate JSON Schema for TypeScript interface?
??x
You can use the `typescript-json-schema` package to convert TypeScript interfaces into JSON Schema. Here’s how:

1. Define an interface in a file:
   ```typescript
   // api.ts
   export interface CreateComment {
     postId: string;
     title: string;
     body: string;
   }
   ```

2. Generate the JSON Schema from this type using `typescript-json-schema`:
   ```
   $ npx typescript-json-schema --schema api.ts '*' > api.schema.json
   ```

3. Load and validate the schema at runtime:

   ```typescript
   import Ajv from 'ajv';
   import apiSchema from './api.schema.json';
   import { CreateComment } from './api';

   const ajv = new Ajv();

   app.post('/comment', (request, response) => {
     const { body } = request;
     if (!ajv.validate(apiSchema.definitions.CreateComment, body)) {
       return response.status(400).send('Invalid request');
     }
     // Proceed with the valid data
   });
   ```

This method ensures that your API types and JSON Schema remain in sync.
x?
---

---
#### Choosing Between Different Validation Approaches
Background context: The choice of validation approach depends on whether you have another specification for your types, need to reference external TypeScript types, or prefer additional build steps. Each option has trade-offs.

:p Which approach is best if your types are already expressed in some other form like an OpenAPI schema?
??x
If your types are already expressed in some other form, such as an OpenAPI schema, using that as the source of truth for both your types and validation logic can be ideal. This reduces redundancy and ensures consistency.

However, you may need to use tools or frameworks that support this schema format (e.g., Swagger or Redoc). This approach incurs some tooling and process overhead but is worthwhile if it provides a single source of truth.
x?
---

---
#### TypeScript Types at Runtime
Background context: TypeScript types are erased before runtime. You can access them through additional tools like Zod, which uses TypeScript for validation but introduces a new build step.

:p Can you explain the concept of generating values from your TypeScript types using `typescript-json-schema`?
??x
Generating values from your TypeScript types using `typescript-json-schema` means converting your TypeScript interfaces into JSON Schema. This allows you to leverage all TypeScript tools and definitions while still performing validation at runtime.

Here’s a step-by-step guide:

1. Define an interface:
   ```typescript
   // api.ts
   export interface CreateComment {
     postId: string;
     title: string;
     body: string;
   }
   ```

2. Generate JSON Schema using `typescript-json-schema`:
   ```
   $ npx typescript-json-schema --schema api.ts '*' > api.schema.json
   ```

3. Use the generated schema for validation:
   ```typescript
   import Ajv from 'ajv';
   import apiSchema from './api.schema.json';

   const ajv = new Ajv();

   app.post('/comment', (request, response) => {
     const { body } = request;
     if (!ajv.validate(apiSchema.definitions.CreateComment, body)) {
       return response.status(400).send('Invalid request');
     }
     // Proceed with the valid data
   });
   ```

This approach ensures your API types and JSON Schema remain in sync without introducing a new build step.
x?
---

#### DOM Hierarchy and TypeScript
Background context: The text discusses how understanding the Document Object Model (DOM) hierarchy becomes crucial when using TypeScript, especially for web applications. It mentions that JavaScript and its APIs are deeply integrated with the DOM structure, making knowledge of these elements essential for debugging type errors and ensuring robust application development.

:p What is the significance of knowing the DOM hierarchy in a TypeScript context?
??x
Knowing the DOM hierarchy in TypeScript helps identify potential null values and ensures correct property usage. This knowledge prevents runtime errors by allowing developers to write more precise code that accounts for the possible types of elements they interact with, such as `Node`, `Element`, and `EventTarget`. For instance, `document.getElementById` can return an `Element` or `null`.
```typescript
const targetEl = document.getElementById('someId');
if (targetEl) {
    // Safe to use targetEl.classList.add('dragging') here.
}
```
x??

#### Event Target in TypeScript
Background context: The text highlights how the concept of `EventTarget` is crucial for type safety when dealing with events in web applications. `EventTarget` is a base interface that all event targets (like elements) implement, but it lacks specific properties and methods.

:p What is an `EventTarget` in TypeScript?
??x
An `EventTarget` is a base interface shared by objects that can receive DOM events. It does not include specific properties or methods like `classList`. When dealing with generic event handling, you should use `EventTarget`, but for more detailed operations (like adding/removing classes), you need to cast the `EventTarget` back to its specific type (e.g., `Element`).
```typescript
function handleDrag(eDown: Event) {
    const targetEl = eDown.currentTarget as HTMLElement;
    // Now safe to use targetEl.classList.
}
```
x??

#### Null Safety in TypeScript
Background context: The text points out that the TypeScript compiler flags null safety issues when dealing with DOM elements, emphasizing the importance of ensuring that variables are not `null` before using them.

:p Why does TypeScript flag errors related to null safety?
??x
TypeScript flags errors related to null safety because it aims to prevent potential runtime crashes by catching type-related issues at compile time. The compiler checks if a variable might be `null`, and if so, warns about operations that could fail if the variable is indeed `null`.

For example, the TypeScript compiler will flag an error when you try to call a method on a possibly null variable:
```typescript
const surfaceEl = document.getElementById('surface');
if (surfaceEl) {
    // Safe to use surfaceEl.addEventListener here.
}
```
x??

#### Element vs. EventTarget in Handling Events
Background context: The text emphasizes the difference between `Element` and `EventTarget`, highlighting that while `EventTarget` is a base interface, `Element` provides specific properties like `classList`.

:p What are the differences between `Element` and `EventTarget`?
??x
`Element` is a more specific type of `EventTarget` that extends it with additional properties and methods, such as `classList`. When you receive an event in a generic context, TypeScript will treat the current target as `EventTarget`, but for operations requiring element-specific functionality (like adding/removing classes), you need to cast the target to `Element`.

For example:
```typescript
const handleDrag = (eDown: Event) => {
    const targetEl = eDown.currentTarget as Element;
    targetEl.classList.add('dragging');
};
```
x??

--- 

These flashcards cover key concepts from the provided text, focusing on DOM hierarchy in TypeScript, understanding `EventTarget`, null safety, and the differences between `Element` and `EventTarget`.

#### EventTarget and Its Types
Background context: In JavaScript, `EventTarget` is a base interface for objects that can receive events. It provides methods to add, remove, or dispatch events but does not provide specific properties like `classList`. The hierarchy of DOM types includes `EventTarget`, which extends to `Node`, then to `Element`, and finally to more specific types like `HTMLElement`.
:p What is the significance of `EventTarget` in relation to event handling?
??x
`EventTarget` serves as a base interface for objects capable of receiving events, providing generic methods such as adding or removing event listeners. However, it does not provide properties specific to certain DOM elements.
x??

---

#### Type Hierarchy in the DOM
Background context: The DOM (Document Object Model) has a hierarchical structure where `EventTarget` is at the top, followed by `Node`, then `Element`, and finally more specific types like `HTMLElement`. This hierarchy defines the type relationships between different DOM objects.
:p What are the main types in the DOM hierarchy, and how do they relate to each other?
??x
The main types in the DOM hierarchy from most general to most specific are:
- `EventTarget`: Base interface for event handling.
  - `Node`: Can be elements or text nodes.
    - `Element`: HTML elements (like `<p>`, `<i>`).
      - `HTMLElement` and its subclasses: Specific HTML element types.

Example of type hierarchy:
```javascript
class EventTarget {}
class Node extends EventTarget {}
class Element extends Node {}
class HTMLElement extends Element {}
```
x??

---

#### CurrentTarget vs. Target in Event Handling
Background context: When handling events, the `currentTarget` property refers to the element that registered the event listener, while the `target` property points to the element where the event originated. These properties can be of different types due to the type hierarchy.
:p What are the differences between `currentTarget` and `target` in an event handler?
??x
- `currentTarget`: The element that is listening for the event (could be any type, like `Node`, `Element`, or even a `Window`).
- `target`: The element where the event actually occurred (will be of a more specific type, such as `HTMLElement`).

For example:
```javascript
function handleDrag(eDown) {
    const targetEl = eDown.currentTarget;
    // Could be any Node type like window or XMLHttpRequest.
    console.log(targetEl);
    
    const origEl = eDown.target;
    // More likely to be an HTMLElement, but still could vary based on the event.
    console.log(origEl);
}
```
x??

---

#### `classList` Property and EventTarget
Background context: The `classList` property is a specific feature of `HTMLElement`s, allowing for manipulation of class names. However, since `EventTarget` does not have this property, TypeScript will flag any attempt to use it on an `EventTarget`.
:p Why can't we directly access `classList` using `currentTarget` in an event handler?
??x
You cannot directly access `classList` from `currentTarget` because `currentTarget` is of type `EventTarget`, which does not have a `classList` property. This property is only available on more specific types like `HTMLElement`.

For instance:
```javascript
function handleDrag(eDown: Event) {
    const targetEl = eDown.currentTarget;
    // TypeScript error: Property 'classList' does not exist on type 'EventTarget'.
    targetEl.classList.add('dragging');
}
```
x??

---

#### `children` and `childNodes` Properties in DOM Nodes
Background context: The `children` property returns an array-like collection of child elements, while `childNodes` includes all child nodes (elements, text, comments, etc.). These properties are useful for traversing the DOM tree.
:p How do `children` and `childNodes` differ when used to traverse a DOM node?
??x
- `children`: Returns only direct element children in an array-like collection.
  - Example:
    ```javascript
    const p = document.querySelector('p');
    console.log(p.children); // HTMLCollection [i]
    ```

- `childNodes`: Includes all child nodes, including text and comment nodes.
  - Example:
    ```javascript
    const p = document.querySelector('p');
    console.log(p.childNodes); // NodeList (5) [text, i, text, comment, text]
    ```
x??

---

#### Differences Between `Element` and `HTMLElement`
Background context: While both are part of the DOM hierarchy, `Element` is a more general type that includes non-HTML elements like SVG tags. `HTMLElement` specifically refers to HTML elements.
:p What distinguishes an `Element` from an `HTMLElement`?
??x
An `Element` can be any node in the DOM tree (e.g., `<div>`, `<span>`), while an `HTMLElement` is a specific type of `Element` that corresponds to standard HTML elements.

For example:
- An `HTMLParagraphElement` and `HTMLHeadingElement` are subclasses of `HTMLElement`.
- An `SVGElement` is part of the SVG hierarchy but still falls under the broader `Element` category.
x??

---

#### Specificity of DOM Property Types
TypeScript's type declarations for the DOM make liberal use of literal types to provide the most specific type possible. However, this is not always achievable with certain methods like `document.getElementById`.

:p What are the implications when using `document.getElementById` in TypeScript?
??x
When using `document.getElementById`, you might receive a more general type such as `HTMLElement | null`. This happens because `getElementById` can return any element or null. To get a specific type, you may need to perform a runtime check or use a type assertion.

```typescript
const div = document.getElementById('my-div');
if (div instanceof HTMLDivElement) {
    console.log(div); // const div: HTMLDivElement
}
```

x??

---
#### Handling Nullable Returns from DOM Methods with Type Assertions
With `document.getElementById`, the return value can be `null`. To handle this, you might need to use a type assertion or an if statement.

:p How should you handle null values returned by `getElementById` in TypeScript?
??x
To handle null values returned by `document.getElementById`, you can use a type assertion or an if statement. A type assertion is appropriate when you are certain about the element type, while an if statement is necessary to check for null.

```typescript
const div = document.getElementById('my-div') as HTMLDivElement; // Type assertion
// OR

const div = document.getElementById('my-div');  // Nullable return
if (div) {
    console.log(div); // const div: HTMLDivElement
}
```

x??

---
#### Event Hierarchy in TypeScript
The `Event` type is the most generic, while more specific types like `MouseEvent`, `UIEvent`, etc., exist. The `clientX` and `clientY` properties are available only on specific event types.

:p What causes the error when trying to use `clientX` and `clientY` in an event handler?
??x
The error occurs because the events are declared as `Event`, which does not include `clientX` and `clientY`. These properties exist only on more specific event types such as `MouseEvent`.

To resolve this, you can either inline the event handler to provide context or explicitly declare the event parameter type.

```typescript
function handleDrag(eDown: Event) {
    // ... const dragStart = [eDown.clientX, eDown.clientY]; // Error here

    // Solution 1: Inline the mousedown handler for better context
    el.addEventListener('mousedown', (eDown: MouseEvent) => { 
        const dragStart = [eDown.clientX, eDown.clientY];
        // ...
    });

    // Solution 2: Explicitly declare event type
    function addDragHandler(el: HTMLElement) {
        el.addEventListener('mousedown', (eDown: MouseEvent) => {  
            const dragStart = [eDown.clientX, eDown.clientY];
            // ...
        });
    }
}
```

x??

---
#### Non-Null Assertions and Conditional Checks for DOM Elements
When dealing with potentially null returns from `getElementById`, it is important to account for the possibility of receiving a null value. Use non-null assertions or conditional checks.

:p How can you ensure your code handles elements that might not exist in the document?
??x
To handle elements that might not exist, use a non-null assertion if you are certain they will always be present, or include an if statement to check for `null`.

```typescript
const surfaceEl = document.getElementById('surface');
if (surfaceEl) { 
    addDragHandler(surfaceEl); // If the element exists, proceed with handling
} else {
    console.log("Element not found"); // Handle case where element is missing
}
```

Or using a non-null assertion:

```typescript
const surfaceEl = document.getElementById('surface')!;
addDragHandler(surfaceEl);
// This will result in an error if the element does not exist, so use with caution.
```

x??

---

