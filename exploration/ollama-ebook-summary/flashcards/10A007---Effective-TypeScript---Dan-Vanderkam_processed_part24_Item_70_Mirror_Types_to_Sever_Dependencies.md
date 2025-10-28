# Flashcards: 10A007---Effective-TypeScript---Dan-Vanderkam_processed (Part 24)

**Starting Chapter:** Item 70 Mirror Types to Sever Dependencies

---

#### Understanding Dynamic `this` Binding in Callbacks
Dynamic `this` binding is a common issue when dealing with callbacks, especially in JavaScript. The context of `this` changes based on how and where the function is called. It’s important to provide an explicit type for `this` if it's part of your API.
:p How does dynamic `this` binding work in JavaScript?
??x
Dynamic `this` binding works by determining the value of `this` at runtime, depending on the context in which a function is called. For example, inside a method of an object, `this` refers to that object. However, when you pass a method as a callback and call it outside its original context, `this` might refer to something else.
```javascript
const obj = {
  value: 'original',
  logValue() {
    console.log(this.value);
  }
};

obj.logValue(); // logs "original"
obj.logValue.call({value: 'called'}); // logs "called"
```
x??

---

#### Void Dynamic `this` Binding in New APIs
In newer JavaScript APIs, it’s good practice to use a void dynamic `this` binding. This means that the context of `this` should not matter or be dynamically bound.
:p Why is it important to have a void dynamic `this` binding?
??x
Having a void dynamic `this` binding ensures that your function works correctly regardless of how and where it’s called. It makes your functions more predictable, especially when used as callbacks or in functional programming contexts.
```javascript
function example() {
  return this.someValue; // This is problematic because of possible `this` context changes
}

const obj = { someValue: 'value' };
example.call(obj); // Works fine
example(); // May not work as expected if the context of `this` matters
```
x??

---

#### Mirroring Types to Serve Dependencies for CSV Parsing Library
When creating a library like one that parses CSV files, you need to consider how your type definitions interact with users' dependencies.
:p How does the use of `@types/node` affect your library's dependencies?
??x
Using `@types/node` as a devDependency means it’s not installed when someone installs your package. This can cause issues for TypeScript web developers who might see errors like "Cannot find module 'node:buffer'". To solve this, you should provide your own simplified type definitions.
```typescript
export interface CsvBuffer {
  toString(encoding?: string): string;
}
```
This approach ensures that the library works correctly without requiring `@types/node` for users, but it also means you need to write tests to ensure compatibility with real `Buffer`.
x??

---

#### Structural Typing and Simplified Interfaces
Structural typing allows TypeScript to infer types based on object shape rather than name. This can be useful when dealing with interfaces that are only used in a specific context.
:p How does structural typing help in the CSV library example?
??x
Structural typing helps by allowing you to create simpler, more focused type definitions without relying heavily on large external libraries like `@types/node`. You can define your own interface based on the actual methods and properties needed.
```typescript
export interface StringEncodable {
  toString(encoding?: string): string;
}

// Example test case
import { Buffer } from 'node:buffer';
import { parseCSV } from './parse-csv';

test('parse CSV in a buffer', () => {
  expect(parseCSV(new Buffer("column1,column2 val1,val2", "utf-8"))).toEqual([
    { column1: 'val1', column2: 'val2' }
  ]);
});
```
This test ensures that both the runtime behavior and type compatibility are correct.
x??

---

#### Avoid Transitive Type Dependencies in Published npm Modules
Background context: This concept is about ensuring that when you publish an npm module, it does not depend on transitive @types dependencies. This helps maintain a good experience for all TypeScript developers and can improve compiler performance by severing unnecessary dependencies.

:p Why should published npm modules avoid transitive type dependencies?
??x
Published npm modules should avoid transitive type dependencies because they can negatively impact the development experience for users who might not need or want those dependencies. By removing them, you provide a cleaner and more efficient setup for your module's consumers.
x??

---

#### Use Module Augmentation to Improve Types
Background context: This concept involves using TypeScript interfaces to modify existing declarations in libraries. This is useful for making built-in types stricter and more precise, or for disallowing certain constructs that might be problematic.

:p How can you use interface augmentation to improve the return type of JSON.parse?
??x
You can use interface augmentation to change the return type of `JSON.parse` from `any` to `unknown`, ensuring better type safety. By defining your own `JSON` interface and merging it with the library declarations, TypeScript will prioritize your overload.

```typescript
// declarations/safe-json.d.ts
interface JSON {
    parse(
        text: string,
        reviver?: (this: any, key: string, value: any) => any
    ): unknown;
}
```
x??

---

#### Ban Problematic Constructs Using Declaration Merging
Background context: This concept explains how to ban certain problematic constructs like the `Set` constructor that accepts a string by using interface augmentation. This helps prevent bugs in your code and provides better type safety.

:p How can you ban the use of `new Set('abc')`?
??x
You can ban the use of `new Set('abc')` by overriding the `SetConstructor` interface to return `void` when called with a string argument. This way, attempting to construct a `Set` from a string will result in a void type, indicating an error.

```typescript
// declarations/ban-set-string-constructor.d.ts
interface SetConstructor {
    new (str: string): void;
}
```
x??

---

#### Use Overloads for Better Type Safety
Background context: This concept involves using function overloads to provide more specific and safer types for certain functions, such as `JSON.parse` and `Response.prototype.json()`.

:p How can you use overloads to improve the type safety of JSON.parse?
??x
You can use overloads to change the return type of `JSON.parse` from `any` to `unknown`. This ensures that when you parse a JSON string, TypeScript will understand the structure better and provide more accurate type information.

```typescript
// declarations/safe-json.d.ts
interface JSON {
    parse(
        text: string,
        reviver?: (this: any, key: string, value: any) => any
    ): unknown;
}
```
x??

---

#### Declaration Merging in TypeScript
Background context: This concept describes how declaration merging works in TypeScript. Interface declarations from different sources can be merged to form a final result, which is useful for making local modifications without affecting the global namespace.

:p How does declaration merging work with interface augmentation?
??x
Declaration merging allows you to define interfaces or type aliases that will merge with existing library declarations. This means if you have an interface with the same name as one in a library, TypeScript will combine them into a single definition.

```typescript
// Example of merging interfaces
interface JSON {
    parse(
        text: string,
        reviver?: (this: any, key: string, value: any) => any
    ): unknown;
}

declare var JSON: JSON; // The `JSON` interface is now merged with the library declaration.
```
x??

---

#### Type Safety with Unknown vs Any
Background context: This concept explains why using `unknown` instead of `any` can provide better type safety, as `unknown` requires explicit assertions to access its properties.

:p Why should you use unknown over any for safer types?
??x
Using `unknown` provides better type safety because it requires explicit type assertions when accessing properties. This avoids silent type errors and makes your code more robust against potential bugs.

```typescript
const response = JSON.parse(apiResponse) as ApiResponse;
const cacheExpirationTime = response.lastModified + 3600; // Type assertion ensures type safety.
```
x??

---

#### Marking Deprecated Constructs with @deprecated
Background context: This concept involves marking certain constructs as deprecated to give users a clear warning and hint that the feature is no longer recommended.

:p How can you mark a function or method as deprecated using TypeScript?
??x
You can mark a function or method as deprecated by adding the `@deprecated` comment above its declaration. This will display the deprecation message in editors, guiding users to avoid using it.

```typescript
interface SetConstructor {
    /** @deprecated */
    new (str: string): void;
}
```
x??

---

#### TypeScript vs. ECMAScript Features
Background context: The relationship between TypeScript and JavaScript has evolved over time, with TC39 adding many features to core JavaScript that are not compatible with early versions of TypeScript. This led TypeScript to adopt a principle where it innovates solely in the type space while TC39 defines the runtime.

:p What is the primary reason TypeScript chose to innovate only in the type space?
??x
TypeScript chose this approach because adopting new features from the standard would have required maintaining existing code, which was awkward. By focusing on types, they ensure compatibility with alternative TypeScript compilers and maintain alignment with future standards.
x??

---
#### Enums in TypeScript
Background context: TypeScript adds enums to JavaScript for type safety and clarity. However, there are different variants of enums that behave differently.

:p What is the key difference between a regular enum and a const enum in TypeScript?
??x
A regular enum retains its enum structure at runtime, whereas a const enum is completely rewritten by the compiler into numeric values or removed entirely. For example, using `const enum` can change how the enum members are used at runtime.
x??

---
#### Number-valued Enums
Background context: In TypeScript, enums can be number-valued, which means they use numbers to represent their values. This variant is not very safe because the number type can be assigned to it.

:p Why might you choose a number-valued enum over other types?
??x
You might choose a number-valued enum when you need to maintain compatibility with existing code that uses numbers or bit flags, as TypeScript allows numeric enums.
x??

---
#### String-valued Enums
Background context: String-valued enums provide type safety and more informative values at runtime. However, they are not structurally typed like other types in TypeScript.

:p What is a key characteristic of string-valued enums?
??x
String-valued enums offer type safety and informative values but do not use structural typing for assignability, unlike other types in TypeScript.
x??

---
#### const Enum with preserveConstEnums Flag
Background context: The `const enum` can behave differently depending on the presence or absence of the `preserveConstEnums` flag. When this flag is set, the `const enum` emits runtime code similar to a regular enum.

:p What happens when you use `const enum` without setting the `preserveConstEnums` flag?
??x
Without `preserveConstEnums`, `const enum` members are completely removed at runtime, and using them would result in an error. This is different from how regular enums behave.
x??

---
#### Nominally Typed Enums
Background context: String-valued enums in TypeScript are nominally typed, meaning they enforce the exact type of the values assigned to them.

:p Why might this cause issues when publishing a library?
??x
When publishing a library that uses string-valued enums, users may encounter type mismatches if they use strings that aren't exactly the same as defined in your enum. This can lead to runtime errors or unexpected behavior.
x??

---
#### Summary of TypeScript Enums
Background context: TypeScript provides several variants of enums with different behaviors and characteristics.

:p What are the key differences between number-valued, string-valued, and const enum variants?
??x
Number-valued enums allow numeric values but lack type safety. String-valued enums provide better type safety and more informative values at runtime but are not structurally typed. Const enums can be completely removed or preserved depending on the `preserveConstEnums` flag.
x??

---

#### Using Literal Types for Enums

Background context: In TypeScript, enums provide a way to define named constants. However, using string literals can offer better type safety and translate more directly to JavaScript. This is particularly useful when you have both JavaScript and TypeScript users interacting with your code.

:p Why might TypeScript developers prefer literal types over numeric or string enums?
??x
Literal types in TypeScript offer strong type checking at the compile-time level, which means that any invalid value will be caught before runtime. For example, using `type Flavor = 'vanilla' | 'chocolate' | 'strawberry';` ensures that only valid flavors can be assigned to a variable of this type. This is beneficial for both TypeScript and JavaScript users since string literals directly translate to strings in JavaScript.

Code Example:
```typescript
type Flavor = 'vanilla' | 'chocolate' | 'strawberry';
let favoriteFlavor: Flavor = 'chocolate';  // OK
favoriteFlavor = 'americone dream';        // ~~~~ Type '"americone dream"' is not assignable to type 'Flavor'
```
x??

---

#### Parameter Properties in TypeScript

Background context: Parameter properties allow you to assign a constructor parameter directly to a class property. This can make the code more concise and readable.

:p What are parameter properties, and how do they work?
??x
Parameter properties enable you to initialize class properties from constructor parameters directly. For example, instead of writing `constructor(name: string) { this.name = name; }`, you can use `constructor(public name: string) {}`.

Code Example:
```typescript
class Person {
    // Traditional way
    constructor(name: string) {
        this.name = name;
    }

    // Using parameter property
    constructor(public name: string) {}
}
```

The `public` keyword is optional and can be omitted, but it makes the intention clear that the property should be public. The generated JavaScript code will look like:
```javascript
var Person = (function () {
    function Person(name) {
        this.name = name;
    }
    return Person;
})();
```
x??

---

#### Complications with Parameter Properties

Background context: While parameter properties can make your TypeScript classes more concise, they also introduce some challenges. These include potential issues with type checking and class design clarity.

:p What are the downsides of using parameter properties in a class?
??x
One downside is that parameter properties can sometimes lead to misleading code structure, especially if there are many parameters or non-parameter properties. This can make it difficult to understand the actual state of an object from its declaration.

Code Example:
```typescript
class Person {
    first: string;
    last: string;

    constructor(public name: string) {
        [this.first, this.last] = name.split(' ');
    }
}

// The class has three properties (first, last, name), but it's hard to read off the code.
```

Another issue is that parameter properties are compiled into regular class properties, which can result in unused parameters in the generated JavaScript. This can be confusing when debugging or maintaining the code.

x??

#### Triple-Slash Imports and Namespaces
Background context explaining how JavaScript lacked an official module system before ECMAScript 2015. Different environments like Node.js and AMD used their own methods, while TypeScript introduced its own module system using a `module` keyword and triple-slash imports.
:p What is the role of triple-slash imports in TypeScript?
??x
Triple-slash imports were used to include type declaration files or external modules before ECMAScript 2015. However, they are now considered historical and should not be used in modern code unless necessary for compatibility with older libraries.

```typescript
/// <reference path="other.ts" />
foo.bar();
```
x??

---

#### Decorators in TypeScript
Background context explaining how decorators were added to support Angular but have since evolved into a more standardized form. Early decorators required the `--experimentalDecorators` flag, while modern decorators can be used directly without any flags.

:p What is the difference between experimental and standard decorators in TypeScript?
??x
Experimental decorators (enabled with the `--experimentalDecorators` flag) are non-standard and were primarily designed to support Angular before the official decorators proposal was finalized. Standard decorators can now be used directly in your code without any flags, following the latest ECMAScript standards.

```typescript
// Experimental decorator
function logged(originalFn: any, context: ClassMethodDecoratorContext) {
    return function(this: any, ...args: any[]) {
        console.log(`Calling ${String(context.name)}`);
        return originalFn.call(this, ...args);
    };
}

class Greeter {
    greeting: string;
    constructor(message: string) {
        this.greeting = message;
    }
    @logged
    greet() {
        return `Hello, ${this.greeting}`;
    }
}

console.log(new Greeter('Dave').greet()); // Logs: Calling greet Hello, Dave

// Standard decorator
class GreeterStandard {
    greeting: string;
    constructor(message: string) {
        this.greeting = message;
    }
    @logged
    greet() {
        return `Hello, ${this.greeting}`;
    }
}
```
x??

---

#### Module System in TypeScript
Background context explaining how the module system was introduced to fill the gap left by JavaScript before ECMAScript 2015. The module keyword and triple-slash imports were used to manage code organization and avoid naming conflicts.

:p How did modules work in older versions of TypeScript?
??x
In older versions of TypeScript, the `module` keyword was used alongside triple-slash imports to create a modular structure for code. The `namespace` keyword served as a synonym for `module` to maintain compatibility with other module systems. However, these are now considered historical and should not be used in modern TypeScript projects.

```typescript
// older module usage
namespace foo {
    export function bar() {}
}

// newer import/export style
import { bar } from './other';
bar();
```
x??

---

#### Using ECMAScript 2015 Modules
Background context explaining that after the official module system was introduced in ECMAScript 2015, TypeScript added support for `import` and `export`. This should be used instead of older triple-slash imports or custom modules.

:p Why should you use ECMAScript 2015-style modules over old methods?
??x
ECMAScript 2015 introduced a standardized module system with `import` and `export`, which is now the recommended way to manage code in modern JavaScript applications. Using these features ensures compatibility, maintainability, and adherence to industry standards.

```typescript
// ECMAScript 2015 style modules
import { bar } from './other';

class Greeter {
    greeting: string;
    constructor(message: string) {
        this.greeting = message;
    }
    greet() {
        return `Hello, ${this.greeting}`;
    }
}

console.log(new Greeter('Dave').greet()); // No decorator usage here
```
x??

---

#### Avoiding Decorators in Code
Background context explaining that while decorators can be useful for certain tasks, they may make code harder to follow and should be used judiciously. Specifically, avoid changing method type signatures with decorators.

:p How should you use decorators in your TypeScript projects?
??x
Decorators are a powerful feature but should be used cautiously. Avoid writing nonstandard decorators or modifying method type signatures with them. Stick to standard decorators when possible to maintain clarity and compatibility.

```typescript
class Greeter {
    greeting: string;
    constructor(message: string) {
        this.greeting = message;
    }
    // @logged - this is a standard decorator usage
    greet() {
        return `Hello, ${this.greeting}`;
    }
}

// Avoid this kind of usage:
function logged(originalFn: any, context: ClassMethodDecoratorContext) {
    return function(this: any, ...args: any[]) {
        console.log(`Calling ${String(context.name)}`);
        return originalFn.call(this, ...args);
    };
}
```
x??

---
#### JavaScript's Private Fields Before ES2022
Background context explaining the concept. The provided text discusses how JavaScript historically lacked a way to enforce private properties and methods, leading to workarounds like prefixing variables with underscores.

:p What were the limitations of using an underscore prefix as a workaround for making fields private in JavaScript?
??x
The limitation was that this approach only discouraged users from accessing "private" data but did not provide any real enforcement. Users could still access the properties by circumventing the convention, such as through type assertions or iteration.

```javascript
class Foo {
    _private = 'secret123';
}

const f = new Foo();
f._private; // 'secret123'
```
x??

---
#### TypeScript's Private Fields
Background context explaining the concept. The text describes how TypeScript introduced public, protected, and private field visibility modifiers, but these are more of a type system feature rather than actual enforcement mechanisms.

:p How do TypeScript's private fields work in terms of enforcement?
??x
TypeScript's private fields provide some level of discouragement for accessing private data through features of the type system. However, at runtime, they behave similarly to JavaScript properties without any enforced privacy. TypeScript's private fields still rely on conventions and can be accessed using methods like type assertions.

```typescript
class Diary {
    private secret = 'cheated on my English test';
}

const diary = new Diary();
(diary as any).secret; // Still accessible with a type assertion
```
x??

---
#### ES2022 Private Fields in JavaScript
Background context explaining the concept. The text highlights that ES2022 introduced support for private fields, which are enforced both at type-checking and runtime levels.

:p What is a key difference between TypeScript's private fields and ES2022 private fields?
??x
The key difference is that ES2022 private fields are standard and widely supported, providing real enforcement of privacy. They are not just part of the type system but also prevent access from outside the class at runtime.

```javascript
class PasswordChecker {
    #passwordHash: number;
    constructor(passwordHash: number) {
        this.#passwordHash = passwordHash;
    }

    checkPassword(password: string) {
        return hash(password) === this.#passwordHash;
    }
}

const checker = new PasswordChecker(hash('s3cret'));
checker.#passwordHash; // Error: Property '#passwordHash' is not accessible
```
x??

---
#### Public and Protected in JavaScript (and TypeScript)
Background context explaining the concept. The text explains that public fields are the default visibility, so no annotation is needed, while protected implies inheritance.

:p Why are practical uses of `protected` quite rare according to the text?
??x
Practical uses of `protected` are quite rare because the general rule in object-oriented programming is to prefer composition over inheritance. This means that instead of inheriting from a class, developers often opt for more flexible and modular design patterns.

```javascript
class Parent {
    protected data = 'protected data';
}

class Child extends Parent {
    // Can access `data` but not encouraged due to preference for composition.
}
```
x??

---
#### Readonly as a Field Modifier in TypeScript
Background context explaining the concept. The text mentions that `readonly` is a type-level construct and can be used alongside private fields.

:p How can you use `readonly` with private fields?
??x
You can use `readonly` to enforce immutability at the type level, ensuring that once a private field is set, it cannot be changed. This is useful for maintaining invariants within a class.

```typescript
class ReadOnlyPasswordChecker {
    #passwordHash: readonly number;
    constructor(passwordHash: number) {
        this.#passwordHash = passwordHash; // Immutable after construction
    }
}
```
x??

---
#### Member Visibility Modifiers
Background context explaining the concept. The text discusses various member visibility modifiers in JavaScript, TypeScript, and ES2022.

:p What should you use instead of TypeScript's private fields for better security?
??x
For better security, you should use ES2022 private fields, as they are standard, widely supported, and provide real enforcement at both type-checking and runtime levels. This ensures that properties marked with `#` cannot be accessed from outside the class.

```javascript
class Diary {
    #secret = 'cheated on my English test';
}

const diary = new Diary();
diary.#secret; // Error: Property '#secret' is not accessible
```
x??

---

