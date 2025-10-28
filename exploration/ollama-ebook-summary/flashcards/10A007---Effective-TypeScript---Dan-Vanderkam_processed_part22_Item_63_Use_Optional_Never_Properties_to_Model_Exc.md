# Flashcards: 10A007---Effective-TypeScript---Dan-Vanderkam_processed (Part 22)

**Starting Chapter:** Item 63 Use Optional Never Properties to Model Exclusive Or

---

#### Exclusive Or (XOR) in TypeScript
Background context: In TypeScript, the inclusive OR (`|`) operator can sometimes lead to unexpected behaviors when modeling exclusive conditions. The traditional understanding of "or" as an exclusive operation is common in ordinary speech but not always reflected in programming languages like JavaScript and TypeScript.

:p How does TypeScript handle the `|` (inclusive OR) operator?
??x
TypeScript treats the `|` operator as inclusive, meaning a type can be both types on either side. This can lead to unexpected results when modeling exclusive conditions.
```typescript
interface ThingOne {
    shirtColor: string;
}

interface ThingTwo {
    hairColor: string;
}

type Thing = ThingOne | ThingTwo;

const bothThings = { 
    shirtColor: 'red', 
    hairColor: 'blue' 
};

const thing1: ThingOne = bothThings;  // ok
const thing2: ThingTwo = bothThings;  // ok
```
x??

---

#### Using Optional `never` to Model Exclusive Or
Background context: To model exclusive OR (XOR) in TypeScript, you can use optional properties with the type `never`. This ensures that a property cannot be present on an object of the resulting union type.

:p How does using optional never (`?: never`) help model exclusive OR?
??x
By making one of the properties optional and assigning it `never`, you force the type to allow only objects that have exactly one of the properties. This effectively models exclusive OR.
```typescript
interface OnlyThingOne {
    shirtColor: string;
    hairColor?: never;
}

interface OnlyThingTwo {
    hairColor: string;
    shirtColor?: never;
}

type ExclusiveThing = OnlyThingOne | OnlyThingTwo;

const thing1: OnlyThingOne = bothThings;  // Error, incompatible property 'hairColor'
const thing2: OnlyThingTwo = bothThings;  // Error, incompatible property 'shirtColor'
```
x??

---

#### Applying `never` to Other Types
Background context: The concept of using optional never can be applied beyond just modeling exclusive OR. It can also be used to disallow certain properties in structural types.

:p How can you use the `never` type to model a 2D vector?
??x
By adding an optional property with the type `never`, you can ensure that objects only have properties valid for 2D vectors, preventing the inclusion of extraneous properties like `z`.
```typescript
interface Vector2D {
    x: number;
    y: number;
    z?: never;
}

const v = { 
    x: 3, 
    y: 4, 
    z: 5 // Error due to z property being optional with never type
};

function norm(v: Vector2D) {
    return Math.sqrt(v.x ** 2 + v.y ** 2);
}
```
x??

---

#### Tagged Unions for Exclusive Or
Background context: Another approach to modeling exclusive OR is using tagged unions. This involves adding a discriminator property that ensures only one variant can be present in an object.

:p How does using tagged unions help model exclusive OR?
??x
Tagged unions use a unique identifier (discriminator) to ensure that objects are of exactly one type within the union. In this case, `type` acts as the discriminator, making it impossible for both types to coexist on the same object.
```typescript
interface ThingOneTag {
    type: 'one';
    shirtColor: string;
}

interface ThingTwoTag {
    type: 'two';
    hairColor: string;
}

type Thing = ThingOneTag | ThingTwoTag;

const thing1: ThingOneTag = { 
    type: 'one', 
    shirtColor: 'red' 
};

const thing2: ThingTwoTag = { 
    type: 'two', 
    hairColor: 'blue' 
};
```
x??

---

#### XOR Helper Type
Background context: For more complex scenarios, a generic helper type `XOR` can be defined to automatically generate types that model exclusive OR based on the given interface definitions.

:p How does the `XOR` type work?
??x
The `XOR` type uses intersection and exclusion patterns to create union types where each type in the resulting union only has properties from one of the input interfaces, ensuring they are mutually exclusive.
```typescript
type XOR<T1, T2> = 
    (T1 & {[k in Exclude<keyof T2, keyof T1>]?: never}) | 
    (T2 & {[k in Exclude<keyof T1, keyof T2>]?: never});

type ExclusiveThing = XOR<ThingOne, ThingTwo>;

const allThings: ExclusiveThing = {  // Error due to incompatible property 'hairColor'
    shirtColor: 'red', 
    hairColor: 'blue' 
};
```
x??

#### Nominal Typing vs Structural Typing
In TypeScript, nominal typing differs from structural typing. In structural typing, a type is determined by its shape (properties and methods). For example:
```typescript
interface Vector2D {
    x: number;
    y: number;
}

function calculateNorm(p: Vector2D) {
    return Math.sqrt(p.x ** 2 + p.y ** 2);
}
```
Here, `calculateNorm` accepts any object with `x` and `y` properties.
:p What is the difference between nominal typing and structural typing in TypeScript?
??x
Nominal typing defines a type by its name or tag, rather than its shape. For example:
```typescript
interface Vector2D {
    type: '2d';
    x: number;
    y: number;
}
```
In this case, `Vector2D` is specifically defined as having a `type` property with the value `'2d'`. This makes it distinct from other objects that may have similar shapes but different types.
x??

---

#### Branding for Nominal Typing
Branding can be used to give otherwise structurally identical types a unique identity. For example, distinguishing between absolute and relative paths:
```typescript
type AbsolutePath = string & {_brand: 'abs'};
function isAbsolutePath(path: string): path is AbsolutePath {
    return path.startsWith('/');
}
```
:p How does branding help in distinguishing similar types?
??x
Branding helps by adding a specific tag to the type, making it distinct from other types with the same structure. In this case, `AbsolutePath` has an additional `_brand` property set to `'abs'`, which makes it clear that it is an absolute path.
x??

---

#### Tagged Unions and Runtime Overhead
Tagged unions are used when you want to enforce exclusive typing in TypeScript:
```typescript
interface Vector2D {
    type: '2d';
    x: number;
    y: number;
}
```
Here, `Vector2D` is tagged with a `type` property of `'2d'`. This makes it an "explicit tag" that enforces the exclusive nature of this type.
:p What are some downsides of using explicit tagging in TypeScript?
??x
The main downside of explicit tagging is runtime overhead. It transforms simple types into more complex ones, adding a string `type` property for each variant. Additionally, tagged unions only work with object types, not primitive types like numbers or strings.
x??

---

#### Using Brands to Brand Primitive Types
Branding can also be used with primitive types:
```typescript
type Meters = number & {_brand: 'meters'};
type Seconds = number & {_brand: 'seconds'};

const meters = (m: number) => m as Meters;
const seconds = (s: number) => s as Seconds;

const oneKm = meters(1000); // oneKm is a number with the brand 'meters'
const oneMin = seconds(60); // oneMin is a number with the brand 'seconds'
```
:p How can brands be used to document types of numeric parameters?
??x
Brands can help document that certain numbers have specific units, such as meters or seconds. This approach avoids runtime overhead and can improve type safety by ensuring that operations involving mixed units are not allowed.
x??

---

#### Using Unique Symbols for Branding
Unique symbols can also be used to brand types:
```typescript
declare const brand: unique symbol;
export type Meters = number & {[brand]: 'meters'};
```
:p What is the advantage of using a unique symbol for branding?
??x
The advantage is that since the `brand` symbol is not exported, users cannot create new types compatible with it directly. This prevents accidental mixing of branded and unbranded types.
x??

---

#### Branding Lists to Enforce Order
Branding can be used to ensure lists are sorted:
```typescript
type SortedList<T> = T[] & {_brand: 'sorted'};
function isSorted<T>(xs: T[]): xs is SortedList<T> {
    for (let i = 0; i < xs.length - 1; i++) {
        if (xs[i] > xs[i + 1]) return false;
    }
    return true;
}
```
:p How does branding help in ensuring lists are sorted?
??x
Branding helps by requiring explicit proof that a list is sorted before using operations that depend on this property. This ensures type safety and correctness, as unsorted lists may not behave as expected.
x??

---

#### Understanding npm Dependencies in TypeScript
Background context: In JavaScript, npm is a package manager that handles both the libraries and dependencies required for projects. It categorizes these into different types based on their usage—dependencies, devDependencies, and peerDependencies.

:p What are the main categories of dependencies managed by npm?
??x
- `dependencies`: Packages necessary to run your application.
- `devDependencies`: Packages needed during development (like test frameworks) but not at runtime.
- `peerDependencies`: Packages that must be installed separately but should be compatible with each other, such as libraries and their type declarations.

x??

---
#### TypeScript as a Development Dependency
Background context: Since TypeScript is used for development purposes and its types do not exist at runtime, it's advisable to treat it as a devDependency. This ensures consistent versioning across team members and projects.

:p Why should TypeScript be treated as a devDependency?
??x
Because TypeScript is a development tool that gets transpiled into JavaScript before execution. Installing it globally can lead to discrepancies in versions among different developers or environments, whereas making it a devDependency ensures everyone uses the same version when running `npm install`.

x??

---
#### Managing Type Definitions with @types
Background context: For libraries without built-in TypeScript types, you might find type definitions on DefinitelyTyped under the `@types` scope. These packages only contain the necessary type information and not the actual implementation.

:p How should type declarations be managed for a library like React?
??x
You should install both the library and its corresponding @types package as devDependencies. For example, to use React along with its TypeScript types, you would run:
```
npm install react --save
npm install --save-dev @types/react
```

This ensures that the type definitions are available for development but not included in the final production bundle.

x??

---
#### Separating Development and Production Dependencies
Background context: Even if you're building a web app that won't be published, separating devDependencies from dependencies still offers benefits. This practice helps maintain clarity and manage project dependencies more effectively.

:p Why is it beneficial to keep @types as devDependencies even for non-publishable projects?
??x
Keeping `@types` as devDependencies ensures that type definitions are available during development without cluttering the final production build. It also promotes better separation of concerns, making it easier to manage different types of dependencies and ensuring consistency across team members.

For example:
```json
{
  "devDependencies": {
    "@types/react": "^18.2.23",
    "typescript": "^5.2.2"
  },
  "dependencies": {
    "react": "^18.2.0"
  }
}
```

x??

---
#### Running TSC Using npx
Background context: `npx` is a command-line utility that allows you to run npm packages without installing them globally. It's useful for running tools like TypeScript’s compiler (tsc) in the same way as other npm scripts.

:p How can you use `npx` to run the tsc command?
??x
You can use `npx` to run the TypeScript compiler installed by npm directly from the command line, without needing a global installation. For example:
```
$ npx tsc
```

This ensures that you're using the version of tsc that is compatible with your project's dependencies.

x??

---

