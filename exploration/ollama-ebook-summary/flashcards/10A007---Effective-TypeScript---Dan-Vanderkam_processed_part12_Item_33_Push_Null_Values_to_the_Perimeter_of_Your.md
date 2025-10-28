# Flashcards: 10A007---Effective-TypeScript---Dan-Vanderkam_processed (Part 12)

**Starting Chapter:** Item 33 Push Null Values to the Perimeter of Your Types

---

#### Push Null Values to the Perimeter of Your Types
When you first turn on strictNullChecks, it may seem as though you have to add scores of if statements checking for null and undefined values throughout your code. This is often because the relationships between null and non-null values are implicit: when variable A is non-null, you know that variable B is also non-null and vice versa.

For example, consider a function `extent` that calculates the minimum and maximum of an iterable list of numbers.
:p How does the initial implementation of the `extent` function handle nulls?
??x
The initial implementation of the `extent` function handles nulls by allowing `min` and `max` to be either defined or undefined. This is problematic because it doesn't properly represent the relationship between these variables, leading to potential issues when both are `undefined`.

```typescript
function extent(nums: Iterable<number>) {
    let min, max;
    for (const num of nums) {
        if (!min) { // Incorrect check for min being undefined
            min = num;
            max = num;
        } else {
            min = Math.min(min, num);
            max = Math.max(max, num);
        }
    }
    return [min, max];
}
```

This function will fail when `nums` is an empty list and might also have issues if both `min` and `max` are zero.
x??

---
#### Use Objects to Track Non-null Values
To improve the design of the `extent` function, you can use a single object to track the extent values, making them either fully null or fully non-null.

:p How does the improved implementation of the `extent` function handle nulls?
??x
In the improved implementation, both `min` and `max` are tracked within an array. This ensures that if one is defined, the other will be as well. The object containing these values can either be fully non-null or null.

```typescript
function extent(nums: Iterable<number>) {
    let minMax: [number, number] | null = null;
    for (const num of nums) {
        if (!minMax) { // Check for undefined state correctly
            minMax = [num, num];
        } else {
            const [oldMin, oldMax] = minMax;
            minMax = [Math.min(num, oldMin), Math.max(num, oldMax)];
        }
    }
    return minMax;
}
```

This approach ensures that the returned object is either `[min, max]` or `null`, making it easier for clients to work with.

```typescript
const range = extent([0, 1, 2]);
if (range) {
    const [min, max] = range;
    const span = max - min; // Safe to use here
}
```

By using a single object to track the extent, you improve your design and make it clearer for TypeScript to understand.
x??

---
#### Design Classes with Non-null Values
When designing classes that may contain null values, consider pushing these values to the perimeter of your API by making larger objects either null or fully non-null.

:p How can the `UserPosts` class be redesigned to handle null values better?
??x
The original `UserPosts` class has properties that are initially set to `null`. These properties might hold both null and non-null states, leading to confusion and potential bugs. A better design is to initialize these properties only after all necessary data is fetched.

```typescript
class UserPosts {
    user: UserInfo;
    posts: Post[];

    constructor(user: UserInfo, posts: Post[]) {
        this.user = user;
        this.posts = posts;
    }

    static async init(userId: string): Promise<UserPosts> {
        const [user, posts] = await Promise.all([
            fetchUser(userId),
            fetchPostsForUser(userId)
        ]);
        return new UserPosts(user, posts);
    }

    getUserName() {
        return this.user.name; // Safe to call as user is non-null
    }
}
```

By initializing `user` and `posts` only when all data is available, the class ensures that these properties are always either both null or both non-null. This makes it easier to write correct methods without needing to handle null states.

```typescript
const userPosts = await UserPosts.init(userId);
console.log(userPosts.getUserName()); // Safe call
```

This approach reduces complexity and improves clarity.
x??

---

#### Prefer Unions of Interfaces to Interfaces with Unions

Background context explaining the concept. This section discusses how to define interfaces that represent valid state combinations, avoiding mixed property values which can lead to errors and complexity.

If applicable, add code examples with explanations:
```typescript
// Initial interface with union types
interface Layer {
    layout: FillLayout | LineLayout | PointLayout;
    paint: FillPaint | LinePaint | PointPaint;
}

// Possible usage leading to potential issues
const layerExample = { 
    layout: new FillLayout(), // This could be any of the three options for `layout`
    paint: new LinePaint()   // This could also be any of the three options for `paint`
};

// Better approach by splitting into separate interfaces and using a union type
interface FillLayer {
    layout: FillLayout;
    paint: FillPaint;
}

interface LineLayer {
    layout: LineLayout;
    paint: LinePaint;
}

interface PointLayer {
    layout: PointLayout;
    paint: PointPaint;
}

type Layer = FillLayer | LineLayer | PointLayer;

// Enhanced usage with explicit type information
function drawLayer(layer: Layer) {
    if (layer.layout instanceof FillLayout) { // TypeScript can narrow the type based on `layout`
        const {paint} = layer;  // paint will be of type `FillPaint`
        console.log('Drawing fill layer');
    } else if (layer.layout instanceof LineLayout) {
        const {paint} = layer;  // paint will be of type `LinePaint`
        console.log('Drawing line layer');
    } else {
        const {paint} = layer;  // paint will be of type `PointPaint`
        console.log('Drawing point layer');
    }
}
```

:p Why is it better to use a union of interfaces rather than an interface with union types for the `Layer` example?
??x
Using a union of interfaces for `Layer` ensures that each layout and paint property are correctly matched, reducing the risk of invalid combinations. This approach allows TypeScript to provide better type checking and autocompletion support.
```typescript
// Example function usage
const layerExample = new FillLayer(); // Correct combination of FillLayout and FillPaint
drawLayer(layerExample); // TypeScript can infer `layer` as a `FillLayer`
```
x??

---

#### Tagged Unions

Background context explaining the concept. Tagged unions, or discriminated unions, are used to model relationships between properties more clearly by using a tag property that indicates which type is present.

If applicable, add code examples with explanations:
```typescript
// Using tagged union approach for `Layer`
interface FillLayer {
    type: 'fill';
    layout: FillLayout;
    paint: FillPaint;
}

interface LineLayer {
    type: 'line';
    layout: LineLayout;
    paint: LinePaint;
}

interface PointLayer {
    type: 'point';
    layout: PointLayout;
    paint: PointPaint;
}

type Layer = FillLayer | LineLayer | PointLayer;

// Function to demonstrate usage
function drawLayer(layer: Layer) {
    switch (layer.type) { // `type` serves as a tag for discriminating the union
        case 'fill':
            const {paint} = layer;  // paint is now guaranteed to be of type FillPaint
            console.log('Drawing fill layer');
            break;
        case 'line':
            const {paint} = layer;  // paint is now guaranteed to be of type LinePaint
            console.log('Drawing line layer');
            break;
        case 'point':
            const {paint} = layer;  // paint is now guaranteed to be of type PointPaint
            console.log('Drawing point layer');
            break;
    }
}
```

:p How does using tagged unions improve the `Layer` definition?
??x
Using tagged unions improves the `Layer` definition by explicitly defining the relationship between layout and paint properties. The `type` property acts as a tag, allowing TypeScript to narrow down the type of the object based on its value.

This approach ensures that:
- Only valid combinations of `layout` and `paint` are possible.
- TypeScript can provide better type checking during development.
```typescript
// Example function usage
const fillLayer = new FillLayer(); // Correct combination with 'fill' tag
drawLayer(fillLayer); // TypeScript can infer the specific layer type
```
x??

---

#### Optional Fields and Unions

Background context explaining the concept. The section discusses how optional fields in interfaces might indicate a problem, suggesting that properties should be grouped into objects to represent their relationships more clearly.

If applicable, add code examples with explanations:
```typescript
// Initial problematic interface
interface Person {
    name: string;
    placeOfBirth?: string; // This could be missing or present
    dateOfBirth?: Date;  // This could also be missing or present
}

// Example usage leading to potential issues
const alanT = { 
    name: 'Alan Turing', 
    placeOfBirth: 'London' // Could be missing in some cases
};

// Better approach by grouping related optional fields into an object
interface Person {
    name: string;
    birth?: {  // Birth details as a single object
        place: string;
        date: Date;
    }
}

// Example usage with improved clarity
const alanT = { 
    name: 'Alan Turing', 
    birth: { 
        place: 'London', 
        date: new Date('1912-06-23') // Both `place` and `date` are required together
    }
};

// Function to demonstrate usage
function eulogize(person: Person) {
    console.log(`Name: ${person.name}`);
    if (person.birth) {  // Check for presence of birth details
        console.log(`Born on ${person.birth.date} in ${person.birth.place}.`);
    }
}
```

:p How does grouping related optional fields into an object improve the `Person` interface?
??x
Grouping related optional fields into an object improves the `Person` interface by making it clear that both birth-related properties must be present or absent together. This approach ensures that TypeScript can enforce this relationship and provide better type checking.

By doing so, you avoid potential issues where only one part of the optional field is provided:
```typescript
// Example usage with improved clarity
const alanT = { 
    name: 'Alan Turing', 
    birth: {
        place: 'London',
        date: new Date('1912-06-23') // Both `place` and `date` are required together
    }
};
```
x??

#### Prefer More Precise Alternatives to String Types
String types are often used broadly, but they can mask errors and hide relationships between types. By using more precise alternatives, you improve type safety and code readability.

:p What is a "stringly typed" function parameter?
??x
A "stringly typed" function parameter uses the string type without specifying the exact kind of data it should contain. This can lead to issues because any string value can be passed, even if only certain types are valid.
```typescript
function recordRelease(title: string, date: string) {
    // ... implementation ...
}
recordRelease(kindOfBlue.releaseDate, kindOfBlue.title);  // OK, should be error
```
x??

---

#### Use Union Types for Narrowing String Types
Union types allow you to specify multiple possible values for a type, making the code more precise and catching potential errors.

:p How can you define a union type for recording types?
??x
You can define a union type using the `|` operator to list all possible values. This ensures that only valid types are used.
```typescript
type RecordingType = 'studio' | 'live';
```
x??

---

#### Use Keyof Types for Function Parameters
The `keyof` type allows you to specify which keys of an object can be used as function parameters, improving type safety.

:p How does using `keyof Album` improve the pluck function?
??x
Using `keyof Album` ensures that only valid keys from the Album interface are accepted, preventing errors and making the code more readable.
```typescript
function pluck<T, K extends keyof T>(records: T[], key: K): T[K][] {
    return records.map(r => r[key]);
}
```
x??

---

#### Avoid Stringly Typed Code
String types should be avoided when not every string is a possibility. Using more precise types ensures better error detection and improved code readability.

:p Why are stringly typed fields problematic in the Album interface?
??x
Stringly typed fields can mask errors because any string value can be used, even if only specific values are valid. This leads to incorrect assignments and makes it harder to catch bugs.
```typescript
interface Album {
    artist: string; // Can contain any string, not just "Miles Davis"
    title: string;  // Can contain any string, not just "Kind of Blue"
    releaseDate: string; // Can be incorrectly formatted like 'August 17th, 1959'
    recordingType: string; // Can be incorrectly capitalized or misspelled
}
```
x??

---

#### Use Date Objects for Date Fields
Using `Date` objects instead of strings can avoid issues with formatting and make the code more precise.

:p How does using a `Date` object in the Album interface improve error checking?
??x
Using a `Date` object ensures that the release date is always correctly formatted, preventing errors like incorrect dates. This makes the code more robust and easier to maintain.
```typescript
interface Album {
    artist: string;
    title: string;
    releaseDate: Date; // Ensures correct formatting
    recordingType: 'studio' | 'live';
}
```
x??

---

#### Use Named Types for Better Documentation
Explicitly defining types can provide better documentation and make the code more understandable.

:p How does using a named type like `RecordingType` improve the Album interface?
??x
Using a named type provides clear documentation about what values are valid. This makes it easier to understand the code and prevents bugs.
```typescript
type RecordingType = 'live' | 'studio';
interface Album {
    artist: string;
    title: string;
    releaseDate: Date;
    recordingType: RecordingType; // Clearer documentation
}
```
x??

---

#### Use Template Literal Types for Infinite Sets of Strings
Template literal types can be used to model infinite sets, such as all strings that start with "http:".

:p How do template literal types differ from union types?
??x
Template literal types allow you to specify patterns in string values, whereas union types list specific possible values. For example, a template literal type could represent any string starting with "http:", while a union type would list specific valid strings.
```typescript
type HttpUrl = `http://${string}`;
```
x??
---

