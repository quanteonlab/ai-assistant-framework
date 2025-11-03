# Flashcards: 10A007---Effective-TypeScript---Dan-Vanderkam_processed (Part 10)

**Starting Chapter:** Item 27 Use async Functions Instead of Callbacks to Improve Type Flow

---

#### Introduction to Promises
Promises are a way of handling asynchronous operations in JavaScript. They represent a value that may be available now, in the future, or never. A Promise can be in one of three states: pending (initial), fulfilled (has a value), or rejected (has an error).

A Promise is created by calling its constructor with a callback function which gets two arguments: `resolve` and `reject`. These are functions that you pass values to when the operation completes successfully, or reject it if something went wrong.

:p What does a Promises constructor receive as parameters?
??x
The `Promise` constructor receives two functions as parameters: `resolve` and `reject`.

```js
const myPromise = new Promise((resolve, reject) => {
  // code that may resolve or reject the promise
});
```
x??

---

#### Using Promises in Asynchronous Operations
Promises allow for cleaner and more readable asynchronous operations compared to callbacks. They enable you to handle errors and perform multiple tasks sequentially or concurrently.

:p How can you use promises to fetch pages concurrently?
??x
You can use `Promise.all` to fetch pages concurrently:

```js
async function fetchPages() {
  const [response1, response2, response3] = await Promise.all([
    fetch(url1),
    fetch(url2),
    fetch(url3)
  ]);
}
```

Here, `await Promise.all` waits for all promises in the array to resolve. The result is an array of resolved values.

x??

---

#### Async/Await with Promises
Async/await is a syntactic sugar that simplifies working with Promises by allowing you to write asynchronous code that looks synchronous. It's supported by TypeScript and most modern JavaScript runtimes.

:p How does async/await handle errors in functions?
??x
In async functions, any error thrown inside the function will be caught as an exception in the `catch` block:

```js
async function fetchPages() {
  try {
    const response1 = await fetch(url1);
    const response2 = await fetch(url2);
    const response3 = await fetch(url3);
  } catch (e) {
    // handle error
  }
}
```

If the `await` expression resolves to a rejected Promise, an exception is thrown and caught in the `catch` block.

x??

---

#### Promises with Higher-Order Functions
Promises can be combined using higher-order functions like `Promise.all`, `Promise.race`, etc. These allow you to handle multiple asynchronous operations more effectively.

:p How does `Promise.race` work?
??x
`Promise.race` returns a new Promise that resolves as soon as one of the promises in the array fulfills or rejects:

```js
function timeout(timeoutMs: number): Promise<never> {
  return new Promise((resolve, reject) => {
    setTimeout(() => reject('timeout'), timeoutMs);
  });
}

async function fetchWithTimeout(url: string, timeoutMs: number) {
  return Promise.race([fetch(url), timeout(timeoutMs)]);
}
```

In this example, `Promise.race` returns a new promise that resolves with the first value or rejects immediately after the timeout.

x??

---

#### Using Async Functions
Async functions are always asynchronous and must return Promises. This makes them easier to reason about compared to regular Promise callbacks.

:p Why should you prefer async/await over raw Promises?
??x
You should prefer async/await for a few reasons:
- It produces more concise and straightforward code.
- It enforces that async functions always return Promises, avoiding potential bugs from mixing synchronous and asynchronous operations.

For example:

```js
async function getNumber() {
  return 42;
}

// The TypeScript type infers this as Promise<number>
const getNumber = async () => 42;

// Raw Promise equivalent is less readable
const getNumber = () => Promise.resolve(42);
```

x??

---

#### Consistent Asynchronous Behavior with Async Functions
Async functions ensure that code runs consistently, avoiding issues like unexpected state changes due to caching or other asynchronous operations.

:p How does introducing an async function help in maintaining consistent behavior?
??x
Introducing an `async` function helps maintain consistency by ensuring that the flow of execution remains clear and predictable:

```js
const _cache: {[url: string]: string} = {};

async function fetchWithCache(url: string) {
  if (url in _cache) {
    return _cache[url];
  }
  const response = await fetch(url);
  const text = await response.text();
  _cache[url] = text;
  return text;
}
```

Using `async` ensures that the state (`requestStatus`) is updated consistently, making the behavior predictable.

x??

---

#### Async Functions and Promises

Background context: In TypeScript, async functions return a Promise. The return value of an `async` function is automatically wrapped in a `Promise`. However, if you explicitly return a Promise from an `async` function, it will not be wrapped again.

:p What happens when you return a Promise directly from an `async` function?
??x
When you return a Promise directly from an async function, TypeScript does not wrap the returned value in another Promise. The result is that the type of the function's return value is `Promise<T>` rather than `Promise<Promise<T>>`.

```typescript
async function getJSON(url: string): Promise<any> {
    const response = await fetch(url);
    const jsonPromise = response.json();
    return jsonPromise; // No additional wrapping occurs here
}
```
x??

---

#### Prefer Promises and Async/Await

Background context: Promises are generally preferred over callbacks for better composability and type flow. `async` and `await` make the code more readable and easier to reason about by avoiding deep nested callbacks.

:p Why should you prefer using `async`/`await` over raw Promises?
??x
Using `async`/`await` is preferable because it produces cleaner, more readable code that is less prone to callback hell. The use of `await` within an `async` function allows you to write asynchronous operations in a synchronous-like manner, which can make debugging and understanding the flow easier.

```typescript
// Using async/await for fetching JSON data
async function getJSON(url: string): Promise<any> {
    const response = await fetch(url);
    return response.json();
}
```
x??

---

#### TypeScript Type Inference

Background context: TypeScript has strong type inference capabilities, but sometimes it can be too strict. The `fetchAPI` example illustrates this limitation; you need to provide both the API and Path types explicitly because of how type inference works in TypeScript.

:p Why do we get an error when using `fetchAPI<SeedAPI>('/seed/strawberry')`?
??x
The error occurs because TypeScript requires all type parameters to be specified explicitly when calling a generic function, even if some can be inferred. In the case of `fetchAPI`, both the API and Path types must be provided.

```typescript
// Incorrect usage with an error: fetchAPI<SeedAPI>('/seed/strawberry');
// Expected 2 type arguments, but got 1.
```
x??

---

#### Class for Type Capture

Background context: Using a class can help separate the explicit specification of generic types from the inference of others. This approach is particularly useful when you have multiple methods that require the same type parameters.

:p How does using a class to create an `ApiFetcher` solve the problem?
??x
Using a class allows us to define a single type parameter during object creation and infer another within method calls. This separation makes the code more flexible and easier to use while still leveraging TypeScript's strong typing system.

```typescript
// Using a class for Api Fetching
declare class ApiFetcher<API> {
    fetch<Path extends keyof API>(path: Path): Promise<API[Path]>;
}

const fetcher = new ApiFetcher<SeedAPI>();
const berry = await fetcher.fetch('/seed/strawberry'); // Correct usage
```
x??

---

#### Currying for Type Capture

Background context: Currying can be used to create a function that returns another function, each with its own type parameters. This approach allows for more flexible and inferred type handling.

:p How does currying help in the `fetchAPI` example?
??x
Currying helps by allowing us to return a new function for each call, where the API parameter can be specified once, and subsequent calls infer the Path correctly.

```typescript
// Using currying for fetch API
declare function fetchAPI<API>(): <Path extends keyof API>(path: Path) => Promise<API[Path]>;

const berry = await fetchAPI<SeedAPI>()('/seed/strawberry'); // Correct usage
```
x??

---

#### Local Type Aliases with Currying

Background context: When using curried functions, you can define local type aliases within the function's implementation to make complex types more manageable.

:p How does defining a `Routes` type alias help in the `fetchAPI` example?
??x
Defining a local type alias like `Routes` helps simplify and clarify the type constraints. This makes it easier to understand and work with the generic parameters.

```typescript
// Example of using a local type alias with currying
function fetchAPI<API>() {
    type Routes = keyof API & string;  // Local type alias
    return <Path extends Routes>(
        path: Path
    ): Promise<API[Path]> => fetch(path).then(r => r.json());
}
```
x??

---

#### Valid State Representation
Background context: When designing types, it is crucial to ensure that they can only represent valid states. This makes your code more understandable and reduces the likelihood of bugs. If a type allows invalid states, it can lead to confusion when reading or maintaining the code.

:p What is the importance of ensuring types always represent valid states?
??x
Ensuring that types always represent valid states helps in making the code straightforward and less error-prone. By defining types such that they only reflect meaningful scenarios, you make your code easier to understand and maintain. This approach reduces the number of potential bugs because invalid states cannot be created through type mismatches.

For example, consider a `State` interface:
```typescript
interface State {
  pageText: string;
  isLoading: boolean;
  error?: string;
}
```
If this type is used correctly, it ensures that any state passed to functions like `renderPage` or `changePage` will always have a valid combination of properties. This helps in preventing logic errors where invalid states could lead to confusion.

```typescript
// Example function with invalid state handling
function renderPage(state: State) {
  if (state.error) {
    return `Error. Unable to load ${currentPage}: ${state.error}`;
  } else if (state.isLoading) {
    return `Loading ${currentPage}...`;
  }
  return `<h1>${currentPage}</h1> ${state.pageText}`;
}
```
??x
The function `renderPage` can encounter issues when both `isLoading` and `error` are set, leading to ambiguous behavior. By ensuring that the type only allows valid combinations of these properties, you prevent such ambiguities.
```typescript
// Revised State interface to better represent valid states
interface State {
  pageText: string;
  isLoading: boolean;
  error?: never; // Ensures error can't be set while isLoading is true
}
```
This revised version ensures that if `isLoading` is true, `error` must not exist, making the function logic clearer and less prone to errors.

---
#### Handling Invalid State in Code
Background context: When designing functions that manipulate state, it's essential to handle invalid states gracefully. If your code allows for invalid states, it can lead to unexpected behavior or bugs. Ensuring valid state transitions is a key aspect of robust type design.

:p How can you ensure that your code handles invalid states correctly?
??x
To ensure that your code handles invalid states correctly, you should design types and functions such that they only allow valid combinations of properties. This involves carefully crafting interfaces and ensuring that any function that modifies state does so in a way that maintains the validity of the overall state.

For example, consider an `async` function `changePage`:
```typescript
async function changePage(state: State, newPage: string) {
  state.isLoading = true;
  try {
    const response = await fetch(getUrlForPage(newPage));
    if (!response.ok) { // Corrected condition to check for non-OK status
      throw new Error(`Unable to load ${newPage}: ${response.statusText}`);
    }
    const text = await response.text();
    state.isLoading = false; // Set isLoading back to false after successful fetch
    state.pageText = text;
  } catch (e) {
    state.error = e.toString(); // Use toString method instead of concatenation
  }
}
```
In this example, the function ensures that `isLoading` is set to `false` when a request succeeds and updates the `pageText`. Additionally, it sets `error` correctly in case of an error. These changes prevent potential issues like lingering loading or error states.

---
#### State Transition Examples
Background context: When working with state in applications, understanding how different states transition can help in writing more robust and maintainable code. Ensuring that state transitions are well-defined and handled correctly is crucial for avoiding bugs and making the code easier to understand.

:p What issues arise from poorly designed state transitions?
??x
Poorly designed state transitions can lead to several issues, including ambiguous behavior, incorrect error handling, and potential data inconsistencies. For instance, in an application that involves loading content, if the state representation does not ensure valid combinations of properties, it can result in logical errors.

Consider the `changePage` function from earlier:
```typescript
async function changePage(state: State, newPage: string) {
  state.isLoading = true;
  try {
    const response = await fetch(getUrlForPage(newPage));
    if (response.ok) { // Corrected condition to check for OK status
      throw new Error(`Unable to load ${newPage}: ${response.statusText}`);
    }
    const text = await response.text();
    state.isLoading = false; // Set isLoading back to false after successful fetch
    state.pageText = text;
  } catch (e) {
    state.error = '' + e; // Improper concatenation, use toString() instead
  }
}
```
This function has several issues:
- It sets `isLoading` to `false` only if the request succeeds.
- It uses improper string concatenation in the error handling block.

To fix these issues, you should handle both success and failure cases properly:
```typescript
async function changePage(state: State, newPage: string) {
  state.isLoading = true;
  try {
    const response = await fetch(getUrlForPage(newPage));
    if (!response.ok) { // Corrected condition to check for non-OK status
      throw new Error(`Unable to load ${newPage}: ${response.statusText}`);
    }
    const text = await response.text();
    state.isLoading = false; // Set isLoading back to false after successful fetch
    state.pageText = text;
  } catch (e) {
    state.error = e.toString(); // Use toString method instead of concatenation
  }
}
```
This revised version ensures that `isLoading` is set to `false` in both success and failure scenarios, making the code more robust and easier to understand.

---
#### Valid State vs. Invalid State
Background context: Understanding the difference between valid and invalid states is crucial for designing types and functions that handle state transitions correctly. Ensuring that your code only allows valid states can prevent many common bugs and make your application more reliable.

:p How do you differentiate between valid and invalid states in type design?
??x
In type design, you differentiate between valid and invalid states by carefully defining the properties of an object or interface such that they represent meaningful combinations of values. Invalid states are those where certain conditions cannot be met simultaneously.

For example, consider a `State` interface:
```typescript
interface State {
  pageText: string;
  isLoading: boolean;
  error?: never; // Ensures error can't be set while isLoading is true
}
```
This type ensures that if `isLoading` is `true`, then `error` cannot exist. This helps in preventing ambiguous states where both properties might overlap, leading to unexpected behavior.

Here’s another example:
```typescript
interface LoginPageState {
  username: string;
  password: string;
  submitting?: never; // Ensures submitting can't be true if either field is empty
}
```
This type ensures that the `submitting` property cannot be set to `true` unless both `username` and `password` are non-empty.

By defining types in this way, you make your code more robust and easier to reason about. This approach helps in reducing bugs by preventing invalid state transitions.

---
#### Type Design for Valid State
Background context: When designing types that represent valid states, it's crucial to ensure that the type constraints reflect meaningful combinations of properties. By doing so, you can avoid ambiguous or inconsistent states, which can lead to errors and confusion.

:p What are the benefits of using type definitions to enforce valid state transitions?
??x
Using type definitions to enforce valid state transitions offers several benefits:
1. **Clarity**: It makes the intended state combinations clear to developers reading the code.
2. **Reduced Bugs**: By constraining states, you prevent invalid or ambiguous scenarios that could lead to bugs.
3. **Maintainability**: Well-defined types make it easier to maintain and understand the code.

For example, consider a `UserState` interface:
```typescript
interface UserState {
  isLoggedIn: boolean;
  username?: string; // Optional if not logged in
  permissions?: Permissions; // Optional set of permissions
  isUpdatingProfile?: never; // Ensures this can't be true while not logged in
}
```
This type ensures that `isUpdatingProfile` cannot be `true` unless the user is logged in, making the logic clearer and reducing potential bugs.

Another example:
```typescript
interface OrderState {
  orderId: number;
  status?: 'pending' | 'shipped' | 'delivered';
  paymentStatus?: never; // Ensures paymentStatus can't be set if not shipped
}
```
This type ensures that `paymentStatus` cannot be set unless the order is in a "shipped" state, making the logic clear and preventing invalid states.

By enforcing such constraints through types, you create a more robust and maintainable codebase.

#### Improved State Representation
Background context: The original state representation was ambiguous, leading to difficulties in implementing `render()` and `changePage()`. A new approach using tagged unions (discriminated unions) explicitly models the different states a network request can be in. This approach eliminates invalid states and makes it easier to implement functions like `renderPage` and `changePage`.
:p How does the improved state representation eliminate ambiguity?
??x
The improved state representation uses tagged unions (`RequestPending`, `RequestError`, and `RequestSuccess`) to explicitly model different states of network requests, ensuring that each request is in exactly one state. This eliminates the ambiguity present in the original implementation where it was unclear what the current page was or how pending and error states interacted.
```typescript
interface RequestPending {
    state: 'pending';
}

interface RequestError {
    state: 'error';
    error: string;
}

interface RequestSuccess {
    state: 'ok';
    pageText: string;
}

type RequestState = RequestPending | RequestError | RequestSuccess;

interface State {
    currentPage: string;
    requests: {[page: string]: RequestState};
}
```
x??

---
#### RenderPage Function
Background context: The `renderPage` function takes the state as input and determines what to render based on the current page's request state.
:p What is the logic behind the `renderPage` function?
??x
The `renderPage` function uses a switch statement to determine the appropriate content to render based on the state of the current page. It checks the request state for each page, rendering different messages depending on whether the request is pending, an error occurred, or the data has been successfully fetched.
```typescript
function renderPage(state: State) {
    const {currentPage} = state;
    const requestState = state.requests[currentPage];
    
    switch (requestState.state) {
        case 'pending': 
            return `Loading ${currentPage}...`;
        case 'error':
            return `Error. Unable to load ${currentPage}: ${requestState.error}`;
        case 'ok':
            return `<h1>${currentPage}</h1> ${requestState.pageText}`;
    }
}
```
x??

---
#### ChangePage Function
Background context: The `changePage` function updates the state when the user changes pages, handling both pending and error states appropriately.
:p What does the `changePage` function do?
??x
The `changePage` function updates the state to reflect a new page being selected. It sets the request state of the new page to 'pending' and then attempts to fetch the content for that page. If successful, it updates the request state to 'ok' with the fetched text; otherwise, it sets the request state to 'error' with the error message.
```typescript
async function changePage(state: State, newPage: string) {
    state.requests[newPage] = {state: 'pending'};
    state.currentPage = newPage;

    try {
        const response = await fetch(getUrlForPage(newPage));
        if (!response.ok) {
            throw new Error(`Unable to load ${newPage}: ${response.statusText}`);
        }
        const pageText = await response.text();
        state.requests[newPage] = {state: 'ok', pageText};
    } catch (e) {
        state.requests[newPage] = {state: 'error', error: e.toString()};
    }
}
```
x??

---
#### Airbus 330 Flight 447 Accident
Background context: The accident of Air France Flight 447 highlighted the dangers of poor state management in complex systems, particularly in automated aircraft like the Airbus 330. The separate control inputs for pilots and copilots contributed to the confusion during the flight.
:p How did the design of controls on the Airbus 330 contribute to the accident?
??x
The design of controls on the Airbus 330 had two sets: one for the pilot and another for the co-pilot. This separation led to a situation where both pilots were trying to control the aircraft independently, causing confusion during critical moments. When the aircraft encountered severe turbulence, the automated systems malfunctioned due to conflicting commands from both sets of controls.
```java
// Pseudocode example
public class FlightControlSystem {
    private boolean pilotControl;
    private boolean copilotControl;

    public void applyPilotInput(double input) {
        if (pilotControl) {
            // Apply pilot's control input
        }
    }

    public void applyCopilotInput(double input) {
        if (copilotControl) {
            // Apply co-pilot's control input
        }
    }
}
```
x??

---

#### Designing Interfaces for Simplicity and Safety
Background context: The text discusses the importance of designing interfaces, especially those involving critical systems like aircraft controls, to ensure they are simple, unambiguous, and safe. It uses the example of Airbus 330’s side sticks and the getStickSetting function to illustrate potential pitfalls in complex system designs.

:p What is the main issue highlighted in the text regarding the design of the `getStickSetting` function?
??x
The main issue highlighted is that the `getStickSetting` function, as designed with dual input mode, can fail silently. The function averages the inputs from both side sticks, which could lead to unexpected behavior if both pilots are pulling in opposite directions.

This leads to a situation where neither pilot's intended action (climbing or diving) is properly executed due to averaging out their conflicting commands.
x??

---
#### Mechanical Connection of Controls
Background context: The text mentions that in most planes, the controls are mechanically linked. If one pilot moves a control, the other will experience the same movement. This simplifies the state representation since only one angle needs to be tracked.

:p How does the mechanical connection between cockpit controls simplify the design?
??x
The mechanical connection between cockpit controls simplifies the design by ensuring that both pilots' actions are linked. If one side stick is moved, the other will move in a corresponding manner. This means that only one `stickAngle` value needs to be tracked rather than two independent values.

This simplification reduces complexity and potential ambiguity, making it easier to determine the current state of the controls.
x??

---
#### Averaging Input vs. Linked Controls
Background context: The text provides examples of how averaging input from multiple controls (like the side sticks) can lead to unexpected outcomes. It contrasts this with a simpler design where controls are linked mechanically.

:p How does averaging inputs from both side sticks in `getStickSetting` function pose a risk?
??x
Averaging inputs from both side sticks in the `getStickSetting` function poses a significant risk because it can result in no input being effectively applied if one pilot pulls back and the other pushes forward. This can lead to dangerous situations, such as the Airbus 330 case where pilots were moving their controls in opposite directions but the system averaged out their inputs, causing the plane to remain stationary despite intended actions.

This averaging mechanism does not account for conflicting commands effectively and can result in a failure of critical safety measures.
x??

---
#### Design Principles for Critical Systems
Background context: The text emphasizes the importance of designing interfaces that are simple and unambiguous. It suggests that by excluding invalid states, the code becomes easier to write and safer to use.

:p What principle does the text suggest when designing interfaces for critical systems?
??x
The text suggests the principle of designing interfaces in a way that only valid states are allowed, making the code simpler and reducing the potential for errors. This involves carefully considering which values should be included and which should be excluded to ensure safety and clarity.

For example, instead of using separate left and right side sticks with dual input mode, a simpler design would link both controls mechanically so that they move together. This ensures that only one angle needs to be tracked, making the system safer and more straightforward.
x??

---

