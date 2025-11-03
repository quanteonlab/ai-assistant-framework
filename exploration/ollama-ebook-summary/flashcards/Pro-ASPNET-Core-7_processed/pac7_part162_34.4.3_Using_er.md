# Flashcards: Pro-ASPNET-Core-7_processed (Part 162)

**Starting Chapter:** 34.4.3 Using error boundaries

---

#### Using Blazor Server, Part 2

Blazor is a framework for building interactive web UIs using C# and .NET. In this section, we explore how to handle errors within Blazor applications.

:p What element is shown when an error occurs in a Blazor Server application, and what does it do?
??x
When an error occurs in a Blazor Server application, the `blazor-error-ui` div is displayed. It shows a warning message telling the user that an error has occurred and provides a button to reload the browser.
```html
<div id="blazor-error-ui"
     class="text-center bg-danger h6 text-white p-2 fixed-top w-100"
     style="display:none">
    An error has occurred. This application will not respond until reloaded.
    <button class="btn btn-sm btn-primary m-1" onclick="location.reload()">
        Reload
    </button>
</div>
```
x??

---

#### Displaying an Error Message

Blazor applications can display specific messages for different types of errors.

:p How does a Blazor application show an error message when the user selects "Sales" from the Highlight menu?
??x
When the user selects "Sales" from the Highlight menu, the `SelectFilter` component throws an exception. This is contained by an error boundary, which displays a custom error message.
```razor
<ErrorBoundary>
    <ChildContent>
        <!-- Content that might throw exceptions -->
    </ChildContent>
    <ErrorContent>
        <h4 class="bg-danger text-white text-center h-100 p-2">
            Inline error: Sales Selected
        </h4>
    </ErrorContent>
</ErrorBoundary>
```
x??

---

#### Using Error Boundaries

Error boundaries are components that catch errors in the child component hierarchy.

:p How does an error boundary handle exceptions thrown by its child components?
??x
An error boundary catches and handles exceptions thrown by its child components. When an exception is caught, it replaces the content of the child with a custom error message.
```razor
<ErrorBoundary>
    <ChildContent>
        <!-- Content that might throw exceptions -->
    </ChildContent>
    <ErrorContent>
        <!-- Custom error message -->
    </ErrorContent>
</ErrorBoundary>
```
x??

---

#### Defining Error Content

Defining the error content directly within the error boundary allows for more control over how errors are displayed.

:p How can you define the error content directly in a Blazor component?
??x
To define error content directly, use the `ChildContent` and `ErrorContent` tags within an `ErrorBoundary`.
```razor
<ErrorBoundary>
    <ChildContent>
        <!-- Content that might throw exceptions -->
    </ChildContent>
    <ErrorContent>
        <!-- Custom error message -->
    </ErrorContent>
</ErrorBoundary>
```
x??

---

#### Recovering from Exceptions

Error boundaries can also provide a way for users to recover from errors.

:p What does the `Recover` method do in an error boundary?
??x
The `Recover` method allows users to recover from errors by resetting the state of the component hierarchy. It is typically used when an exception has been handled and the application should continue running.
```razor
@code {
    ErrorBoundary? boundary;

    // Handle button click to call Recover()
}
```
x??

---

#### Summary

Blazor components can be combined to present composite features, configured using attributes in markup, define custom events, and wrap content. Errors can be presented to the user through specific elements, and error boundaries help contain these errors.

:p What is an error boundary in Blazor?
??x
An error boundary in Blazor is a component that catches exceptions thrown by its child components and displays a custom error message instead of crashing the application.
```razor
<ErrorBoundary>
    <ChildContent>
        <!-- Content that might throw exceptions -->
    </ChildContent>
    <ErrorContent>
        <!-- Custom error message -->
    </ErrorContent>
</ErrorBoundary>
```
x??

#### URL Routing in Blazor
Background context: URL routing allows components to respond to changes in the URL without requiring a new HTTP connection. This is particularly useful for creating complex applications that benefit from the Blazor architecture.

:p How does URL routing work in Blazor?
??x
URL routing in Blazor works by configuring routes using `@page` directives and setting up a routing system with built-in components. When a user navigates to a specific URL, the appropriate component is loaded and rendered without making an additional HTTP request.
```razor
// Example of setting up a route
@page "/about"
<h1>About Us</h1>
```
x??

---

#### Using Layouts in Blazor
Background context: A layout can be used to provide common content across multiple routed components. This helps maintain consistency and reduces code duplication.

:p How do you use layouts with routed components?
??x
You can create a layout file that contains shared content such as navigation menus, headers, or footers. Then, inside the `@layout` directive in your component, you can specify which parts of the layout should be replaced by the component's content.
```razor
// Example of a layout and a component using it
@layout Layout.razor

@page "/home"
<h1>Welcome to Home</h1>
```
x??

---

#### Component Lifecycle Methods in Blazor
Background context: The component lifecycle methods allow components to participate actively in the execution of the application, which is especially important when using URL routing.

:p What are the key lifecycle methods in Blazor?
??x
The key lifecycle methods in Blazor include `OnInitialized()`, `OnInitializedAsync()`, `OnParametersSet()`, and `OnParametersSetAsync()`. These methods are called at different stages of a component's lifecycle, allowing you to perform initialization tasks or handle asynchronous operations.
```razor
@code {
    protected override void OnInitialized()
    {
        // Code executed on initialization
    }

    protected override async Task OnInitializedAsync()
    {
        await base.OnInitializedAsync();
        // Asynchronous code
    }
}
```
x??

---

#### Interactions Between Components and JavaScript in Blazor
Background context: Blazor provides several ways for components to interact with each other and with external JavaScript code, which is useful for complex applications.

:p How can components interact outside of parent-child relationships?
??x
Components can interact using the `@ref` expression or by utilizing the interoperability features provided by Blazor. For example, you can use `Blazored.LocalStorage` to store data and access it from different components.
```razor
// Example of using @ref for component interaction
<ChildComponent @ref="childRef" />

@code {
    private ChildComponent childRef;

    protected override void OnInitialized()
    {
        // Perform actions with the child component via `childRef`
    }
}
```
x??

---

#### Managing Component Interactions
Background context: In addition to using `@ref` and interoperability features, components can interact by retaining references and managing state across different components.

:p How do you retain references between components?
??x
You can retain references between components by passing them through parent-child relationships or by using shared services. For example, you can define a service that holds references to multiple components and allows them to communicate indirectly.
```razor
// Example of retaining references with @ref in a parent component
@page "/main"
<ChildA @ref="childA" />
<ChildB @ref="childB" />

@code {
    private ChildComponent childA;
    private AnotherChildComponent childB;

    protected override void OnInitialized()
    {
        // Perform actions to interact with `childA` and `childB`
    }
}
```
x??

---

#### Advanced Interactions in Blazor
Background context: Advanced interactions involve coordinating activities between multiple components and interacting with external JavaScript code, which can be complex but are essential for creating rich applications.

:p How do you coordinate the activities of multiple components?
??x
You can use shared state management solutions like `Blazored.LocalStorage` or implement custom services to manage state across different components. This allows components to share data and perform coordinated actions.
```razor
// Example of using a service to retain references
@page "/main"
<ChildA @ref="childA" />
<ChildB @ref="childB" />

@code {
    private ChildComponent childA;
    private AnotherChildComponent childB;

    [Inject]
    public IMyService MyService { get; set; }

    protected override void OnInitialized()
    {
        // Use `MyService` to coordinate actions between components
    }
}
```
x??

---

---
#### Dropping the Database
Background context: This section explains how to drop a database using Entity Framework Core commands. It is essential to ensure that you have no critical data before running these commands, as they will remove your current database.

:p How do you use Entity Framework Core to drop a database?
??x
You can use the `dotnet ef database drop --force` command in the terminal or PowerShell to drop the database. This command uses the Entity Framework Core tools to execute migrations and delete the associated database schema.

```powershell
dotnet ef database drop --force
```
x??

---
#### Running the Example Application
Background context: The application can be run using the `dotnet run` command, which will start the development server and make the application accessible via a web browser. This allows you to test the functionalities of the Blazor app.

:p How do you run the example application?
??x
You can run the example application by executing the following command in your terminal or PowerShell:

```powershell
dotnet run
```
This command will start the development server, and the application will be accessible at `http://localhost:5000`.

x??

---
#### Using Component Routing
Background context: Blazor supports routing through the ASP.NET Core framework to display different Razor components based on URL changes. This allows for dynamic content loading without page reloads.

:p What is the role of the Router component in Blazor?
??x
The `Router` component acts as a bridge between Blazor and ASP.NET Core's routing features. It provides a way to route URLs to specific components dynamically. The `Router` component defines two sections: `Found`, which displays the matched component, and `NotFound`, which shows content when no matching component is found.

```razor
<Router AppAssembly="typeof(Program).Assembly">
    <Found>
        <RouteView RouteData="@context" />
    </Found>
    <NotFound>
        <h4 class="bg-danger text-white text-center p-2">No Matching Route Found</h4>
    </NotFound>
</Router>
```
x??

---
#### Preparing the Razor Page for Component Routing
Background context: To use component routing effectively, it is recommended to create a dedicated page that acts as the entry point. This ensures that URLs are distinct and easier to manage.

:p How do you configure a fallback route in the `Program.cs` file?
??x
To handle unmatched routes using component routing, you need to add a fallback route configuration in your `Program.cs` file. Here’s how you can do it:

```csharp
var app = builder.Build();
app.UseStaticFiles();
app.MapControllers();
app.MapControllerRoute(
    name: "controllers",
    pattern: "controllers/{controller=Home}/{action=Index}/{id?}"
);
app.MapRazorPages();
app.MapBlazorHub();
app.MapFallbackToPage("/_Host");
```

The `MapFallbackToPage` method ensures that if no other route matches, the `_Host.cshtml` file will be rendered.

x??

---
#### The Routed Component
Background context: The `Routed.razor` component uses the `Router` and `RouteView` components to display different Razor components based on the current URL.

:p How is the `Found` section of the `Router` component utilized?
??x
The `Found` section of the `Router` component contains a `RouteView` that renders the appropriate component based on the matched route. The `RouteData` property is passed to the `RouteView`, which then determines and displays the corresponding Razor component.

```razor
<Found>
    <RouteView RouteData="@context" />
</Found>
```

The `RouteView` component takes the `RouteData` parameter, which includes information about the current route, such as the name of the component to render.

x??

---

