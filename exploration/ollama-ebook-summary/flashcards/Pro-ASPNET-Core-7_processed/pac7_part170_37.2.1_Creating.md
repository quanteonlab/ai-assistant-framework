# Flashcards: Pro-ASPNET-Core-7_processed (Part 170)

**Starting Chapter:** 37.2.1 Creating the shared project. 37.2.3 Preparing the ASP.NET Core project

---

#### Creating a Shared Project
Background context: Blazor WebAssembly requires a separate project for Razor Components, which are compiled into JavaScript and executed by the browser. This separation ensures that the application can be efficiently delivered to the client. A shared project is created to contain common data models used across different projects.

:p What is the purpose of creating a shared project in Blazor WebAssembly?
??x
The purpose of creating a shared project is to house common data model classes, such as `Person`, `Department`, and `Location`, which can be reused by both the server-side ASP.NET Core project and the client-side Blazor WebAssembly project. This promotes code reusability and maintainability.

For example, you might need these data models in multiple parts of your application, so having them in a shared project makes it easier to manage and update them.
??x

---

#### Creating the Blazor WebAssembly Project
Background context: A separate Blazor WebAssembly project is needed because Razor Components are compiled into JavaScript and executed by the browser. This project will be hosted on an ASP.NET Core server, which serves both the compiled components and data through web services.

:p How do you create a new Blazor WebAssembly project?
??x
To create a new Blazor WebAssembly project, navigate to the directory containing your existing Advanced project and run the following commands in a PowerShell command prompt:

```powershell
dotnet new blazorwasm -o ../BlazorWebAssembly -f net7.0
```

This command creates a new Blazor WebAssembly project named `BlazorWebAssembly` that targets .NET 7.0.

Next, add a reference to the shared DataModel project:

```powershell
dotnet add ../BlazorWebAssembly reference ../DataModel
```

These commands set up the necessary structure and references for your Blazor WebAssembly application.
??x

---

#### Moving Classes into Shared Project
Background context: To share data models between different projects, you need to move the relevant classes from the existing Advanced project into a new shared DataModel project. This ensures that both the server-side and client-side applications can use these common data structures.

:p How do you move classes into a shared project?
??x
To move classes into a shared project, follow these steps:

1. Close Visual Studio or Visual Studio Code.
2. Open a PowerShell command prompt.
3. Navigate to the Advanced project folder (the one containing `Advanced.csproj`).
4. Run the following commands:

```powershell
dotnet new classlib -o ../DataModel -f net7.0
Move-Item -Path @("Models/Person.cs", "Models/Location.cs", "Models/Department.cs") ../DataModel
```

The first command creates a new Class Library project named `DataModel` targeting .NET 7.0.

The second command moves the relevant data model classes from the Advanced project to the new DataModel project.
??x

---

#### Adding References Between Projects
Background context: To enable communication and data sharing between the Blazor WebAssembly client-side application and the server-side ASP.NET Core project, references need to be added appropriately. This ensures that necessary classes are available in both projects.

:p How do you add a reference from one project to another?
??x
To add a reference from the BlazorWebAssembly project to the DataModel project, use the following command:

```powershell
dotnet add ../BlazorWebAssembly reference ../DataModel
```

This command adds a reference to the `DataModel` project in the `BlazorWebAssembly` project. This step is crucial because it makes the data model classes available for use in your Blazor WebAssembly application.
??x

#### Adding StaticWebAssetBasePath in BlazorWebAssembly.csproj File
Background context: When working with Blazor WebAssembly, it's crucial to configure the project file (`BlazorWebAssembly.csproj`) correctly. The `StaticWebAssetBasePath` property is used to specify the base path for static web assets served by the server-side application.

Adding this configuration helps in organizing and serving your static files properly when running a Blazor WebAssembly app on ASP.NET Core.

:p What does the StaticWebAssetBasePath attribute do in the BlazorWebAssembly.csproj file?
??x
The `StaticWebAssetBasePath` attribute specifies the base path that will be used to serve static assets (like images, JavaScript files, and CSS) from the server-side application. This is important because it ensures that these assets are served correctly under a specific path defined in your Blazor WebAssembly project.

For example, if you set `StaticWebAssetBasePath` to `/webassembly/`, then any static file referenced by the client-side will be accessible via this base path on the server.
```xml
<PropertyGroup>
    <TargetFramework>net6.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <StaticWebAssetBasePath>/webassembly/</StaticWebAssetBasePath>
</PropertyGroup>
```
x??

---
#### Running the Example Application
Background context: After setting up your Blazor WebAssembly project with the correct configuration, you need to test and run it to ensure that everything is set up as expected. This involves starting the ASP.NET Core server and navigating to a specific URL.

:p How do you start running the example application using Visual Studio?
??x
To start running the example application in Visual Studio:
1. Open your solution.
2. Select `Start Without Debugging` or `Run Without Debugging` from the `Debug` menu.
3. Alternatively, if you prefer to use the command prompt, navigate to the project folder and run the following command:

```shell
dotnet run
```

This command will start the ASP.NET Core server, allowing your Blazor WebAssembly application to be served over HTTP.

x??

---
#### Testing Placeholder Components in Browser
Background context: After running the application, you can test various components of your Blazor WebAssembly project. This includes placeholder content and dynamic parts of the app that might change based on user interactions or data fetching.

:p How do you navigate to see the placeholder content in a browser?
??x
To view the placeholder content in your browser:
1. Open any web browser.
2. Navigate to `http://localhost:5000/webassembly`.

This URL should display the placeholder content that was added by the template used to create your Blazor WebAssembly project.

x??

---
#### Testing Links and Dynamic Content
Background context: Once you have confirmed that the static assets are serving correctly, you can test dynamic parts of your application. This involves clicking on links or navigating through different components to ensure they behave as expected.

:p How do you click links and see dynamic content in the Blazor WebAssembly app?
??x
To interact with the Blazor WebAssembly app and view dynamic content:
1. Open a browser.
2. Navigate to `http://localhost:5000/webassembly`.
3. Click on the "Counter" and "Fetch Data" links.

These actions should display different content as defined by your application's logic, such as updating counters or fetching data from an API endpoint.

x??

---

---
#### Importing Namespaces into _Imports.razor
Background context: In Blazor WebAssembly, components often rely on shared namespaces for common functionalities like HTTP operations, routing, and UI elements. The `_Imports.razor` file is used to import these namespaces globally, making them accessible across all components without needing to declare `@using` in each component.

:p How do you ensure that the necessary namespaces are imported for a Blazor WebAssembly project?
??x
You need to add the required namespaces to the `_Imports.razor` file located in the root folder of your BlazorWebAssembly project. This approach ensures that the namespaces are available across all components without needing individual `@using` directives.

```csharp
@using System.Net.Http
@using System.Net.Http.Json
@using Microsoft.AspNetCore.Components.Forms
@using Microsoft.AspNetCore.Components.Routing
@using Microsoft.AspNetCore.Components.Web
@using Microsoft.AspNetCore.Components.Web.Virtualization
@using Microsoft.AspNetCore.Components.WebAssembly.Http
@using Microsoft.JSInterop
@using BlazorWebAssembly
@using BlazorWebAssembly.Shared
@using Advanced.Models
```
x??

---
#### Creating a Razor Component in Blazor WebAssembly
Background context: A Razor component is the basic building block for creating user interfaces in Blazor. In this example, you are creating a `List.razor` component to display a list of people from a data model.

:p How do you define and create a Razor component named `List.razor` in the Pages folder?
??x
To create a Razor component in the Pages folder, follow these steps:
1. Navigate to the `Pages` folder.
2. Create a new file named `List.razor`.
3. Define the content of the component within this file.

Here is an example of what the `List.razor` might look like:

```razor
@page "/forms"
@page "/forms/list"

<h5 class="bg-primary text-white text-center p-2">People (WebAssembly)</h5>
<table class="table table-sm table-striped table-bordered">
    <thead>
        <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Dept</th>
            <th>Location</th>
            <th></th>
        </tr>
    </thead>
    <tbody>
        @if (People.Count() == 0) {
            <tr>
                <th colspan="5" class="p-4 text-center">Loading Data...</th>
            </tr>
        } else {
            @foreach (Person p in People) {
                <tr>
                    <td>@p.PersonId</td>
                    <td>@p.Surname, @p.Firstname</td>
                    <td>@p.Department?.Name</td>
                    <td>@p.Location?.City</td>
                    <td class="text-center">
                        <NavLink class="btn btn-sm btn-info" href="@GetDetailsUrl(p.PersonId)">
                            Details
                        </NavLink>
                        <NavLink class="btn btn-sm btn-warning" href="@GetEditUrl(p.PersonId)">
                            Edit
                        </NavLink>
                        <button class="btn btn-sm btn-danger" @onclick="() => HandleDelete(p)">
                            Delete
                        </button>
                    </td>
                </tr>
            }
        }
    </tbody>
</table>
<NavLink class="btn btn-primary" href="forms/create">Create</NavLink>

@code {
    [Inject]
    public HttpClient? httpClient { get; set; }

    private List<Person> People = new List<Person>();

    protected override async Task OnInitializedAsync()
    {
        // Fetch data from the server
        People = await httpClient.GetFromJsonAsync<List<Person>>("api/people");
    }

    private string GetDetailsUrl(int id) => $"forms/details/{id}";
    private string GetEditUrl(int id) => $"forms/edit/{id}";

    private async Task HandleDelete(Person person)
    {
        // Logic to delete the person
        await httpClient.DeleteAsync($"api/people/{person.PersonId}");
        People.Remove(person);
    }
}
```
x??

---
#### Using Routing in Blazor WebAssembly Components
Background context: Blazor WebAssembly supports routing, which allows for organizing your application's content into different pages and sections. The `@page` directive is used to define the URL path that triggers a specific component.

:p How do you use the `@page` directive to route to a specific component in Blazor WebAssembly?
??x
To use the `@page` directive, you need to add it at the top of your Razor component file. The `@page` attribute defines the URL path that triggers the rendering of this component.

Here’s an example of using the `@page` directive:

```razor
@page "/forms"
@page "/forms/list"

<h5 class="bg-primary text-white text-center p-2">People (WebAssembly)</h5>
<!-- Component content -->
```

In this case, navigating to `/forms` or `/forms/list` in the browser will render the `List.razor` component.

x??

---
#### Handling Data with HttpClient in Blazor WebAssembly
Background context: The `HttpClient` class is used for making HTTP requests and handling responses from a server in Blazor applications. It’s commonly used to fetch data or perform CRUD operations.

:p How do you use the `HttpClient` service to fetch a list of people in a Blazor component?
??x
To use the `HttpClient` service, you need to inject it into your Razor component and then make HTTP requests using its methods.

Here's an example:

```razor
@page "/forms"
@page "/forms/list"

<h5 class="bg-primary text-white text-center p-2">People (WebAssembly)</h5>
<!-- Component content -->

@code {
    [Inject]
    public HttpClient? httpClient { get; set; }

    private List<Person> People = new List<Person>();

    protected override async Task OnInitializedAsync()
    {
        // Fetch data from the server
        People = await httpClient.GetFromJsonAsync<List<Person>>("api/people");
    }
}
```

In this example, `OnInitializedAsync` is called when the component initializes. It fetches a list of people using the `GetFromJsonAsync` method.

x??

---

#### Blazor WebAssembly Component Structure
Background context explaining that a Blazor WebAssembly component is essentially a C# class executed in the browser, similar to its server counterpart but with differences due to the client-side execution environment. The component includes properties for HTTP and an array of people, along with methods for updating data and handling actions like deletions.
:p What key difference exists between a Blazor WebAssembly component and a Blazor Server component?
??x
A Blazor WebAssembly component is executed in the browser, whereas a Blazor Server component runs on the server. This affects how HTTP requests are handled and how navigation URLs are constructed.
The answer explains that while both components share many core features like Razor directives and C# statements, the execution environment (browser vs. server) influences aspects such as URL construction and data retrieval methods.
```csharp
protected async override Task OnInitializedAsync() { await UpdateData(); }
```
x??

---

#### Navigation in Blazor WebAssembly Components
Background context explaining that navigation URLs in Blazor WebAssembly components should be relative, using a base element to define the root path. This ensures correct URL construction and prevents unintended navigations.
:p How are navigation URLs handled differently in Blazor WebAssembly compared to server-side Blazor?
??x
Navigation URLs in Blazor WebAssembly must use relative paths, which are combined with the application's root URL defined by a base element. Server-side Blazor can use absolute URLs without issues.
The answer explains that using relative URLs ensures navigation is performed correctly within the managed URL space of the Blazor WebAssembly app. Absolute URLs could lead to unexpected behavior or incorrect navigation.
```html
<NavLink class="btn btn-primary" href="forms/create">Create</NavLink>
```
x??

---

#### Data Retrieval in Blazor WebAssembly Components
Background context explaining that Blazor WebAssembly components use HTTP requests via HttpClient to interact with web services instead of Entity Framework Core. This change is necessary due to browser restrictions on SQL execution.
:p Why can't Blazor WebAssembly components use Entity Framework Core like their server-side counterparts?
??x
Blazor WebAssembly components cannot use Entity Framework Core because the browser environment restricts application code to HTTP requests and does not allow direct database access via SQL. Instead, they must consume web services to retrieve or modify data.
The answer provides context that the execution in a web assembly (browser) environment constrains the types of operations that can be performed, such as avoiding direct database interaction for security and performance reasons.
```csharp
People = await Http.GetFromJsonAsync<Person[]>("/api/people");
```
x??

---

#### HTTP Client Service in Blazor WebAssembly Components
Background context explaining that a service for HttpClient is created during the startup of a Blazor WebAssembly application, allowing components to inject and use this service through dependency injection.
:p How does dependency injection work with HttpClient in a Blazor WebAssembly project?
??x
Dependency injection allows components to receive an instance of the HttpClient class through its constructor or property. This setup is facilitated by the Blazor WebAssembly project's startup code, which registers HttpClient for dependency injection.
The answer explains that the `HttpClient` service can be injected into any component that needs it, enabling HTTP requests without needing to manually instantiate the client.
```csharp
@inject HttpClient Http
```
x??

---

#### HttpClient Component Interaction
Background context: The `List` component interacts with a web service using an injected `HttpClient`. This interaction is done primarily through HTTP methods provided by `HttpClient`.

:p How does the `List` component use `HttpClient` to delete objects?
??x
The `List` component uses the `DeleteAsync` method from `HttpClient` to send an HTTP DELETE request to the web service. The URL for this request includes the unique identifier (`PersonId`) of the object to be deleted.

```csharp
public async Task HandleDelete(Person p)
{
    if (Http != null) 
    {
        HttpResponseMessage resp = await Http.DeleteAsync($"/api/people/{p.PersonId}");
        
        // Check if the response indicates success.
        if (resp.IsSuccessStatusCode) 
        { 
            await UpdateData(); 
        }
    }
}
```
x??

---

#### Update Data Method
Background context: The `UpdateData` method is used to refresh data from the web service. This is done using an HTTP GET request.

:p How does the `List` component retrieve and update its local data?
??x
The `List` component uses the `GetFromJsonAsync<T>` extension method of `HttpClient`. It sends a GET request to the `/api/people` endpoint, expecting a JSON response that will be deserialized into an array of `Person` objects.

```csharp
private async Task UpdateData()
{
    if (Http != null) 
    {
        People = await Http.GetFromJsonAsync<Person[]>("/api/people") ?? Array.Empty<Person>();
    }
}
```
x??

---

#### HTTP Request Methods Overview
Background context: The `HttpClient` class provides several methods for sending different types of HTTP requests. These include `GetAsync`, `PostAsync`, `PutAsync`, `PatchAsync`, and `DeleteAsync`.

:p What are the main methods provided by HttpClient for making HTTP requests?
??x
The `HttpClient` class offers several methods to make HTTP requests, including:

- **GetAsync(url)**: Sends an HTTP GET request.
- **PostAsync(url, data)**: Sends an HTTP POST request with serialized data.
- **PutAsync(url, data)**: Sends an HTTP PUT request with serialized data.
- **PatchAsync(url, data)**: Sends an HTTP PATCH request with serialized data.
- **DeleteAsync(url)**: Sends an HTTP DELETE request.

These methods return a `Task<HttpResponseMessage>`, which can be used to check the status of the response and handle any necessary actions.

x??

---

#### HttpResponseMessage Properties
Background context: The `HttpResponseMessage` class contains properties that provide information about the response received from the server. These include `Content`, `HttpResponseHeaders`, `StatusCode`, and `IsSuccessStatusCode`.

:p What are some useful properties of the `HttpResponseMessage` class?
??x
Some useful properties of the `HttpResponseMessage` class include:

- **Content**: Returns the content returned by the server.
- **HttpResponseHeaders**: Returns the response headers.
- **StatusCode**: Returns the response status code.
- **IsSuccessStatusCode**: Returns true if the response status code is between 200 and 299, indicating a successful request.

x??

---

#### Extension Methods for HttpClient
Background context: The `HttpClient` class has extension methods that are particularly useful when working with JSON data. These include `GetFromJsonAsync<T>` for GET requests and `PostJsonAsync<T>`, `PutJsonAsync<T>` for POST and PUT requests, respectively.

:p What is the purpose of using extension methods like `GetFromJsonAsync<T>` in HttpClient?
??x
The extension methods like `GetFromJsonAsync<T>` are used to send HTTP GET requests and automatically deserialize the JSON response into a C# object. This simplifies working with data returned by web services.

```csharp
private async Task UpdateData()
{
    if (Http != null) 
    {
        People = await Http.GetFromJsonAsync<Person[]>("/api/people") ?? Array.Empty<Person>();
    }
}
```
x??

---

