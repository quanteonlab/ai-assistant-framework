# Flashcards: Pro-ASPNET-Core-7_processed (Part 171)

**Starting Chapter:** 37.4 Completing the Blazor WebAssembly Form application

---

#### HttpClient Class Usage in Blazor WebAssembly

Background context: The `HttpClient` class is a fundamental component for making HTTP requests from Blazor applications. It does not have scope or lifecycle issues and sends requests only when explicitly invoked by methods such as those described in Tables 37.2 and 37.4.

Relevant example: Suppose you need to fetch data from an API endpoint after deleting an object. You could requery the web service instead of simply removing the object from the local cache.

:p How should data be refreshed when an object is deleted using `HttpClient` in Blazor WebAssembly?
??x
Refreshing data when an object is deleted can be handled by making a new request to the API endpoint that provides the list of objects. For example:

```csharp
@code {
    async Task DeleteObject(int id)
    {
        await _httpClient.DeleteAsync($"api/objects/{id}");
        
        // Refresh the list after deletion
        var response = await _httpClient.GetAsync("api/objects");
        List<ObjectDto> updatedObjects = await response.Content.ReadFromJsonAsync<List<ObjectDto>>();
        
        // Update your local state or re-render the UI with the new data
    }
}
```

x??

---

#### Customizing Layout in Blazor WebAssembly

Background context: The default layout provided by the template includes navigation features. To customize this, you can create a new `LayoutComponentBase` and set it as the default layout through routing configuration.

Relevant example: Create an `EmptyLayout.razor` file to serve as your custom layout without any navigation features.

:p How do you apply a custom layout in Blazor WebAssembly?
??x
To apply a custom layout, create a new Razor component named `EmptyLayout.razor`. Then modify the `App.razor` routing configuration to use this new layout as default.

```csharp
@page "/"
<Router AppAssembly="typeof(App).Assembly">
    <Found Context="routeData">
        <RouteView RouteData="@routeData" DefaultLayout="@typeof(EmptyLayout)" />
        <FocusOnNavigate RouteData="@routeData" Selector="h1" />
    </Found>
    <NotFound>
        <PageTitle>Not found</PageTitle>
        <LayoutView Layout="@typeof(EmptyLayout)">
            <p role="alert">Sorry, there's nothing at this address.</p>
        </LayoutView>
    </NotFound>
</Router>
```

x??

---

#### Modifying CSS Styles in Blazor WebAssembly

Background context: The default project template includes Bootstrap CSS for styling and an additional stylesheet for error and validation elements. To use a custom Bootstrap CSS, you need to replace the existing link references.

Relevant example: Modify the `index.html` file located in the `wwwroot` folder.

:p How do you modify the CSS styles in Blazor WebAssembly?
??x
To modify the CSS styles, update the `<link>` tags in the `index.html` file as shown below:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
    <title>BlazorWebAssembly</title>
    <base href="/webassembly/" />
    <!-- Remove these lines to use the Bootstrap CSS from a CDN -->
    <!-- <link href="css/bootstrap/bootstrap.min.css" rel="stylesheet" /> -->
    <!-- <link href="css/app.css" rel="stylesheet" /> -->
    <link href="/lib/bootstrap/css/bootstrap.min.css" rel="stylesheet" />
    <link rel="icon" type="image/png" href="favicon.png" />
    <link href="BlazorWebAssembly.styles.css" rel="stylesheet" />
</head>
<body>
    <div id="app">
        <svg class="loading-progress">
            <circle r="40%" cx="50%" cy="50%" />
            <circle r="40%" cx="50%" cy="50%" />
        </svg>
        <div class="loading-progress-text"></div>
    </div>
    <div id="blazor-error-ui" class="text-center bg-danger h6 text-white p-2 fixed-top w-100" style="display:none">
        An unhandled error has occurred.
        <a href="" class="reload">Reload</a> 
        <a class="dismiss"></a>
    </div>
    <script src="_framework/blazor.webassembly.js"></script>
</body>
</html>
```

x??

---

#### Creating a Details Component
Background context: This section discusses creating a Details component for a Blazor WebAssembly application. The component displays person details and provides navigation links to edit or return to the list of forms.

:p What is the purpose of the `Details` component in this scenario?
??x
The `Details` component serves to display detailed information about a specific person, including their ID, first name, surname, department, and location. It also includes navigation links to either edit the current record or return to the list of forms.

This component is designed to fetch data from an API using HTTPClient and then render it in a disabled input field for viewing purposes only.
```razor
@page "/forms/details/{id:long}"
<h4 class="bg-info text-center text-white p-2">Details (WebAssembly)</h4>
<div class="form-group">
    <label>ID</label>
    <input class="form-control" value="@PersonData.PersonId" disabled />
</div>
<div class="form-group">
    <label>Firstname</label>
    <input class="form-control" value="@PersonData.Firstname" disabled />
</div>
<div class="form-group">
    <label>Surname</label>
    <input class="form-control" value="@PersonData.Surname" disabled />
</div>
<div class="form-group">
    <label>Department</label>
    <input class="form-control" 
        value="@($"@PersonData.Department?.Name")" 
        disabled />
</div>
<div class="form-group">
    <label>Location</label>
    <input class="form-control"
            value="\"{PersonData.Location?.City}, \" + PersonData.Location?.State" 
            disabled />
</div>
<div class="text-center p-2">
    <NavLink class="btn btn-info" href="@EditUrl">Edit</NavLink>
    <NavLink class="btn btn-secondary" href="forms">Back</NavLink>
</div>

@code {
    [Inject]
    public NavigationManager? NavManager { get; set; }
    
    [Inject]
    public HttpClient? Http { get; set; }

    [Parameter]
    public long Id { get; set; }

    public Person PersonData { get; set; } = new Person();

    protected async override Task OnParametersSetAsync() {
        if (Http == null) {
            PersonData = await Http.GetFromJsonAsync<Person>( 
                $"/api/people/{Id}") 
                ?? new();
        }
    }

    public string EditUrl => $"forms/edit/{Id}";
}
```
x??

---
#### Creating an Editor Component
Background context: This section covers creating an Editor component for a Blazor WebAssembly application, which is used to create or edit person records. The component includes validation and dropdown options for department and location.

:p What does the `Editor` component do in this scenario?
??x
The `Editor` component allows users to either create new or edit existing person records. It supports form validation, dropdown selection for departments and locations, and submission handling through an EditForm.

The component is designed to be flexible by supporting both "Create" and "Edit" modes.
```razor
@page "/forms/edit/{id:long}"
@page "/forms/create"
<link href="/blazorValidation.css" rel="stylesheet" />
<h4 class="bg-@Theme text-center text-white p-2">@Mode (WebAssembly)</h4>
<EditForm Model="PersonData" OnValidSubmit="HandleValidSubmit">
    <DataAnnotationsValidator />
    @if (Mode == "Edit") {
        <div class="form-group">
            <label>ID</label>
            <InputNumber class="form-control" 
                @bind-Value="PersonData.PersonId" readonly />
        </div>
    }
    <div class="form-group">
        <label>Firstname</label>
        <ValidationMessage For="() => PersonData.Firstname" />
        <InputText class="form-control"
            @bind-Value="PersonData.Firstname" />
    </div>
    <div class="form-group">
        <label>Surname</label>
        <ValidationMessage For="() => PersonData.Surname" />
        <InputText class="form-control"
            @bind-Value="PersonData.Surname" />
    </div>
    <div class="form-group">
        <label>Department</label>
        <ValidationMessage For="() => PersonData.DepartmentId" />
        <select @bind="PersonData.DepartmentId" class="form-control">
            <option selected disabled value="0">Choose a Department</option>
            @foreach (var kvp in Departments) {
                <option value="@kvp.Value">@kvp.Key</option>
            }
        </select>
    </div>
    <div class="form-group">
        <label>Location</label>
        <ValidationMessage For="() => PersonData.LocationId" />
        <select @bind="PersonData.LocationId" class="form-control">
            <option selected disabled value="0">Choose a Location</option>
            @foreach (var kvp in Locations) {
                <option value="@kvp.Value">@kvp.Key</option>
            }
        </select>
    </div>
    <div class="text-center p-2">
        <button type="submit" class="btn btn-@Theme">Save</button>
        <NavLink class="btn btn-secondary" href="forms">Back</NavLink>
    </div>
</EditForm>

@code {
    [Inject]
    public HttpClient? Http { get; set; }

    private string Mode { get; set; } = "Create";
    
    // Sample properties and methods
    public Person PersonData { get; set; } = new Person();
}
```
x??

---
#### HTTPClient for Data Fetching in Blazor WebAssembly
Background context: This section explains how to use the `HttpClient` service within a Blazor WebAssembly component to fetch data from an API.

:p How does the `Details` component use `HttpClient`?
??x
The `Details` component uses the `HttpClient` service to fetch person details from the backend API. When the component is initialized, it checks if the `Http` service is not null and then makes a GET request to retrieve the person data using the ID passed as a parameter.

This ensures that the component can dynamically load data based on user interaction.
```razor
protected async override Task OnParametersSetAsync() {
    if (Http != null) {
        PersonData = await Http.GetFromJsonAsync<Person>(
            $"api/people/{Id}") 
            ?? new();
    }
}
```
x??

---
#### Navigation Links in Blazor WebAssembly Components
Background context: This section describes how navigation links are used within a Blazor WebAssembly component to provide user interactions like editing or returning to the main form list.

:p How do navigation links work in the `Details` and `Editor` components?
??x
Navigation links in both the `Details` and `Editor` components allow users to navigate between different parts of the application. In the `Details` component, these links enable switching from viewing details to editing or returning to the main list.

Similarly, in the `Editor` component, navigation links provide a way for users to either save their changes by submitting the form or return to the main list without saving.
```razor
<div class="text-center p-2">
    <NavLink class="btn btn-info" href="@EditUrl">Edit</NavLink>
    <NavLink class="btn btn-secondary" href="forms">Back</NavLink>
</div>

@code {
    public string EditUrl => $"forms/edit/{Id}";
}
```
x??

---

---
#### Blazor WebAssembly Overview
Blazor WebAssembly creates client-side applications that do not need to maintain a persistent connection to the ASP.NET Core server. This approach allows for more efficient use of resources and better performance, as the application runs entirely on the user's browser.
:p What does Blazor WebAssembly enable in terms of application development?
??x
Blazor WebAssembly enables the creation of client-side applications that can run without maintaining a persistent connection to the ASP.NET Core server. This means that the application logic, including form handling and data manipulation, runs directly in the user's browser, reducing server load and enhancing performance.
x??

---
#### Dependency Injection for HTTP Client
In Blazor WebAssembly, data access must be performed through an `HttpClient` object received via dependency injection. This allows for dynamic configuration of the client’s behavior and easier management of dependencies across different parts of the application.
:p How is the `HttpClient` used in a Blazor WebAssembly application?
??x
The `HttpClient` is used to perform HTTP requests such as GET, POST, PUT, and DELETE operations from within a Blazor component. It is typically injected into a component via dependency injection so that it can be shared across different parts of the application for consistent behavior.
```csharp
[Inject]
public HttpClient Http { get; set; }
```
x??

---
#### Parameter Binding in Components
In this context, parameters are bound to properties within the component. The `Id` parameter is used to determine whether a new item or an existing one should be edited. This binding allows for dynamic behavior based on the current route.
:p What determines whether a form is for creating a new item or editing an existing one?
??x
The determination of whether a form is for creating a new item or editing an existing one is made based on the `Id` parameter passed to the component. If `Id` equals 0, it indicates that a new item should be created; otherwise, it will edit the existing item.
```csharp
public string Mode => Id == 0 ? "Create" : "Edit";
```
x??

---
#### Asynchronous Operations in Blazor Components
The `OnParametersSetAsync` method is used to handle asynchronous operations such as fetching data from a web service. This ensures that the component has all necessary parameters set before performing any asynchronous actions.
:p How does the `OnParametersSetAsync` method contribute to the functionality of a Blazor component?
??x
The `OnParametersSetAsync` method contributes by ensuring that the component performs necessary asynchronous operations, such as fetching data from a web service, only after all parameters have been set. This ensures that the component is in a valid state before performing any potentially lengthy or network-dependent tasks.
```csharp
protected async override Task OnParametersSetAsync() {
    if (Http != null) {
        if (Mode == "Edit") {
            PersonData = await Http.GetFromJsonAsync<Person>(...);
        }
        // Other fetch operations...
    }
}
```
x??

---
#### Handling Form Submission in Blazor Components
The `HandleValidSubmit` method handles the submission of a form, performing either a POST or PUT operation to save the data. This ensures that the appropriate HTTP method is used based on whether the user is creating a new item or editing an existing one.
:p How does the `HandleValidSubmit` method handle form submissions in Blazor WebAssembly?
??x
The `HandleValidSubmit` method handles form submissions by determining the appropriate HTTP request (POST for creation, PUT for updating) and sending it to the server. It then navigates away from the current page after successfully submitting the form.
```csharp
public async Task HandleValidSubmit() {
    if (Http != null) {
        if (Mode == "Create") {
            await Http.PostAsJsonAsync("/api/people", PersonData);
        } else {
            await Http.PutAsJsonAsync("/api/people", PersonData);
        }
        NavManager?.NavigateTo("forms");
    }
}
```
x??

---
#### Component Navigation in Blazor
The `NavigationManager` is used to navigate between different pages or views within the application. In this example, it is used to navigate back to a list of forms after submitting changes.
:p How does the `NavigationManager` contribute to component navigation?
??x
The `NavigationManager` contributes by providing an easy way to navigate between different routes and pages within the Blazor application. It allows components to redirect users to other parts of the application based on specific actions, such as form submission.
```csharp
public NavigationManager? NavManager { get; set; }
// Usage in HandleValidSubmit:
NavManager?.NavigateTo("forms");
```
x??

---
#### Conditional Rendering and Styling
The `Theme` property is used to conditionally apply different styles based on the state of the component. This allows for dynamic UI updates based on whether a new item or an existing one is being edited.
:p How does conditional styling work in this Blazor component?
??x
Conditional styling works by using the `Theme` property to determine which CSS class should be applied to the component. If `Id` equals 0, it indicates that a new item is being created and applies the "primary" theme; otherwise, it applies the "warning" theme.
```csharp
public string Theme => Id == 0 ? "primary" : "warning";
```
x??

---

#### Setting Up ASP.NET Core Identity
Background context: This section explains how to integrate ASP.NET Core Identity into an existing project, setting up a database and managing user accounts and roles. ASP.NET Core Identity provides a framework for authentication and authorization features, supporting various models such as two-factor authentication and single sign-on.

:p What is the first step in preparing an application for using ASP.NET Core Identity?
??x
The first step is to drop the existing database that might interfere with setting up the new Identity database. This is done by running a specific command in a PowerShell prompt.
```powershell
dotnet ef database drop --force
```
x??

---

#### Installing ASP.NET Core Identity Packages
Background context: Installing necessary packages is crucial for using ASP.NET Core Identity. The package `Microsoft.AspNetCore.Identity.EntityFrameworkCore` needs to be added to the project, enabling the use of Entity Framework Core with Identity.

:p How do you install the required ASP.NET Core Identity package?
??x
The required package can be installed by running the following command in a PowerShell prompt within the project directory:
```powershell
dotnet add package Microsoft.AspNetCore.Identity.EntityFrameworkCore --version 7.0.0
```
Alternatively, it can be installed through the NuGet Package Manager UI in Visual Studio.
x??

---

#### Creating IdentityContext Class
Background context: The `IdentityContext` class is an Entity Framework Core context that interacts with the database to manage user and role data. It extends the `IdentityDbContext<IdentityUser>` provided by ASP.NET Core Identity.

:p What does the `IdentityContext` class in the code look like?
??x
The `IdentityContext` class looks as follows:
```csharp
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;

namespace Advanced.Models {
    public class IdentityContext: IdentityDbContext<IdentityUser> {
        public IdentityContext(DbContextOptions<IdentityContext> options) : base(options) { }
    }
}
```
It uses `IdentityDbContext` from the ASP.NET Core Identity package, specifying `IdentityUser` as the generic type argument.
x??

---

#### Configuring Connection Strings
Background context: Proper configuration of connection strings is necessary to link the application with the database used by ASP.NET Core Identity. Multiple connection strings can be defined in the `appsettings.json` file.

:p How do you add a connection string for the Identity database?
??x
A connection string for the Identity database is added to the `appsettings.json` file as follows:
```json
{
  "ConnectionStrings": {
    "IdentityConnection": "Server=(localdb)\\MSSQLLocalDB;Database=Identity;MultipleActiveResultSets=True"
  }
}
```
This ensures that ASP.NET Core Identity knows where to store user and role data.
x??

---

#### Configuring Services for Identity
Background context: Once the package is installed, services need to be configured in the `Program.cs` file to properly integrate Identity into the application. This includes setting up database contexts and identity providers.

:p How do you configure ASP.NET Core services for Identity?
??x
Services are configured as follows:
```csharp
builder.Services.AddDbContext<IdentityContext>(opts => 
    opts.UseSqlServer(builder.Configuration["ConnectionStrings:IdentityConnection"]));

builder.Services.AddIdentity<IdentityUser, IdentityRole>()
    .AddEntityFrameworkStores<IdentityContext>();
```
These lines ensure that the `IdentityContext` is set up and that user and role data are stored in the specified database.
x??

---

#### Summary of Configurations
Background context: This summary covers all steps required to integrate ASP.NET Core Identity into a project, including setting up the database, adding necessary packages, configuring connection strings, and services.

:p What key steps are involved in preparing an application for using ASP.NET Core Identity?
??x
Key steps involve:
1. Dropping any existing databases.
2. Installing the `Microsoft.AspNetCore.Identity.EntityFrameworkCore` package.
3. Adding a connection string to the `appsettings.json`.
4. Configuring the database context and identity services in the `Program.cs` file.

These steps ensure that the application is ready for user management features provided by ASP.NET Core Identity.
x??

