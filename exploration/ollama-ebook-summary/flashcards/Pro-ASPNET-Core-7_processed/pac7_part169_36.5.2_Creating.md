# Flashcards: Pro-ASPNET-Core-7_processed (Part 169)

**Starting Chapter:** 36.5.2 Creating a valid-only submit button component

---

#### Custom Validation Component

Background context explaining the concept. The `DepartmentStateValidator` component restricts location choices based on department and state, ensuring data consistency.

:p What is the purpose of the `DepartmentStateValidator` component?
??x
The `DepartmentStateValidator` ensures that only valid locations can be selected for a specific department and state. In this case, it restricts selection to California for the Development department.
x??

---

#### ValidButton Component

Background context explaining the concept. The `ValidButton` component renders a submit button that is enabled only when the form data is valid.

:p How does the `ValidButton` component ensure the button is disabled if there are validation issues?
??x
The `ValidButton` component checks the current `EditContext` for any validation messages upon initialization and whenever the validation state changes. If there are no validation messages, the button remains enabled; otherwise, it is disabled.

Code Example:
```csharp
@code {
    [CascadingParameter]
    public EditContext? CurrentEditContext { get; set; }

    public bool Disabled { get; set; } = true;

    protected override void OnInitialized()
    {
        SetButtonState();
        if (CurrentEditContext != null)
        {
            CurrentEditContext.OnValidationStateChanged += (sender, args) => SetButtonState();
            CurrentEditContext.Validate(); // Ensure initial validation state is checked
        }
    }

    public void SetButtonState()
    {
        Disabled = CurrentEditContext?.GetValidationMessages().Any() ?? false;
    }
}
```

x??

---

#### Handling Validation in Blazor Forms

Background context explaining the concept. The `EditForm` component handles form validation and provides mechanisms to enable buttons only when data is valid.

:p How does the `ValidButton` component determine if it should be disabled?
??x
The `ValidButton` component determines if it should be disabled by checking for any validation messages in the current `EditContext`. If there are no validation messages, the button is enabled; otherwise, it remains disabled.

Explanation:
```csharp
public string ButtonClass => 
    Disabled ? $"btn btn-{BtnTheme} {DisabledClass} mt-2" 
             : $"btn btn-{BtnTheme} mt-2";
```

x??

---

#### Cascading Parameters in Blazor

Background context explaining the concept. The `@CascadingParameter` directive allows components to access parent component properties.

:p How does the `ValidButton` component use cascading parameters?
??x
The `ValidButton` component uses the `@CascadingParameter` directive to access the `CurrentEditContext` from its parent component, which is essential for determining if the form data is valid.

Code Example:
```csharp
[CascadingParameter]
public EditContext? CurrentEditContext { get; set; }
```

x??

---

#### Managing Validation State

Background context explaining the concept. The `OnValidationStateChanged` event allows components to react to changes in validation state.

:p How does the `ValidButton` component handle form validation events?
??x
The `ValidButton` component subscribes to the `OnValidationStateChanged` event of the `CurrentEditContext`. Whenever the validation state changes, the `SetButtonState` method is called to update the button's enabled/disabled status.

Code Example:
```csharp
public void SetButtonState()
{
    if (CurrentEditContext != null)
    {
        Disabled = CurrentEditContext.GetValidationMessages().Any();
    }
}
```

x??

---

#### Entity Framework Core Context Scopes

Background context explaining the concept. Proper management of Entity Framework Core contexts is crucial to avoid stale data issues.

:p How does Entity Framework Core context scope affect form data?
??x
Entity Framework Core context scopes must be carefully managed to prevent accessing outdated or stale data. Managing these scopes through `IDisposable` interfaces or automatic dependency injection ensures that the database operations are performed with up-to-date and fresh data.

Explanation:
```csharp
// Example of using IDisposable interface for scope management
public class MyDbContext : DbContext, IDisposable
{
    // Context implementation

    public void Dispose()
    {
        ((IDisposable)ChangeTracker).Dispose();
        ((IDisposable)SaveChangesTokenSource.Token).Dispose();
    }
}
```

x??

---

#### Introduction to Blazor WebAssembly
Background context: This section introduces Blazor WebAssembly, an implementation of Blazor that runs inside a browser using WebAssembly. WebAssembly allows high-level languages like C# to be compiled into a low-level language-neutral format for execution at near-native performance.

:p What is Blazor WebAssembly and how does it differ from traditional Blazor?
??x
Blazor WebAssembly is an implementation of Blazor that runs in the browser using WebAssembly. It breaks the dependency on the server, executing the Blazor application entirely within the browser without a persistent HTTP connection. This contrasts with Blazor Server, which requires server-side execution and maintains a continuous HTTP connection.

```csharp
// Example of a DataController.cs class definition
using Advanced.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace Advanced.Controllers {
    [ApiController]
    [Route("/api/people")]
    public class DataController : ControllerBase {
        private DataContext context;
        public DataController(DataContext ctx) { context = ctx; }
        
        // Additional methods for managing people data
    }
}
```
x??

---

#### WebAssembly Support
Background context: The chapter discusses the current limitations of WebAssembly, including support only in modern browsers and restricted access to certain ASP.NET Core features.

:p Why are there restrictions when using Blazor WebAssembly?
??x
Blazor WebAssembly is limited by browser support for WebAssembly. Only the latest browser versions can execute WebAssembly. Additionally, because WebAssembly applications run entirely in the browser, they are restricted to the APIs that browsers provide, limiting access to certain ASP.NET Core features like Entity Framework Core.

```csharp
// Example of a DataController method handling HTTP GET requests
[HttpGet]
public IEnumerable<Person> GetAll() {
    IEnumerable<Person> people = context.People
        .Include(p => p.Department)
        .Include(p => p.Location);
    foreach (Person p in people) { 
        if (p.Department?.People == null) { 
            p.Department.People = null; 
        } 
        if (p.Location?.People == null) { 
            p.Location.People = null; 
        } 
    }
    return people;
}
```
x??

---

#### Blazor Components in WebAssembly
Background context: This section explains how to create and use Blazor components within a project dedicated to WebAssembly.

:p How are Blazor components used in a WebAssembly application?
??x
Blazor components are added to a project that is specifically configured for Blazor WebAssembly. These components can be created using C# and are then integrated into the user interface of the web application, providing a way to structure and organize UI elements.

```csharp
// Example of a simple Blazor component class
public partial class PersonComponent : ComponentBase {
    // Properties, methods, and other logic for the component
}
```
x??

---

#### Navigating Between Components
Background context: The chapter covers navigating between components in a WebAssembly application, which is similar to traditional client-side applications.

:p How can you navigate between Blazor components?
??x
Navigating between Blazor components involves using navigation links or buttons that trigger the router to change the current component being rendered. This navigation can be implemented by modifying the router configuration or programmatically navigating within a method.

```csharp
// Example of navigating to a different route in C#
@code {
    void NavigateToDetails(long id) {
        NavigationManager.NavigateTo($"/people/{id}");
    }
}
```
x??

---

#### Forms Application with Blazor WebAssembly
Background context: The chapter demonstrates creating a forms application using Blazor components, showcasing the use of input fields and form handling in client-side applications.

:p How can you create a simple form in Blazor WebAssembly?
??x
Creating a simple form involves defining input elements (like text boxes) and handling user interactions to submit data. Forms can be managed by binding input values to component properties and using event handlers for validation or submission logic.

```csharp
// Example of a simple form with input fields
<EditForm Model="@person">
    <InputText @bind-Value="person.Name" />
    <InputNumber @bind-Value="person.Age" />
    <button type="submit">Submit</button>
</EditForm>

@code {
    private Person person = new Person();
    
    // Handle form submission
    async Task SubmitForm() {
        await Save(person);
    }
}
```
x??

---

#### Limitations and Alternatives
Background context: The chapter discusses the limitations of Blazor WebAssembly, such as browser support and restricted access to certain features, and provides alternatives if needed.

:p What are the main limitations of using Blazor WebAssembly?
??x
The main limitations include:
- Limited browser support for WebAssembly.
- Restricted access to ASP.NET Core features like Entity Framework Core due to browser constraints.
- Larger initial download size compared to traditional client-side frameworks.

These limitations can make it unsuitable for projects that need broad browser compatibility or extensive use of .NET Core features.

```csharp
// Example of handling a form submission and saving data
[HttpPost]
public async Task Save([FromBody] Person p) {
    await context.People.AddAsync(p);
    await context.SaveChangesAsync();
}
```
x??

---

---
#### Asynchronous Database Operations
This section describes how to handle asynchronous operations for database interactions, specifically focusing on deleting a `Person` object and fetching `Location` and `Department` objects. The use of `async/await` ensures that these operations do not block other tasks while the database is being accessed.
:p What are the key features of handling database operations asynchronously in this controller?
??x
In this controller, we handle database operations asynchronously to ensure that they do not block other tasks. For instance, when deleting a `Person`, the operation is performed using `async/await` to run the deletion task without blocking other processes. Similarly, fetching `Location` and `Department` objects are handled in an asynchronous manner.
```csharp
public async Task Delete(long id) {
    context.People.Remove(new Person() { PersonId = id });
    await context.SaveChangesAsync();
}
[HttpGet("/api/locations")]
public IAsyncEnumerable<Location> GetLocations() => 
    context.Locations.AsAsyncEnumerable();
[HttpGet("/api/departments")]
public IAsyncEnumerable<Department> GetDepts() =>
    context.Departments.AsAsyncEnumerable();
```
x??

---
#### Dropping the Database
This section explains how to drop a database using Entity Framework Core commands. The `dotnet ef database drop --force` command is used to remove the current database, ensuring that any existing data is deleted.
:p How can you use Entity Framework Core to drop a database?
??x
To drop a database using Entity Framework Core, you can run the following command in a PowerShell window:
```
dotnet ef database drop --force
```
This command removes the specified database and its schema. The `--force` option ensures that the operation proceeds even if there are existing connections to the database.
x??

---
#### Running the Application with Browser Requests
The text mentions running an application and making HTTP requests using a browser. This is done by starting the application with `dotnet run` and then accessing specific API endpoints via URL in a web browser.
:p How do you start and access your application using a browser?
??x
To start the application, you use the following command:
```
dotnet run
```
Once the application is running, you can make HTTP requests to it using a web browser. For example, to fetch all `Person` objects from the database, you would navigate to:
```
http://localhost:5000/api/people
```
This will return a JSON representation of the `Person` objects stored in the database.
x??

---
#### Example of Fetching Data via API
The text provides an example of fetching data via API endpoints. The specific endpoint `/api/locations` is used to fetch all `Location` entities from the database.
:p What does the `/api/locations` endpoint do?
??x
The `/api/locations` endpoint is designed to return all `Location` objects stored in the database. It uses asynchronous programming to ensure that the retrieval of these entities is non-blocking and efficient.
```csharp
[HttpGet("/api/locations")]
public IAsyncEnumerable<Location> GetLocations() => 
    context.Locations.AsAsyncEnumerable();
```
This method converts the `IQueryable<Location>` result into an `IAsyncEnumerable<Location>` which can be iterated over asynchronously, providing a non-blocking way to fetch and process data.
x??

---

