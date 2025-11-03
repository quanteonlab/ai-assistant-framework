# Flashcards: Pro-ASPNET-Core-7_processed (Part 167)

**Starting Chapter:** 36.2.3 Handling form events

---

#### Blazor Validation Features Overview
Background context explaining the validation features available in Blazor. These features help ensure that form data is correct before submission.

:p What are the main validation events associated with `EditForm`?
??x
The main validation events associated with `EditForm` include:
- `OnValidSubmit`: Triggered when the form is submitted and passes all validations.
- `OnInvalidSubmit`: Triggered when the form is submitted and fails any validations.
- `OnSubmit`: Triggered before validation occurs, allowing you to handle submission actions regardless of validation results.

```razor
@code {
    private string FormSubmitMessage = "";

    public void HandleValidSubmit()
    {
        FormSubmitMessage = "Form submitted successfully.";
    }

    public void HandleInvalidSubmit()
    {
        FormSubmitMessage = "Form submission failed due to invalid data.";
    }
}
```
x??

---

#### Handling `EditForm` Events
Background context explaining how the `EditForm` component in Blazor handles form events, including user interaction and validation.

:p How does the `EditForm` component handle form submission events?
??x
The `EditForm` component handles form submission by triggering specific events based on whether the form data passes or fails validation. The `OnValidSubmit`, `OnInvalidSubmit`, and `OnSubmit` events are triggered in different scenarios:

- `OnValidSubmit`: Triggered when the form is submitted and all validations pass.
- `OnInvalidSubmit`: Triggered when the form is submitted but fails any of the validation rules.
- `OnSubmit`: Triggered before validation to allow custom handling regardless of validation results.

Here's an example of how these events are handled:

```razor
@page "/forms/edit/{id:long}"
@layout EmptyLayout

<h4 class="bg-primary text-center text-white p-2">Edit</h4>
<h6 class="bg-info text-center text-white p-2">@FormSubmitMessage</h6>

<FormSpy PersonData="PersonData">
    <EditForm Model="PersonData" OnValidSubmit="HandleValidSubmit"
              OnInvalidSubmit="HandleInvalidSubmit">
        <DataAnnotationsValidator />
        <ValidationSummary />
        <!-- Form input fields go here -->
        <div class="text-center">
            <button type="submit" class="btn btn-primary mt-2">Submit</button>
            <NavLink class="btn btn-secondary mt-2" href="/forms">Back</NavLink>
        </div>
    </EditForm>
</FormSpy>

@code {
    private string FormSubmitMessage = "";
    PersonData PersonData { get; set; } = new PersonData();

    public void HandleValidSubmit()
    {
        FormSubmitMessage = "Form submitted successfully.";
    }

    public void HandleInvalidSubmit()
    {
        FormSubmitMessage = "Form submission failed due to invalid data.";
    }
}
```

x??

---

#### Using `EditForm` with Form Events
Background context explaining the integration of form events within an `EditForm`. This is essential for custom handling based on validation status.

:p How can you add a submit button and handle form submission events in an `EditForm`?
??x
To add a submit button and handle form submission events, you need to define buttons inside the `EditForm` component and use the appropriate event handlers. The `OnValidSubmit`, `OnInvalidSubmit`, and `OnSubmit` methods are used for different scenarios.

Here's how you can do it:

```razor
@page "/forms/edit/{id:long}"
@layout EmptyLayout

<h4 class="bg-primary text-center text-white p-2">Edit</h4>
<h6 class="bg-info text-center text-white p-2">@FormSubmitMessage</h6>

<FormSpy PersonData="PersonData">
    <EditForm Model="PersonData" OnValidSubmit="HandleValidSubmit"
              OnInvalidSubmit="HandleInvalidSubmit">
        <DataAnnotationsValidator />
        <ValidationSummary />
        <!-- Form input fields go here -->
        <div class="text-center">
            <button type="submit" class="btn btn-primary mt-2">Submit</button>
            <NavLink class="btn btn-secondary mt-2" href="/forms">Back</NavLink>
        </div>
    </EditForm>
</FormSpy>

@code {
    private string FormSubmitMessage = "";
    PersonData PersonData { get; set; } = new PersonData();

    public void HandleValidSubmit()
    {
        FormSubmitMessage = "Form submitted successfully.";
    }

    public void HandleInvalidSubmit()
    {
        FormSubmitMessage = "Form submission failed due to invalid data.";
    }
}
```

x??

---

#### Customizing `EditForm` with Validation
Background context explaining how to customize the validation process in an `EditForm`. This includes adding custom select components and handling validation messages.

:p How can you add custom validation logic, such as a department or location selection, within an `EditForm`?
??x
To add custom validation logic, like selecting a department or location, within an `EditForm`, you need to use custom select components and ensure that the selected values are validated correctly. Here's how you can implement this:

```razor
@page "/forms/edit/{id:long}"
@layout EmptyLayout

<h4 class="bg-primary text-center text-white p-2">Edit</h4>
<h6 class="bg-info text-center text-white p-2">@FormSubmitMessage</h6>

<FormSpy PersonData="PersonData">
    <EditForm Model="PersonData" OnValidSubmit="HandleValidSubmit"
              OnInvalidSubmit="HandleInvalidSubmit">
        <DataAnnotationsValidator />
        <ValidationSummary />
        <!-- Form input fields go here -->
        <div class="form-group">
            <label>Person ID</label>
            <InputNumber class="form-control" @bind-Value="PersonData.PersonId" disabled />
        </div>

        <div class="form-group">
            <label>Firstname</label>
            <ValidationMessage For="@(() => PersonData.Firstname)" />
            <InputText class="form-control" @bind-Value="PersonData.Firstname" />
        </div>

        <div class="form-group">
            <label>Surname</label>
            <ValidationMessage For="@(() => PersonData.Surname)" />
            <InputText class="form-control" @bind-Value="PersonData.Surname" />
        </div>

        <div class="form-group">
            <label>Dept ID</label>
            <ValidationMessage For="@(() => PersonData.DepartmentId)" />
            <CustomSelect TValue="long" Values="Departments"
                          Parser="@((string str) => long.Parse(str))"
                          @bind-Value="PersonData.DepartmentId">
                <option selected disabled value="0">Choose a Department</option>
            </CustomSelect>
        </div>

        <div class="form-group">
            <label>Location ID</label>
            <ValidationMessage For="@(() => PersonData.LocationId)" />
            <CustomSelect TValue="long" Values="Locations"
                          Parser="@((string str) => long.Parse(str))"
                          @bind-Value="PersonData.LocationId">
                <option selected disabled value="0">Choose a Location</option>
            </CustomSelect>
        </div>

        <div class="text-center">
            <button type="submit" class="btn btn-primary mt-2">Submit</button>
            <NavLink class="btn btn-secondary mt-2" href="/forms">Back</NavLink>
        </div>
    </EditForm>
</FormSpy>

@code {
    private string FormSubmitMessage = "";
    PersonData PersonData { get; set; } = new PersonData();

    public void HandleValidSubmit()
    {
        FormSubmitMessage = "Form submitted successfully.";
    }

    public void HandleInvalidSubmit()
    {
        FormSubmitMessage = "Form submission failed due to invalid data.";
    }
}
```

x??

---
---

#### Entity Framework Core Context Scope Issue in Blazor Applications
Background context: In a conventional ASP.NET Core application, each request has its own scoped Entity Framework Core context. However, in Blazor applications, URL changes can alter components without sending new HTTP requests, leading to shared dependency injection scopes among multiple components. This means that changes made by one component affect other components even if those changes are not written to the database.
:p How does the shared context issue manifest in a Blazor application?
??x
The shared context issue manifests when a user makes edits on a form but then navigates back without saving, and those edits persist across multiple components. This is because the context retains these changes even if they have not been committed to the database.
```razor
// Example of handling invalid submit in Blazor
public void HandleInvalidSubmit() => FormSubmitMessage = "Invalid Data Submitted";
```
x??

---
#### Handling Validation Errors and Submit Messages in Blazor Forms
Background context: The code snippet includes methods for handling valid and invalid form submissions. These methods set the `FormSubmitMessage` property to indicate whether data was submitted with errors or successfully.
:p What is the purpose of `HandleValidSubmit` and `HandleInvalidSubmit` methods?
??x
The purpose of these methods is to update the `FormSubmitMessage` property based on the result of form submission. `HandleValidSubmit` sets the message when valid data is submitted, while `HandleInvalidSubmit` updates the message with an error when invalid data is submitted.
```razor
// Example of setting FormSubmitMessage for invalid submit
public void HandleInvalidSubmit() => FormSubmitMessage = "Invalid Data Submitted";
```
x??

---
#### Discarding Unsaved Changes in Blazor Components
Background context: To prevent unsaved changes from persisting across components, the `Dispose` method can be implemented to detach entities when a component is destroyed. This ensures that any unsaved data is not used for future requests.
:p How does implementing the `IDisposable` interface help manage unsaved data in Blazor applications?
??x
Implementing the `IDisposable` interface allows components to properly dispose of their context and detach entities when they are about to be destroyed, preventing unsaved changes from being used for future requests. This is particularly useful in scenarios where shared contexts might retain temporary or invalid data.
```razor
// Example of Dispose method implementation
public void Dispose() {
    if (Context != null) {
        Context.Entry(PersonData).State = EntityState.Detached;
    }
}
```
x??

---
#### Using Scoped Services for Components in Blazor Applications
Background context: To ensure that each component has its own scoped Entity Framework Core context, the `@inherits` directive can be used to derive from a base class that provides a scoped service. This prevents shared contexts from retaining unsaved changes across components.
:p How does using a scoped service with the `@inherits` directive help manage data in Blazor applications?
??x
Using the `@inherits` directive with a base class that provides a scoped service ensures that each component has its own Entity Framework Core context. This prevents shared contexts from retaining unsaved changes, ensuring that only committed data is used for future requests.
```razor
// Example of using OwningComponentBase to manage scoped services
@page "/forms/edit/{id:long}" @layout EmptyLayout @inherits OwningComponentBase
```
x??

---
#### Understanding the Use of `OwningComponentBase` in Blazor Applications
Background context: The `OwningComponentBase` class provides a convenient way to use scoped services within components. It defines properties that can be used to access services specific to the component’s lifecycle.
:p What is the purpose of using the `OwningComponentBase` class?
??x
The purpose of using the `OwningComponentBase` class is to provide a base class for components that need to use scoped services. It simplifies the process by offering properties and methods to access services specific to the component’s lifecycle, ensuring that each component has its own isolated context.
```razor
// Example of using OwningComponentBase in a Blazor component
@page "/forms/edit/{id:long}" @layout EmptyLayout @inherits OwningComponentBase<DataContext>
```
x??

---

#### Entity Framework Core and Blazor Data Context Issues
Background context: This concept explores how using separate data contexts in different components can lead to issues like unsaved changes or redundant database queries. In the provided scenario, the `Editor` component uses a different data context than the `List` component, leading to lost edits when navigating away from the form.

:p How does the use of separate data contexts affect form editing and navigation in Blazor?
??x
When using separate data contexts for different components (like `Editor` and `List`), any changes made in one context are not automatically saved or propagated to another. In this case, edits made via the `Editor` component’s distinct data context get discarded when navigating back to a list view managed by its own data context.

This issue can be mitigated by ensuring both components share the same data context or implementing explicit state management and synchronization mechanisms.
x??

---
#### Blazor Increment Button Counter
Background context: This example demonstrates how Blazor's rendering mechanism can lead to an increase in database queries when a component re-renders due to state changes. Each time the counter is incremented, it triggers a new query, even though the data might not have changed.

:p What issue does adding a button that increments a counter reveal about Blazor's handling of state and rendering?
??x
Blazor responds efficiently to state changes but must render components to detect necessary updates for sending them to the browser. In this scenario, each time the `Increment` button is clicked, it triggers an update in the component’s state (incrementing the counter), which forces a re-render and thus a new database query.

To mitigate this issue, you might want to implement caching mechanisms or ensure that unnecessary queries are avoided by checking for actual state changes before performing expensive operations.
x??

---
#### Entity Framework Core Query Logging
Background context: The provided logging messages show repeated SQL queries being executed each time the counter is incremented. These queries retrieve data from the database without any apparent change in the underlying dataset, indicating an increase in query volume due to component re-renders.

:p What does the repeated query issue demonstrate about Blazor's rendering process and how it interacts with Entity Framework Core?
??x
The repeated query issue highlights that Blazor’s state-driven approach can cause frequent database queries even when no actual data changes occur. Each time a component updates its state (like incrementing a counter), it triggers a re-render, leading to new SQL queries being executed.

To address this, you might want to use caching strategies or implement custom logic to avoid unnecessary database calls.
x??

---

#### Entity Framework Core and Blazor Interaction Issues
Background context explaining the concept. Entity Framework Core (EF Core) and Blazor work together to manage data queries and component rendering. EF Core uses LINQ expressions that send new database requests each time a property is read, while Blazor re-renders components to determine changes.
:p What issue arises when using EF Core with Blazor?
??x
When the People property is read twice by the List component—once for determining if data has loaded and once for generating table rows. Additionally, clicking the Increment button causes the component to be rendered again, leading to more database queries.
x??

---
#### Unnecessary Database Queries
Background context explaining the concept. Each time a property that queries from the database is read, EF Core sends a new request to the database. This can lead to unnecessary overhead and unexpected data updates for users if they make unrelated interactions.
:p Why do unnecessary queries occur in this scenario?
??x
Unnecessary queries occur because Blazor re-renders the component each time an interaction happens (like clicking Increment), which forces reading of properties that trigger additional database requests.
x??

---
#### Managing Queries to Prevent Unnecessary Requests
Background context explaining the concept. To mitigate unnecessary database queries, one should query the database only when necessary and provide options for users to manually refresh data if needed.
:p How can developers manage queries to prevent unnecessary database requests?
??x
Developers can limit database queries by querying once initially and re-querying only under specific conditions where updates are expected. They can also offer an explicit option for users to reload the data, as shown in Listing 36.18.
x??

---
#### Example Code for Controlling Queries
Background context explaining the concept. The code example provided demonstrates how to manage queries by querying once and then re-querying only when necessary.
:p What does the UpdateData method do?
??x
The `UpdateData` method performs a database query only if it hasn't been done before, using `ToListAsync<Person>` to force evaluation of the Entity Framework Core query. It avoids redundant queries by checking if the context is null and then loading the data into the People property.
x??

---
#### Re-rendering and Increment Button
Background context explaining the concept. When a button like Increment is clicked, Blazor re-renders components, leading to repeated database requests for properties that are read during rendering.
:p Why does clicking the Increment button cause multiple database queries?
??x
Clicking the Increment button causes the List component to be rendered again, which reads the People property twice—once to determine if data has loaded and once to generate table rows. This triggers additional database queries.
x??

---
#### Blazor's Rerender Mechanism
Background context explaining the concept. Blazor must rerender components after interactions like clicking buttons to determine changes that need to be reflected in the UI. This can lead to redundant database requests if properties are read multiple times during rendering.
:p How does Blazor handle component re-rendering?
??x
Blazor rerenders components after interactions, evaluating all Razor expressions and determining necessary HTML changes. This process reads properties like People twice when the Increment button is clicked, causing repeated database queries.
x??

---

---
#### Data Binding and Querying Mechanism
Explanation: The provided code demonstrates how data binding works with a list of `Person` objects in a Blazor component. The initial query is made only when the component initializes, but additional queries are triggered by user interactions like clicking buttons.

:p How does the initial data fetching work in this Blazor component?
??x
The initial data fetching occurs during the initialization of the component via the `OnInitializedAsync` method. This ensures that the UI can display data without requiring an immediate interaction from the user.
```csharp
protected async override Task OnInitializedAsync()
{
    await UpdateData();
}
```
x??
---
#### Button to Update Data
Explanation: The `UpdateButton` in the Blazor component allows users to manually trigger a new database query by calling the `UpdateData()` method. This is useful for demonstrating or testing data refreshes.

:p What does clicking the "Update" button do in this context?
??x
Clicking the "Update" button invokes the `UpdateData()` method, which fetches fresh data from the database and updates the component's state.
```csharp
private async Task UpdateData()
{
    People = await Query.ToListAsync<Person>();
}
```
x??
---
#### Sorting with a New Query
Explanation: The `SortWithQuery` button demonstrates how to sort data by performing a new query. This is done through an `OrderBy` LINQ method on the `IQueryable` object.

:p How does the "Sort (With Query)" button work in this context?
??x
The "Sort (With Query)" button sorts the list by surname using a new database query. It uses the `OrderBy` LINQ method to sort and then fetches the updated data.
```csharp
public async Task SortWithQuery()
{
    await UpdateData(Query.OrderBy(p => p.Surname));
}
```
x??
---
#### Sorting Without Querying
Explanation: The `SortWithoutQuery` method shows how to perform in-memory sorting without issuing a new database query. This is done by directly manipulating the existing list.

:p How does the "Sort (No Query)" button work differently from the other sort methods?
??x
The "Sort (No Query)" button sorts the list in memory using LINQ's `OrderBy` method and converts it to a list without performing an additional query. This is efficient when sorting large datasets that are already loaded.
```csharp
public void SortWithoutQuery()
{
    People = People.OrderBy(p => p.Firstname).ToList<Person>();
}
```
x??
---

