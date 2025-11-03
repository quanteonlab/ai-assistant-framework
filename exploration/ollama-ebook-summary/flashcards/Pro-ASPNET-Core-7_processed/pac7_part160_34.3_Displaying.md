# Flashcards: Pro-ASPNET-Core-7_processed (Part 160)

**Starting Chapter:** 34.3 Displaying child content in a component

---

#### Custom Binding in Razor Components
Background context: Custom binding allows for dynamic updates to UI elements based on component state changes. This is particularly useful when you want to highlight or change the appearance of content based on user interaction.

:p How does custom binding work in a Razor Component?
??x
Custom binding works by updating the UI element's class or style dynamically based on the component’s state. In this case, `SelectedCity` is used to determine which city should be highlighted.

Example code:
```razor
<div class="@GetClass(CurrentCity)">
    // Content that changes based on SelectedCity
</div>

@code {
    public string GetClass(string? city) => 
        CurrentCity == city ? "bg-info text-white" : "";
    
    public string CurrentCity { get; set; } = "New York";
}
```
x??

---

#### Using a Custom Binding Example with Cities
Background context: The provided example demonstrates how to highlight a selected city in a list. It uses the `GetClass` method to change the class of the highlighted element based on user selection.

:p How does the `GetClass` method determine which city should be highlighted?
??x
The `GetClass` method checks if the current `SelectedCity` matches the input city. If they match, it returns `"bg-info text-white"`, indicating that the city should be highlighted with a background info color and white text.

Example code:
```razor
public string GetClass(string? city) => 
    SelectedCity == city ? "bg-info text-white" : "";
```
x??

---

#### Child Content in Razor Components
Background context: Child content allows components to include dynamic content from their parents. This is achieved by defining a `ChildContent` property with the type `RenderFragment`.

:p How does a component receive and display child content?
??x
A component receives and displays child content through the `ChildContent` parameter, which is of type `RenderFragment`. The `@ChildContent` expression includes this content in the component's HTML output.

Example code:
```razor
<ThemeWrapper 
    Theme="themeColor"
    Title="Title Text">
    @ChildContent
</ThemeWrapper>

@code {
    [Parameter] public string? Theme { get; set; }
    [Parameter] public string? Title { get; set; }
    [Parameter] public RenderFragment? ChildContent { get; set; }
}
```
x??

---

#### Restricting Element Reuse in Blazor
Background context: To prevent unexpected behavior when elements are reused, especially after a sequence change, you can use the `@key` attribute to associate elements with specific data items.

:p How does the `@key` attribute help in restricting element reuse?
??x
The `@key` attribute helps restrict element reuse by ensuring that Blazor only reuses elements for which there is still a corresponding data item. If an element's key matches the key of a new or existing data item, it will be reused; otherwise, a new element will be created.

Example code:
```razor
@foreach (Person p in People ?? Enumerable.Empty<Person>()) {
    <tr @key="p.PersonId" class="@GetClass(p?.Location?.City)">
        // table content
    </tr>
}
```
x??

---

#### Difference Between Custom Binding and Child Content
Background context: Both custom binding and child content are mechanisms for dynamic updates in Blazor. However, they serve different purposes: custom binding updates UI elements based on state changes, while child content allows the inclusion of parent-provided content.

:p How does custom binding differ from using child content in a component?
??x
Custom binding is used to update specific parts of the UI dynamically based on the component's state. For example, highlighting a selected city or changing text color based on conditions.

Child content, on the other hand, allows components to include dynamic elements provided by their parents. This can be useful for wrapping and styling external content.

Example code:
```razor
// Custom Binding Example
<div class="@GetClass(CurrentCity)">
    Selected City: @CurrentCity
</div>

@code {
    public string GetClass(string? city) => 
        CurrentCity == city ? "bg-info text-white" : "";

    public string CurrentCity { get; set; } = "New York";
}

// Child Content Example
<ThemeWrapper Theme="info" Title="Title">
    <p>Dynamic content goes here</p>
</ThemeWrapper>

@code {
    [Parameter] public string? Theme { get; set; }
    [Parameter] public string? Title { get; set; }
    [Parameter] public RenderFragment? ChildContent { get; set; }
}
```
x??

#### Displaying Child Content in a Component
Background context explaining that this concept involves using Razor components to display child content. The example shows how to use `RenderFragment` properties within custom components to manage and display data dynamically.

:p How does the `TableTemplate` component handle child content for a table?
??x
The `TableTemplate` component uses `RenderFragment` properties (`Header` and `Body`) to define regions of content that can be customized. The `Header` fragment defines the table header, while the `Body` fragment contains the rows of data.

```razor
<TableTemplate>
    <Header>
        <tr><th>ID</th><th>Name</th><th>Dept</th><th>Location</th></tr>
    </Header>
    <Body>
        @foreach (Person p in People ?? Enumerable.Empty<Person>()) {
            <tr class="@GetClass(p?.Location?.City)">
                <td>@p?.PersonId</td>
                <td>@p?.Surname, @p?.Firstname</td>
                <td>@p?.Department?.Name</td>
                <td>@p?.Location?.City, @p?.Location?.State</td>
            </tr>
        }
    </Body>
</TableTemplate>
```
x??

---

#### Using RenderFragment Properties in Components
Background context explaining the use of `RenderFragment` properties within components to define different sections of content. This allows for more structured and reusable component designs.

:p How does the `PeopleList` component utilize `RenderFragment` properties?
??x
The `PeopleList` component uses the `Header` and `Body` fragments defined in the `TableTemplate` component to structure the table content. The `Header` fragment defines the table headers, while the `Body` fragment iterates over the list of people and generates table rows.

```razor
<TableTemplate>
    <Header>
        <tr><th>ID</th><th>Name</th><th>Dept</th><th>Location</th></tr>
    </Header>
    <Body>
        @foreach (Person p in People ?? Enumerable.Empty<Person>()) {
            <tr class="@GetClass(p?.Location?.City)">
                <td>@p?.PersonId</td>
                <td>@p?.Surname, @p?.Firstname</td>
                <td>@p?.Department?.Name</td>
                <td>@p?.Location?.City, @p?.Location?.State</td>
            </tr>
        }
    </Body>
</TableTemplate>
```
x??

---

#### Applying Template Components in Blazor
Background context explaining the use of template components to define and apply reusable structures. This allows for cleaner and more maintainable code.

:p How does the `PeopleList` component use the `TableTemplate` component?
??x
The `PeopleList` component uses the `TableTemplate` component by providing its own implementation of the `Header` and `Body` fragments. The `Header` fragment defines the table headers, while the `Body` fragment contains the logic to display each person's information as a row in the table.

```razor
<TableTemplate>
    <Header>
        <tr><th>ID</th><th>Name</th><th>Dept</th><th>Location</th></tr>
    </Header>
    <Body>
        @foreach (Person p in People ?? Enumerable.Empty<Person>()) {
            <tr class="@GetClass(p?.Location?.City)">
                <td>@p?.PersonId</td>
                <td>@p?.Surname, @p?.Firstname</td>
                <td>@p?.Department?.Name</td>
                <td>@p?.Location?.City, @p?.Location?.State</td>
            </tr>
        }
    </Body>
</TableTemplate>
```
x??

---

#### Restarting and Requesting ASP.NET Core
Background context explaining the process of restarting an ASP.NET Core application to see changes in the browser. This is often necessary after making modifications to the code.

:p What does restarting the ASP.NET Core application accomplish?
??x
Restarting the ASP.NET Core application ensures that any recent code changes are reflected in the running server. Without a restart, some changes might not be picked up by the application, leading to outdated or incorrect behavior when accessed via HTTP requests.

To see changes made, you need to restart the development server and then request the appropriate URL (e.g., `http://localhost:5000/controllers`).

```bash
dotnet run
```
x??

---

#### Using a Theme Wrapper Component
Background context explaining how theme wrapper components can be used to apply styling themes dynamically. This example shows how to use a `ThemeWrapper` component to change the theme and title based on user interactions.

:p How does the `PeopleList` component interact with the `ThemeWrapper` component?
??x
The `PeopleList` component interacts with the `ThemeWrapper` component by providing it with dynamic properties such as `Theme` and `Title`. The `ThemeWrapper` component then uses these values to apply the appropriate theme and title to its child content.

```razor
<ThemeWrapper Theme="info" Title="Location Selector">
    <SelectFilter values="@Cities" title="@SelectTitle"
                  @bind-SelectedValue="SelectedCity" />
    <button class="btn btn-primary mt-2"
            @onclick="() => SelectedCity = "Oakland")">Change</button>
</ThemeWrapper>
```
x??

---

#### Adding a Generic Type Parameter to Template Components
Background context: In this section, we're enhancing a template component for rendering tables by making it more flexible and data-aware. The previous implementation relied on the parent component to manage row generation, which limits its utility.

Generic type parameters allow components to be more reusable and adaptable to different types of data. By adding a generic type parameter, the component can handle any type of data objects passed from the parent component.

:p What is the purpose of using a generic type parameter in template components?
??x
The purpose of using a generic type parameter in template components is to make them more flexible and reusable across different types of data. This allows the component to be used for rendering tables with various kinds of data without needing to rewrite the component code each time.

By adding a generic type parameter, you can define how the component should handle the data it receives from its parent component. For instance, the template component can generate rows based on the provided data objects and render them using the specified templates.
??x
---

#### Using `@typeparam` to Define Generic Type Parameters
Background context: The `@typeparam` attribute is used in Razor components to define generic type parameters. This allows you to create more flexible components that can work with different types of data.

:p How do you define a generic type parameter using the `@typeparam` attribute?
??x
You define a generic type parameter using the `@typeparam` attribute within your component's `.razor` file. The syntax is as follows:

```razor
@typeparam RowType
```

Here, `RowType` is the name of the generic type parameter.

:p How are properties added to handle data and templates in the template component?
??x
Properties are added to handle the data and templates that will be used by the component. Specifically, you add a property for handling the sequence of data objects (`RowData`) and another for rendering each object using a template (`RowTemplate`).

Here is an example of how these properties are defined:

```razor
[Parameter]
public RenderFragment<RowType>? RowTemplate { get; set; }

[Parameter]
public IEnumerable<RowType>? RowData { get; set; }
```

- `RowTemplate`: This property takes a `RenderFragment<RowType>` to define how each object should be rendered.
- `RowData`: This property is an `IEnumerable<RowType>`, representing the collection of data objects that will be used by the component.

These properties enable the template component to receive and process data from its parent component.
??x
---

#### Example Code for Template Component with Generic Type Parameters
Background context: The example code below demonstrates how to add a generic type parameter, define properties for handling data and templates, and use these in the component's markup.

:p Show the updated code for the template component using generic type parameters.
??x
Here is an example of the updated `Blazor/TableTemplate.razor` file:

```razor
@typeparam RowType

<table class="table table-sm table-bordered table-striped">
    @if (Header != null) {
        <thead>@Header</thead>
    }
    <tbody>
        @if (RowData != null && RowTemplate != null) {
            @foreach (RowType item in RowData) {
                <tr>@RowTemplate(item)</tr>
            }
        }
    </tbody>
</table>

@code {
    [Parameter]
    public RenderFragment? Header { get; set; }

    [Parameter]
    public RenderFragment<RowType>? RowTemplate { get; set; }

    [Parameter]
    public IEnumerable<RowType>? RowData { get; set; }
}
```

Explanation: 
- `@typeparam RowType` defines the generic type parameter.
- The component checks if `Header`, `RowData`, and `RowTemplate` are not null before rendering them.
- When a collection of objects (`RowData`) is passed to the component, it iterates over each item and uses the `RowTemplate` to render it within a table row.

:p How does the component use `RowTemplate` to generate rows?
??x
The component uses `RowTemplate` by invoking it as a method for each object in `RowData`. Here is how it works:

```razor
@foreach (RowType item in RowData) {
    <tr>@RowTemplate(item)</tr>
}
```

In this snippet, the `RowTemplate` property is invoked with `item` as an argument. The result of invoking `RowTemplate(item)` is then rendered inside a table row (`<tr>`).

:p Explain how to constrain generic type parameters.
??x
Blazor generic type parameters can be constrained using the C# `where` keyword, ensuring only types that meet specified characteristics are used. For example:

```razor
@typeparam T where T : class
```

This constraint ensures that only reference types (classes) can be used as the generic parameter.

:p What is the significance of using `RenderFragment<RowType>`?
??x
Using `RenderFragment<RowType>` in the component allows it to receive a content section for each item in the data collection. This enables dynamic and flexible rendering based on the specific type of data being processed. The component can then use this fragment to generate the appropriate UI elements.

:p How does the component handle the header if provided?
??x
If a header is provided, the component checks if `Header` is not null and then renders it within the table's `<thead>` element:

```razor
@if (Header != null) {
    <thead>@Header</thead>
}
```

This ensures that the header can be customized by the parent component, providing flexibility in how the table structure is defined.
??x
---

#### Understanding the Template Component in Blazor
Background context explaining how template components are used in Blazor to render dynamic content based on data. The `TableTemplate` component is a generic template that processes rows of data and displays them as HTML elements, such as table rows (`<tr>`).

:p What does the `RowType` attribute do in the `TableTemplate` component?
??x
The `RowType` attribute specifies the type of objects (e.g., `Person`) that will be processed by the template. This is necessary for Blazor to correctly cast and bind properties from the data source.

```razor
@TableTemplate RowType="Person" RowData="People">
```
x??

---

#### Configuring Data in Blazor Components
Explaining how data sources are configured within a component, specifically with the `People` property which fetches a list of `Person` objects along with their related properties (`Department` and `Location`) using LINQ.

:p How is the `People` collection defined in the `PeopleList` component?
??x
The `People` collection is defined as an observable collection that includes navigation properties for `Department` and `Location`, which helps in fetching related data when querying the database. This is achieved by including the `Department` and `Location` properties in the query.

```csharp
public IEnumerable<Person>? People => Context?.People             .Include(p => p.Department)             .Include(p => p.Location);
```
x??

---

#### Rendering Rows with Template Components
Explaining how rows are rendered using a template, where each row corresponds to an object of type `Person`, and the template fills in the necessary HTML elements based on properties of that object.

:p How does the `RowTemplate` section work within the `TableTemplate` component?
??x
The `RowTemplate` section defines what content should be rendered for each item (a `Person`) from the data source. It uses a named parameter (`Context="p"`) to bind each `Person` object, allowing access to its properties.

```razor
<RowTemplate Context="p">
    <td>@p.PersonId</td>
    <td>@p.Surname, @p.Firstname</td>
    <td>@p.Department?.Name</td>
    <td>@p.Location?.City, @p.Location?.State</td>
</RowTemplate>
```
x??

---

#### Using Attributes to Name Current Objects
Explanation of the `Context` attribute and how it assigns a name (`p`) to each current object being processed in the template. This allows for easy access to properties within the `RowTemplate`.

:p What is the purpose of the `Context` attribute?
??x
The `Context` attribute is used to bind the current object (a `Person`, in this case) and assign it a name (`p`). This makes it easier to reference properties from that object directly within the template.

```razor
<RowTemplate Context="p">
    <td>@p.PersonId</td>
```
x??

---

#### Configuring the Table with Headers
Explanation of how headers are set up in the `TableTemplate` component, ensuring the table has a clear structure before rendering rows.

:p How does the `<Header>` section work within the `TableTemplate` component?
??x
The `<Header>` section defines the column headers for the table. It uses static HTML to create a header row (`<tr>`) with cells representing each column (e.g., ID, Name, Department, Location).

```razor
<Header>
    <tr><th>ID</th><th>Name</th><th>Dept</th><th>Location</th></tr>
</Header>
```
x??

---

#### Restarting ASP.NET Core and Viewing the Output
Explanation of how to start an ASP.NET Core application and view the rendered output.

:p How do you start the ASP.NET Core application and see the table?
??x
To start the ASP.NET Core application, use the command `dotnet run` in the project directory. You can then navigate to `http://localhost:5000/controllers` in a web browser to see the rendered table with rows corresponding to each person.

```bash
dotnet run
```
Visit:
```
http://localhost:5000/controllers
```
x??

---

#### TableTemplate Component Overview
The `TableTemplate` component is a reusable template for displaying data in a table format. It accepts various parameters to customize its behavior, such as header rendering, row templating, and sorting options.

Background context: The component is designed to be flexible by allowing the parent component to define how rows are displayed, sorted, and highlighted. This modular approach makes it easier to maintain and reuse code across different parts of an application.
:p What does the `TableTemplate` component allow a parent component to do?
??x
The parent component can customize how table rows are rendered using a `RowTemplate`. It can also define sorting criteria and highlight rules by providing specific functions. This flexibility is achieved through parameters like `RowTemplate`, `SortDirection`, and `Highlight`.

Example of usage:
```razor
<TableTemplate RowType="Person" 
               RowData="@People" 
               Highlight="@(p => p.Location?.City)" 
               SortDirection="@(p => p.Surname)">
    <Header>
        <tr><th>ID</th><th>Name</th><th>Dept</th><th>Location</th></tr>
    </Header>
    <RowTemplate Context="p">
        <td>@p.PersonId</td>
        <td>@p.Surname, @p.Firstname</td>
        <td>@p.Department?.Name</td>
        <td>@p.Location?.City, @p.Location?.State</td>
    </RowTemplate>
</TableTemplate>
```
x??

---
#### Header Parameter in TableTemplate
The `Header` parameter is used to define the table header. It allows for custom HTML rendering of the header section.

Background context: By using a render fragment for the `Header`, developers can create complex and dynamic headers that are not limited by simple string values.
:p How does the `Header` parameter in the `TableTemplate` component work?
??x
The `Header` parameter is defined as a `RenderFragment`. This means it accepts a block of Razor code to generate the table header dynamically. The developer can use this to create complex HTML structures like `<th>` elements, nested tags, or even conditions.

Example:
```razor
<TableTemplate RowType="Person" 
               RowData="@People" 
               Header="@( () => <tr><th>ID</th><th>Name</th><th>Dept</th><th>Location</th></tr>)">
```
x??

---
#### RowTemplate Parameter in TableTemplate
The `RowTemplate` parameter is used to define how each row of data should be rendered. It takes a `RenderFragment<RowType>` and a context parameter.

Background context: The flexibility of using a render fragment for the `RowTemplate` allows for complex and dynamic rendering logic, which can change based on the type of data being displayed.
:p What is the purpose of the `RowTemplate` parameter in the `TableTemplate` component?
??x
The `RowTemplate` parameter accepts a `RenderFragment<RowType>` that defines how each row should be rendered. It takes an instance of the row's type as its context, allowing for dynamic content generation.

Example:
```razor
<TableTemplate RowType="Person" 
               RowData="@People" 
               RowTemplate="@(p => <tr><td>@p.PersonId</td>
                                          <td>@p.Surname, @p.Firstname</td>
                                          <td>@p.Department?.Name</td>
                                          <td>@p.Location?.City, @p.Location?.State</td></tr>)">
```
x??

---
#### Highlight Parameter in TableTemplate
The `Highlight` parameter is used to define a function that determines which rows should be highlighted based on certain properties.

Background context: This parameter allows the parent component to provide logic for highlighting specific rows, enhancing user interaction and data visualization.
:p How does the `Highlight` parameter work in the `TableTemplate` component?
??x
The `Highlight` parameter is defined as a function that takes an instance of `RowType` and returns a string. This function is used to determine which rows should be highlighted based on certain properties.

Example:
```razor
<TableTemplate RowType="Person" 
               RowData="@People" 
               Highlight="@(p => p.Location?.City)">
```
In this example, rows are highlighted based on the city in their location.
x??

---
#### SortDirection Parameter in TableTemplate
The `SortDirection` parameter is used to define a function that determines how rows should be sorted.

Background context: This parameter allows the parent component to provide logic for sorting data in either ascending or descending order.
:p How does the `SortDirection` parameter work in the `TableTemplate` component?
??x
The `SortDirection` parameter is defined as a function that takes an instance of `RowType` and returns a string. This function is used to determine the primary key for sorting rows.

Example:
```razor
<TableTemplate RowType="Person" 
               RowData="@People" 
               SortDirection="@(p => p.Surname)">
```
In this example, rows are sorted by surname in ascending order.
x??

---
#### HighlightChoices Method in TableTemplate
The `HighlightChoices` method generates a list of values that can be used to highlight table rows.

Background context: This method leverages the flexibility provided by the `RowData` and `Highlight` parameters to dynamically generate a set of options for row highlighting based on the data.
:p What does the `HighlightChoices` method in the `TableTemplate` component do?
??x
The `HighlightChoices` method generates a distinct list of values that can be used as options for row highlighting. It uses the `RowData` and the `Highlight` function to extract unique highlight values.

Example:
```razor
public IEnumerable<string> HighlightChoices() => 
    RowData.Select(item => Highlight(item)).Distinct();
```
This method ensures that only distinct values are provided, preventing duplicate entries in the dropdown.
x??

---
#### SortedData Method in TableTemplate
The `SortedData` method sorts the rows based on the selected sort direction.

Background context: This method dynamically sorts the data based on user input and provides an ordered collection of rows to be displayed in the table.
:p What does the `SortedData` method in the `TableTemplate` component do?
??x
The `SortedData` method sorts the `RowData` based on the selected sort direction. It uses LINQ's `OrderBy` or `OrderByDescending` methods depending on the user’s selection.

Example:
```razor
public IEnumerable<RowType> SortedData() => 
    SortDirectionSelection == "Ascending" ? 
        RowData.OrderBy(SortDirection) : 
        RowData.OrderByDescending(SortDirection);
```
In this example, rows are sorted in ascending order by default if no specific sort direction is selected.
x??

#### Concept: TableTemplate Component for Displaying Data
Background context explaining the concept. The `TableTemplate` component is a reusable template used to display, sort, and highlight data. This template can be adapted to different types of data by changing its generic type parameters.

:p What does the `TableTemplate` component do?
??x
The `TableTemplate` component provides a way to display tables with sortable and highlightable rows for various data types. It uses several attributes like `RowType`, `RowData`, `Highlight`, and `SortDirection` to customize its behavior based on the provided data.

```razor
@TableTemplate RowType="Department" RowData="Departments"
    Highlight="@(d => d.Name)"
    SortDirection="@(d => d.Name)">
```

x??

---

#### Concept: DepartmentList Component Implementation
Background context explaining the concept. The `DepartmentList` component is a specific implementation of the generic `TableTemplate`. It uses the `TableTemplate` to display details about departments, including their people and locations.

:p How does the `DepartmentList` component differ from the `TableTemplate`?
??x
The `DepartmentList` component is an instance of the reusable `TableTemplate` designed specifically for displaying department-related data. It includes a header definition with column names and a row template to format each department's details, such as ID, Name, People (surnames), and Locations (cities).

```razor
<TableTemplate RowType="Department" 
               RowData="Departments"
               Highlight="@(d => d.Name)" 
               SortDirection="@(d => d.Name)">
    <Header>
        <tr><th>ID</th><th>Name</th><th>People</th><th>Locations</th></tr>
    </Header>
    <RowTemplate Context="d">
        <td>@d.Departmentid</td>
        <td>@d.Name</td>
        <td>@(String.Join(", ", d.People.Select(p => p.Surname)))</td>
        <td>@(String.Join(", ", d.People.Select(p => p.Location.City).Distinct()))</td>
    </RowTemplate>
</TableTemplate>
```

x??

---

#### Concept: Data Context for Departments and People
Background context explaining the concept. The `DataContext` provides access to data such as departments and people, which can be filtered and included based on specific navigation properties.

:p How is the `Departments` data accessed in the `DepartmentList` component?
??x
The `Departments` data is accessed through the `DataContext` property in the `DepartmentList` component. The `Departments` property uses LINQ to include related entities like `People` and `Location`.

```razor
public IEnumerable<Department>? Departments => Context?.Departments
    .Include(d => d.People)
    .ThenInclude(p => p.Location);
```

x??

---

#### Concept: RowData and RowType Attributes in TableTemplate
Background context explaining the concept. The `RowData` and `RowType` attributes are crucial for specifying the data source and type used by the `TableTemplate`.

:p What do the `RowData` and `RowType` attributes do?
??x
The `RowData` attribute specifies the collection of objects that will be displayed in the table, while the `RowType` attribute defines the type of each individual row. These attributes are necessary for the `TableTemplate` to correctly render and manipulate the data.

```razor
@TableTemplate RowType="Department" 
               RowData="Departments"
```

x??

---

#### Concept: Highlight Attribute in TableTemplate
Background context explaining the concept. The `Highlight` attribute of the `TableTemplate` component is used to select a property that determines which rows should be highlighted, usually based on their value.

:p How does the `Highlight` attribute work in the `TableTemplate`?
??x
The `Highlight` attribute in the `TableTemplate` sets a function that returns a property value for each row. This value is then used to determine whether a row should be highlighted or not. In this case, it highlights rows based on the department's name.

```razor
Highlight="@(d => d.Name)"
```

x??

---

#### Concept: SortDirection Attribute in TableTemplate
Background context explaining the concept. The `SortDirection` attribute of the `TableTemplate` component is used to select a property that determines the default sorting direction for the table rows.

:p How does the `SortDirection` attribute work?
??x
The `SortDirection` attribute sets a function that returns a property value for each row. This value is then used as the basis for sorting the rows in the table. In this case, it sorts departments based on their name by default.

```razor
SortDirection="@(d => d.Name)"
```

x??

---

