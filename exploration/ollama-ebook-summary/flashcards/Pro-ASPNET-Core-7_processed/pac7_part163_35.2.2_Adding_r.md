# Flashcards: Pro-ASPNET-Core-7_processed (Part 163)

**Starting Chapter:** 35.2.2 Adding routes to components

---

#### Using Component Routing
Background context: In Blazor, component routing allows you to map URLs directly to components. This is useful for creating single-page applications where different components are rendered based on the URL requested by the user.

:p What does the `@page` directive do in a Blazor component?
??x
The `@page` directive is used to specify which URLs should display a particular component. For example, adding `@page "/people"` to a component means that when a request is made to `/people`, this component will be rendered.

```razor
@page "/people"
```
x??

---
#### Adding Routes to Components
Background context: To use the `@page` directive in Blazor components, you need to include it within the `.razor` file. This allows you to map specific URLs to components and control how content is displayed based on the URL.

:p How does the `@page` directive enable routing in a component?
??x
The `@page` directive enables routing by associating a specific URL with a Blazor component. When a request comes in for that URL, the specified component is rendered and its content is displayed.

```razor
@page "/people"
```
This line tells Blazor to use the `PeopleList` component when someone navigates to `/people`.

x??

---
#### Example of `@page` Directive Usage (PeopleList Component)
Background context: The `PeopleList` component is an example where the `@page` directive is used to map a URL (`/people`) to a specific component.

:p What does adding `@page "/people"` in the `PeopleList.razor` file do?
??x
Adding `@page "/people"` in the `PeopleList.razor` file tells Blazor that when a user navigates to the `/people` URL, this component will be rendered and its content displayed.

```razor
@page "/people"
```
This directive ensures that requests to `/people` are handled by the `PeopleList` component.

x??

---
#### Multiple Route Support (DepartmentList Component)
Background context: The `DepartmentList` component is an example where multiple routes can be supported using multiple `@page` directives. This allows for flexibility in URL mapping, enabling different URLs to render the same or similar content.

:p How do you add support for more than one route in a Blazor component?
??x
To support multiple routes in a Blazor component, you can use multiple `@page` directives within the `.razor` file. For example:

```razor
@page "/departments"
@page "/depts"
```
These directives map both `/departments` and `/depts` URLs to the `DepartmentList` component.

x??

---
#### Setting a Default Component Route (PeopleList Component)
Background context: When using component routing, it's often necessary to define a default route for when no other routes match. This is typically done by including a route that matches the root URL (`/`) in one of your components.

:p How does setting `@page "/"` in the `PeopleList` component affect the application?
??x
Setting `@page "/"` in the `PeopleList` component ensures that if no other routes match, this component will be rendered when the user navigates to the root URL (`/`). This provides a default route for unmatched requests.

```razor
@page "/people"
@page "/"
```
This combination of directives means that both `/people` and the root URL (`/`) will render the `PeopleList` component, providing a fallback in case no other routes are matched.

x??

---
#### MapFallbackToPage Method (Component Routing)
Background context: The `MapFallbackToPage` method is used to configure the routing system in Blazor. It specifies that if there are no matching routes for a request, the `_Host` page should be rendered as a fallback.

:p What does `MapFallbackToPage("_Host")` do?
??x
The `MapFallbackToPage("_Host")` method configures the routing system to use the `_Host` page as a last resort when no other routes match. This means that if Blazor can't find a route for the requested URL, it will render the `_Host` page.

```csharp
app.MapFallbackToPage("_Host");
```
This ensures that all unmatched requests are handled by rendering the `_Host` page, providing a default fallback mechanism.

x??

---

#### Introduction to NavLink Component
Background context: The `NavLink` component is part of the routing system in Blazor and allows for navigation between components via URLs. It provides a way to create interactive links within your Blazor application that are aware of the current route, making them useful for both aesthetic and functional purposes.

:p What is the purpose of the `NavLink` component in Blazor?
??x
The `NavLink` component enables navigation between routed components by rendering anchor elements that are integrated into the routing system. This means that when a user clicks on a `NavLink`, they will be directed to a different URL without requiring an additional HTTP request, as the application can handle the transition using JavaScript.

```razor
<NavLink class="btn btn-primary" href="/depts">Departments</NavLink>
```

This code snippet demonstrates how to use the `NavLink` component. The `class` attribute styles the link, and the `href` attribute specifies the URL that the navigation will target.
x??

---

#### Programmatic Navigation Using NavigationManager
Background context: The `NavigationManager` class provides a way to navigate programmatically between components within your Blazor application. This is particularly useful when you want to trigger navigation based on user interactions or other events.

:p How can you perform programmatic navigation in Blazor using the `NavigationManager`?
??x
You can use the `NavigateTo` method of the `NavigationManager` class to navigate programmatically between components. The method takes a URL as its parameter and changes the location without requiring an additional HTTP request. This is done through the JavaScript API provided by the application.

```razor
@code {
    [Inject]
    public NavigationManager? NavManager { get; set; }

    public void HandleClick() => NavManager?.NavigateTo("/people");
}
```

This code snippet shows how to inject `NavigationManager` and use its `NavigateTo` method in a component's event handler. When the button is clicked, it will navigate to the `/people` URL, which corresponds to the `PeopleList` component.
x??

---

#### Understanding Routing in Blazor
Background context: Routing in Blazor allows you to define different pages or components that can be accessed through specific URLs. This helps in creating a more organized and modular application structure.

:p What is routing in Blazor?
??x
Routing in Blazor refers to the process of defining URL paths that correspond to specific components within your application. Each route is associated with a component, allowing users to navigate directly to that component using the specified URL. This enables a cleaner URL structure and better user experience by making navigation more intuitive.

For example:
- `@page "/people"` associates the `/people` URL with the `PeopleList` component.
- `@page "/departments"` or `@page "/depts"` associates the respective URLs with the `DepartmentList` component.

This setup ensures that when a user visits these URLs, the corresponding components are rendered without requiring an HTTP request to load new content.
x??

---

#### Using TableTemplate and RowTemplate
Background context: The `TableTemplate` and `RowTemplate` components in Blazor allow you to render data in a tabular format. They provide flexibility in defining how each row of data should be displayed.

:p How do `TableTemplate` and `RowTemplate` work together in rendering data?
??x
The `TableTemplate` component provides a template for rendering rows of data, while the `RowTemplate` defines the specific content to display within each row. Together, they allow you to define how your data should be presented in a table format.

Here's an example of using these components:

```razor
<TableTemplate RowType="Person" RowData="People"
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

In this example:
- `TableTemplate` specifies the type of data (`Person`) and where to find that data (`People`).
- `RowTemplate` defines how each row should be rendered based on the provided `Context` (in this case, a `Person` object).

This setup ensures that your data is presented in a structured table format, with appropriate headers and content.
x??

---

---
#### Using component routing TIP: NavigationManager.NavigateTo Method
The `NavigationManager.NavigateTo` method can accept an optional argument that, when set to true, forces the browser to send a new HTTP request and reload the HTML document. This is useful for scenarios where you want to force a full page refresh.
:p How does the `NavigationManager.NavigateTo` method behave with the optional argument?
??x
When the optional argument in `NavigationManager.NavigateTo` is set to true, it forces the browser to send a new HTTP request and reload the HTML document. This can be useful for ensuring that certain parts of your application are refreshed or reloaded.
```csharp
// Example usage:
NavigationManager.NavigateTo("/new-page", forceLoad: true);
```
x?
---

---
#### Receiving Routing Data with Component Properties
Components in Blazor can receive segment variables from the routing system by decorating a property with the `@Parameter` attribute. This allows components to access data passed via URL segments.
:p How do Razor Components receive segment variables in their properties?
??x
Razor Components can receive segment variables by defining a property that matches the name of the segment variable and decorating it with the `@Parameter` attribute. The type of the segment variable must match the type specified in the `@page` directive or be set to string if no specific type is provided.
```razor
// Example usage:
@page "/person/{id:long}"
<h5>Editor for Person: @Id</h5>
@code {
    [Parameter]
    public long Id { get; set; }
}
```
x?
---

---
#### Adding Navigation Links in Components
In the `PeopleList` component, navigation links are added using the `NavLink` component. These links navigate to a specific URL when clicked.
:p How do you add navigation links for each Person object in the PeopleList component?
??x
To add navigation links in the `PeopleList` component, use the `NavLink` component within the row template of your data table. The `href` attribute is generated by calling a method that returns the appropriate URL based on the person's ID.
```razor
// Example usage:
<td>
    <NavLink class="btn btn-sm btn-info" 
             href="@GetEditUrl(p.PersonId)">
        Edit
    </NavLink>
</td>
@code {
    public string GetEditUrl(long id) => $"/person/{id}";
}
```
x?
---

---
#### Generating Navigation URLs in Components
The `PeopleList` component uses a method to generate the navigation URL for each person, which is then used by the `NavLink` component.
:p How does the `PeopleList` component generate the navigation URL for each person?
??x
In the `PeopleList` component, the `GetEditUrl` method generates the navigation URL for each person. This method takes a long ID as an input and returns a string representing the URL to navigate to.
```razor
// Example usage:
@code {
    public string GetEditUrl(long id) => $"/person/{id}";
}
```
x?
---

