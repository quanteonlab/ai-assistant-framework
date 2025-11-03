# Flashcards: Pro-ASPNET-Core-7_processed (Part 122)

**Starting Chapter:** 24.1.1 Dropping the database. 24.2 Understanding view components

---

#### Using View Components in ASP.NET Core
Background context: In this section, we are working on an ASP.NET Core application that involves using view components to display data. The `Cities.cshtml` file is part of the Razor Pages structure and contains logic for displaying a table of cities.

:p What does the `Cities.cshtml` file do in the provided example?
??x
The `Cities.cshtml` file defines a Razor Page that retrieves city data from a `CitiesData` class and displays it in an HTML table. The code uses Razor syntax to iterate over the list of cities and display their names, countries, and populations.

```cshtml
@page
@inject CitiesData Data

<div class="m-2">
    <table class="table table-sm table-striped table-bordered">
        <tbody>
            @foreach (City c in Data.Cities)
            {
                <tr>
                    <td>@c.Name</td>
                    <td>@c.Country</td>
                    <td>@c.Population</td>
                </tr>
            }
        </tbody>
    </table>
</div>
```
x??

---

#### Dropping the Database in ASP.NET Core
Background context: To set up or reset a database for an ASP.NET Core application, you can use Entity Framework Core commands. The command `dotnet ef database drop --force` is used to remove the existing database.

:p How do you drop the database in an ASP.NET Core application using Entity Framework Core?
??x
To drop the database in an ASP.NET Core application, you run the following command in a PowerShell prompt:

```sh
dotnet ef database drop --force
```
This command drops the database associated with your project. The `--force` flag is optional but can be used to ensure that the operation proceeds without user confirmation.

x??

---

#### Running the Example Application
Background context: After setting up or resetting the database, you need to run the application to see it in action. This involves compiling and running the ASP.NET Core application using a command line tool.

:p How do you run an ASP.NET Core example application from the command line?
??x
To run an ASP.NET Core example application from the command line, use the following PowerShell command:

```sh
dotnet run
```
This command compiles and runs your ASP.NET Core application. If the database is being seeded as part of the startup process, it will be populated with initial data.

x??

---

#### Viewing the Application in a Web Browser
Background context: Once the application is running, you can access it through a web browser to view its output. The specific URL used depends on your project configuration and development server settings.

:p How do you access an ASP.NET Core application running locally?
??x
To access an ASP.NET Core application running locally, use the following URL in a web browser:

```
http://localhost:5000/cities
```

This URL directs your browser to the specified route within your application. The application will display the data as defined by the `Cities.cshtml` file.

x??

---

---
#### Introduction to View Components
View components are specialized action methods that provide partial views with data, independent of the main action method or Razor Page. They allow embedding reusable content into views without duplicating code and enable better separation of concerns within an application.

:p What is a view component?
??x
A view component is a C# class that provides a partial view with the required data independently from the action method or Razor Page. It can be thought of as a specialized action or page but is used only to provide a partial view with data; it cannot receive HTTP requests and always includes content in the parent view.
x??

---
#### Defining View Components
View components are defined by classes that end with `ViewComponent` and include an `Invoke` or `InvokeAsync` method. Alternatively, they can derive from the `ViewComponent` base class or be decorated with the `ViewComponent` attribute.

:p How do you define a view component?
??x
You define a view component as a C# class that ends with `ViewComponent`, includes an `Invoke` or `InvokeAsync` method, or derives from the `ViewComponent` base class. For example:
```csharp
using Microsoft.AspNetCore.Mvc;
using WebApp.Models;

namespace WebApp.Components
{
    public class CitySummary : ViewComponent
    {
        private CitiesData data;

        public CitySummary(CitiesData cdata)
        {
            data = cdata;
        }

        public string Invoke()
        {
            return $"{data.Cities.Count()} cities, {data.Cities.Sum(c => c.Population)} people";
        }
    }
}
```
x??

---
#### Applying View Components
View components can be applied in two ways: using the `Component` property or through an HTML element. The former uses the `InvokeAsync` method from the `IViewComponentHelper`, while the latter utilizes a custom HTML tag helper.

:p How do you apply a view component to a Razor Page?
??x
You apply a view component using the `Component.InvokeAsync` method with the name of the view component class as the argument. For example, in an `.cshtml` file:
```csharp
@await Component.InvokeAsync("CitySummary")
```
Alternatively, you can use a custom HTML tag helper like `<vc:city-summary />`. The `vc:` prefix is derived from the first letter of "view component".
x??

---
#### Using View Components with Tag Helpers
Tag helpers are custom HTML elements managed by C# classes. They allow applying view components using an HTML element that acts as a tag helper.

:p Can you use tag helpers to apply view components?
??x
Yes, you can use tag helpers to apply view components. This is done by adding the appropriate `@addTagHelper` directive in the `_ViewImports.cshtml` file and then using a custom HTML element like `<vc:city-summary />`. The `vc:` prefix is used for the view component class name.
x??

---
#### Applying View Components to Razor Pages
Razor Pages also support applying view components, either through the `Component` property or via a custom HTML tag helper. You need to add the appropriate directive in their own `_ViewImports.cshtml` file.

:p How do you apply view components in Razor Pages?
??x
In Razor Pages, you can apply view components using the `Component.InvokeAsync` method or a custom HTML element `<vc:city-summary />`. You must add the necessary `@addTagHelper` directive in the `_ViewImports.cshtml` file for Razor Pages.
x??

---
#### Example of Using View Components
An example demonstrates creating and applying a view component to display city summary data. The `CitySummary` class uses dependency injection for the `CitiesData` service.

:p How does the CitySummary view component work?
??x
The `CitySummary` view component uses dependency injection for the `CitiesData` service. It counts cities and sums their populations in the `Invoke` method:
```csharp
public string Invoke()
{
    return $"{data.Cities.Count()} cities, {data.Cities.Sum(c => c.Population)} people";
}
```
It provides this summary data to a partial view.
x??

---

#### Understanding View Components in Razor Pages
Background context: The provided text explains how to use view components within a Razor Page, specifically within an ASP.NET Core application. View components are reusable UI components that can be embedded into views or pages. They help modularize UI logic and promote code reusability.
:p How does the `ViewComponent` class in ASP.NET Core work?
??x
The `ViewComponent` class allows developers to create reusable, server-side components for their Razor Pages and MVC applications. When a view component is invoked, it can return various types of results such as partial views or custom HTML content.

To use a view component within a Razor Page, you typically define an instance in the page's code file (e.g., `Data.cshtml.cs`), inject necessary services, and then call the `InvokeAsync()` method to render the component. The example shows using a simple view component named `CitySummary` which returns a summary of cities and their population.

```csharp
// Data.cshtml.cs - Razor Page code-behind file
@page
@inject DataContext context;
public class DataModel : PageModel
{
    public IActionResult OnGet()
    {
        // Invoke the CitySummary view component to display city data
        return Page();
    }
}
```
x??

---

#### Using a View Component in a Razor Page
Background context: The provided text demonstrates how to use a `CitySummary` view component within a Razor Page (`Data.cshtml`). A view component is embedded into the page and rendered using an `<vc:city-summary />` tag.

:p How do you embed and invoke a view component in a Razor Page?
??x
You can embed and invoke a view component in a Razor Page by using the `vc:` prefix followed by the name of the view component. In this case, the `CitySummary` view component is embedded into the page using `<vc:city-summary />`. The `InvokeAsync()` method within the view component's code file (e.g., `CitySummary.cs`) handles rendering and returning a result.

Example:
```html
<!-- Data.cshtml - Razor Page file -->
<h5 class="bg-primary text-white text-center m-2 p-2">Categories</h5>
<ul class="list-group m-2">
    @foreach (Category c in context.Categories)
    {
        <li class="list-group-item">@c.Name</li>
    }
</ul>
<div class="bg-info text-white m-2 p-2">
    <vc:city-summary />
</div>
```
x??

---

#### Returning a Partial View from a View Component
Background context: The provided text explains how to return a partial view as a result of a `ViewComponent` method. The most common scenario is using the `View()` method within the `Invoke()` or `InvokeAsync()` methods of the view component.

:p How does the `View()` method work in a View Component?
??x
The `View()` method allows a view component to return a partial view as its result. You can call this method without specifying a name, which tells Razor to use the default view (typically named `Default.cshtml`). Alternatively, you can specify a custom view name.

Example:
```csharp
// CitySummary.cs - View Component file
using Microsoft.AspNetCore.Mvc;
using WebApp.Models;

namespace WebApp.Components
{
    public class CitySummary : ViewComponent
    {
        private CitiesData data;
        
        public CitySummary(CitiesData cdata)
        {
            data = cdata;
        }

        public IViewComponentResult Invoke()
        {
            return View(new CityViewModel
            {
                Cities = data.Cities.Count(),
                Population = data.Cities.Sum(c => c.Population)
            });
        }
    }
}
```
x??

---

#### Customizing Search Paths for Razor Views in View Components
Background context: The provided text explains the search paths that ASP.NET Core uses to find views when a view component is invoked. These paths are different based on whether the view component is used with a controller or a Razor Page.

:p What are the search locations for a view component used with a controller?
??x
When a view component is used within an MVC controller, ASP.NET Core searches for the view in these locations:
1. `/Views/[controller]/Components/[viewcomponent]/Default.cshtml`
2. `/Views/Shared/Components/[viewcomponent]/Default.cshtml`
3. `/Pages/Shared/Components/[viewcomponent]/Default.cshtml`

For example, if a `CitySummary` component is used with an `HomeController`, the search path would be:
- /Views/Home/Components/CitySummary/Default.cshtml

If the view component is used with a Razor Page, the paths are different:
1. `/Pages/Components/[viewcomponent]/Default.cshtml`
2. `/Pages/Shared/Components/[viewcomponent]/Default.cshtml`
3. `/Views/Shared/Components/[viewcomponent]/Default.cshtml`

x??

---

#### Defining and Using a View Component ViewModel
Background context: The provided text describes how to define a view component's model class, which is used as the data passed from the view component to the Razor view.

:p How do you create a `CityViewModel` for use with the `CitySummary` view component?
??x
To create a `CityViewModel` for the `CitySummary` view component, you define a class that holds the necessary properties. This model will be used to pass data from the view component's `Invoke()` method to the Razor view.

Example:
```csharp
// CityViewModel.cs - Model file
namespace WebApp.Models
{
    public class CityViewModel
    {
        public int? Cities { get; set; }
        public int? Population { get; set; }
    }
}
```
x??

---

