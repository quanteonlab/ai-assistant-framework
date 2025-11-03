# Flashcards: Pro-ASPNET-Core-7_processed (Part 124)

**Starting Chapter:** 24.6 Creating view components classes

---

#### Creating View Components Classes
Background context: In ASP.NET Core, view components are reusable pieces of UI that can be embedded within a Razor page or another view. They provide a way to encapsulate common UI patterns and share them across different parts of your application.

View components often summarize complex operations handled by controllers or Razor Pages. For example, you might have a summary of a shopping basket, which links to the detailed list provided by a controller.

:p How does a hybrid class work in ASP.NET Core?
??x
A hybrid class can act as both a page model and a view component depending on how it is invoked. When used within a Razor page or view, it behaves like a view component, allowing you to use it for summaries or snapshots of functionality. When used directly as a Razor page, it acts like a normal page model.

For example, in the provided text, the `CitiesModel` class can be used both as a view component and as a page model by different URLs:

```csharp
[ViewComponent(Name = "CitiesPageHybrid")]
public class CitiesModel : PageModel
{
    // ... existing code ...
}
```

When you request `http://localhost:5000/cities`, the class acts as a view component, and when you request `http://localhost:5000/data`, it acts as a page model.

```csharp
@page
@model WebApp.Pages.CitiesModel

<div class="m-2">
    <table class="table table-sm table-striped table-bordered">
        @foreach (City c in Model.Data?.Cities ?? Enumerable.Empty<City>())
        {
            <tr>
                <td>@c.Name</td>
                <td>@c.Country</td>
                <td>@c.Population</td>
            </tr>
        }
    </table>
</div>

// In another page, it acts as a view component
@page
@inject DataContext context;

<h5 class="bg-primary text-white text-center m-2 p-2">Categories</h5>
<ul class="list-group m-2">
    @foreach (Category c in context.Categories)
    {
        <li class="list-group-item">@c.Name</li>
    }
</ul>

<div class="bg-info text-white m-2 p-2">
    <vc:cities-page-hybrid />
</div>
```

x??

---
#### Using ViewComponentContext Attribute
Background context: The `ViewComponentContext` attribute is used to signal that a property should be assigned an object that defines the properties described in Table 24.5 before the `Invoke` or `InvokeAsync` method is invoked.

:p How does the `ViewComponentContext` attribute work?
??x
The `ViewComponentContext` attribute is used to initialize certain properties of a view component, such as `ViewData`, `TempData`, and other context-related data. This allows you to pass additional information or state to your view component without directly modifying the model.

In the provided example:

```csharp
[ViewComponent(Name = "CitiesPageHybrid")]
public class CitiesModel : PageModel
{
    public ViewComponentContext Context { get; set; } = new();

    [ViewComponentContext]
    public void Invoke()
    {
        return new ViewViewComponentResult()
        {
            ViewData = new ViewDataDictionary<CityViewModel>( 
                Context.ViewData, 
                new CityViewModel 
                { 
                    Cities = Data?.Cities.Count(), 
                    Population = Data?.Cities.Sum(c => c.Population) 
                } 
            ) 
        };
    }
}
```

Here, `ViewComponentContext` initializes the context object with the necessary properties before the `Invoke` method is called.

x??

---
#### Summary of Hybrid Classes
Background context: A hybrid class can act as both a page model and a view component. By decorating it with the appropriate attributes, you can control its behavior depending on how it is invoked.

:p What are the differences between using a hybrid class as a page model versus a view component?
??x
When used directly in a Razor page (as shown in `Data.cshtml`), the hybrid class acts as a normal page model and processes the request independently. However, when embedded within another Razor file or view (as shown in the `Default.cshtml` for view components), it behaves like a standard view component.

For example:

- **Page Model**: Directly invoked by a user's request to a specific URL.
- **View Component**: Used as a summary or snapshot of functionality and can be embedded within other views.

```csharp
@page
@model WebApp.Pages.CitiesModel

// ... existing code ...

<vc:cities-page-hybrid /> // This embeds the view component in another page
```

x??

---
#### ViewComponent Example - Cities Summary
Background context: The `Cities` view model is used to summarize city data, providing counts and population totals. This summary can be embedded within other views or pages.

:p How does the `CitiesModel` class summarize city data?
??x
The `CitiesModel` class summarizes city data by counting cities and calculating their total population. It uses these values in a view component context to provide a concise snapshot of the data.

Here’s how it works:

```csharp
public IViewComponentResult Invoke()
{
    return new ViewViewComponentResult()
    {
        ViewData = new ViewDataDictionary<CityViewModel>( 
            Context.ViewData, 
            new CityViewModel 
            { 
                Cities = Data?.Cities.Count(), 
                Population = Data?.Cities.Sum(c => c.Population) 
            } 
        ) 
    };
}
```

This code initializes the `CityViewModel` with the count of cities and their total population, which can then be displayed in a view component.

x??

---

#### Using View Components with Hybrid Controllers
Background context: In this section, we learn how to create hybrid controllers that can both return views and view components. This technique allows for more flexibility in your MVC application by combining the power of actions and view components.

:p What is a hybrid controller?
??x
A hybrid controller is a special kind of controller that can be used to return both action results (such as ViewResult) and view component results. It combines the functionality of a traditional controller with view components, allowing for more complex UI compositions.
??x

#### Creating the Hybrid Controller Class
Background context: The CitiesController class demonstrates how to create a hybrid controller by decorating it with `[ViewComponent]` and implementing both `Index()` action method and `Invoke()` method.

:p How does the CitiesController class work?
??x
The CitiesController class works as follows:
- It uses the `[ViewComponent]` attribute to indicate that this class is a view component.
- The constructor injects an instance of `CitiesData`.
- The `Index()` action returns a view with the list of cities.
- The `Invoke()` method creates and returns a view component result containing summary data about the cities.

:p What does the `Invoke()` method return?
??x
The `Invoke()` method returns an `IViewComponentResult`. Specifically, it returns a `ViewViewComponentResult` that contains a `ViewDataDictionary<CityViewModel>` with summary information about the cities.
??x

#### Using ViewData in Hybrid Controllers
Background context: The `Invoke()` method uses `ViewData` to pass data to the view component. This is done by creating a `ViewDataDictionary<CityViewModel>`.

:p How does the `Invoke()` method set up the view component result?
??x
The `Invoke()` method sets up the view component result by:
1. Creating a new instance of `ViewDataDictionary<CityViewModel>`.
2. Populating this dictionary with summary data about the cities, such as the count and total population.
3. Returning a `ViewViewComponentResult` that uses this dictionary.

:p Can you show an example of how the view component result is created in the `Invoke()` method?
??x
Sure! Here’s an example:
```csharp
public IViewComponentResult Invoke() {
    return new ViewViewComponentResult()
    {
        ViewData = new ViewDataDictionary<CityViewModel>(
            ViewData,
            new CityViewModel
            {
                Cities = data.Cities.Count(),
                Population = data.Cities.Sum(c => c.Population)
            }
        )
    };
}
```

:x?

#### Creating Views for the Hybrid Controller
Background context: To use the hybrid controller, you need to create views that handle both the action result and the view component result.

:p What is required to provide a view for the `Index()` action?
??x
To provide a view for the `Index()` action, you need to create an `Index.cshtml` file in the `Views/Cities` folder. This file should define how the list of cities will be displayed.
??x

:p What is required to provide a view for the view component result?
??x
To provide a view for the view component result, you need to create a `Default.cshtml` file in the `Views/Shared/Components/CitiesControllerHybrid` folder. This file should define how the summary information will be displayed.
??x

#### Applying the Hybrid View Component
Background context: In the Data.cshtml page, we are now going to apply the hybrid view component created.

:p How is the hybrid view component applied in the `Data.cshtml` Razor Page?
??x
The hybrid view component is applied by using the `vc:` tag helper with the appropriate name. Specifically, it looks like this:

```razor
<div class="bg-info text-white m-2 p-2">
    <vc:cities-controller-hybrid />
</div>
```
:p Can you explain what happens when the `vc:cities-controller-hybrid` is rendered?
??x
When the `vc:cities-controller-hybrid` tag helper is rendered, it will execute the `Invoke()` method of the CitiesController class. This method will return a view component result containing summary information about the cities, which will be displayed in the specified table format.
??x

---
Note: Each question and answer should be separated by "??x" and "x??" respectively.

#### View Components Overview
View components are self-contained C# classes used to generate content that isn't directly related to the main purpose of an application. They can be applied using the `Component` property or the `vc` tag helper and can use partial views for generating HTML. They also support dependency injection and can receive data from their parent view context.
:p What is a view component?
??x
View components are C# classes that generate content independently of the main controller actions, often used to modularize parts of the UI. They can be instantiated using the `Component` property or the `vc` tag helper in Razor views and support dependency injection for flexibility. Partial views can also be utilized within view components.
x??

---
#### Tag Helpers Overview
Tag helpers are C# classes that manipulate HTML elements, allowing transformations such as generating dynamic content based on application state. They enhance HTML by adding attributes that execute logic during the rendering process, ensuring consistency in styling and functionality. Tag helpers can be applied to specific elements or used with custom shorthand syntax.
:p What is a tag helper?
??x
Tag helpers are C# classes designed to transform HTML elements within views. These classes allow for dynamic content generation based on application state and context. They enhance the view by executing logic directly in the markup, making it easier to generate complex UI elements or form controls that adhere to specific styling rules.
x??

---
#### Tag Helper Class Structure
A tag helper class is structured around a base class `TagHelper`. It defines methods like `Process` which handles element transformation. The class can include attributes such as `HtmlTargetElement` and `Order` for specifying the elements it targets and the order of processing during rendering.
:p How does a basic tag helper class look?
??x
A basic tag helper class extends from `TagHelper` and may override methods like `Process`. Here's an example:
```csharp
public class ProductPriceTagHelper : TagHelper {
    public string Price { get; set; }

    public override void Process(TagHelperContext context, TagHelperOutput output) {
        // Logic to manipulate the tag helper content
    }
}
```
x??

---
#### Applying Tag Helpers in Views
To apply a tag helper in a view, you typically use an attribute like `asp-taghelper` or `<vc:component>` for view components. For custom tag helpers, you can add them directly as attributes on HTML elements.
:p How do you apply a custom tag helper to an element?
??x
You can apply a custom tag helper by adding it as an attribute to an HTML element in the Razor view. Here's an example:
```html
<div class="price" asp-taghelper="@(new ProductPriceTagHelper { Price = 99.99 })">
</div>
```
x??

---
#### Custom Tag Helper Example
Creating a custom tag helper involves defining a C# class that inherits from `TagHelper`. This class can use properties to accept data and modify the rendered HTML element, ensuring dynamic content generation.
:p How do you create a custom tag helper?
??x
To create a custom tag helper, define a class derived from `TagHelper` and implement methods like `Process`. Here's an example:
```csharp
public class ProductSummaryTagHelper : TagHelper {
    public string Name { get; set; }
    public decimal Price { get; set; }

    public override void Process(TagHelperContext context, TagHelperOutput output) {
        output.Content.SetContent($"<tr><th>Name</th><td>{Name}</td></tr>");
        output.Content.AppendHtml("<tr><th>Price</th><td>${Price:C2}</td></tr>");
    }
}
```
x??

---
#### Using Tag Helpers with Dependency Injection
Tag helpers can be registered as services and accessed within views using dependency injection. This allows for dynamic content generation based on the application state, providing flexibility.
:p Can tag helpers use dependency injection?
??x
Yes, tag helpers can use dependency injection to access application services. You register them in `Startup.cs` or `Program.cs` and then inject dependencies into your tag helper class. Here's an example:
```csharp
public void ConfigureServices(IServiceCollection services) {
    services.AddRazorPages();
    services.AddTransient<ProductService>();
}
```
x??

---
#### Tag Helper Scope Control
Tag helpers can control the scope of elements they target using the `HtmlTargetElement` attribute and the `Order` property. This allows for precise targeting of specific HTML tags or elements.
:p How do you control the scope of a tag helper?
??x
You control the scope by setting the `HtmlTargetElement` and possibly the `Order` properties in your tag helper class:
```csharp
[HtmlTargetElement("product-summary", Attributes = "name, price")]
public class ProductSummaryTagHelper : TagHelper {
    public string Name { get; set; }
    public decimal Price { get; set; }

    // Implementation details
}
```
x??

---
#### Tag Helper for Dynamic Content Generation
Tag helpers can generate dynamic content based on the context in which they are applied. This is useful for creating elements like navigation menus, product listings, or dynamic form controls that change based on user state.
:p How does a tag helper generate dynamic content?
??x
A tag helper generates dynamic content by executing logic during the rendering process. For example, it might fetch data from services and insert it into the HTML output:
```csharp
public class ProductListTagHelper : TagHelper {
    private readonly ProductService _productService;

    public ProductListTagHelper(ProductService productService) {
        _productService = productService;
    }

    public override void Process(TagHelperContext context, TagHelperOutput output) {
        var products = _productService.GetProducts();
        output.Content.SetContent("<ul>");
        foreach (var product in products) {
            output.Content.AppendHtml($"<li>{product.Name} - ${product.Price:C2}</li>");
        }
        output.Content.AppendHtml("</ul>");
    }
}
```
x??

---

