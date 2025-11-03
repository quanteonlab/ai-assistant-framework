# Flashcards: Pro-ASPNET-Core-7_processed (Part 71)

**Starting Chapter:** 7.5 Styling the content. 7.5.1 Installing the Bootstrap package. 7.5.2 Applying Bootstrap styles

---

---
#### Installing LibMan for Client-Side Package Management
Background context: In order to implement Bootstrap, we need to manage client-side packages using a tool. The provided section explains how to install and initialize LibMan for this purpose.

:p How do you install and initialize LibMan in the SportsStore project?
??x
To install and initialize LibMan in the SportsStore project, first uninstall any existing global LibMan package with:
```sh
dotnet tool uninstall --global Microsoft.Web.LibraryManager.Cli
```
Then, install the required version of LibMan using:
```sh
dotnet tool install --global Microsoft.Web.LibraryManager.Cli --version 2.1.175
```
After installing LibMan, initialize it in the SportsStore project folder and install the Bootstrap package with:
```sh
libman init -p cdnjs
libman install bootstrap@5.2.3 -d wwwroot/lib/bootstrap
```
This sets up LibMan to manage client-side packages and installs Bootstrap for styling purposes.
x??
---
#### Applying Bootstrap Styles in _Layout.cshtml
Background context: To use Bootstrap styles, the `bootstrap.min.css` stylesheet needs to be included in the `_Layout.cshtml` file. This allows consistent application-wide styling.

:p How does the inclusion of Bootstrap in the `_Layout.cshtml` affect the overall appearance of the SportsStore application?
??x
Including the Bootstrap stylesheet in the `_Layout.cshtml` file ensures that all views using this layout will inherit its styles, providing a uniform look across the entire application. The relevant code snippet includes:
```html
<link href=\"/lib/bootstrap/css/bootstrap.min.css\" rel=\"stylesheet\" />
```
This line is added within the `<head>` section of the `_Layout.cshtml` file.

Additionally, a basic header and two-column layout are defined with:
```html
<div class="bg-dark text-white p-2">
    <span class="navbar-brand ml-2">SPORTS STORE</span>
</div>

<div class="row m-1 p-1">
    <div id="categories" class="col-3">
        Put something useful here later
    </div>
    <div class="col-9">@RenderBody()</div>
</div>
```
These elements provide a consistent header and structure for the application.
x??
---
#### Styling Products in Index.cshtml with Bootstrap
Background context: The `Index.cshtml` view uses Bootstrap classes to style product listings, ensuring consistency across the application.

:p How are products styled in the `Index.cshtml` file using Bootstrap classes?
??x
Products are styled using Bootstrap classes within the `Index.cshtml` file. For example:
```html
<div class="card card-outline-primary m-1 p-1">
    <div class="bg-faded p-1">
        <h4>
            @p.Name
            <span class="badge rounded-pill bg-primary text-white" style="float:right">
                <small>@p.Price.ToString("c")</small>
            </span>
        </h4>
    </div>
    <div class="card-text p-1">@p.Description</div>
</div>
```
These classes provide consistent styling for product cards, including a background color and badge for prices.

Additionally, pagination buttons are styled using custom attributes in the `PageLinkTagHelper`:
```html
<div page-model="@Model.PagingInfo" page-action="Index"
     page-classes-enabled="true" 
     page-class="btn" 
     page-class-normal="btn-outline-dark" 
     page-class-selected="btn-primary" class="btn-group pull-right m-1">
</div>
```
These attributes map to properties in the `PageLinkTagHelper` and apply specific Bootstrap classes.
x??
---

#### Tag Helper for Paging
Tag helpers are special HTML helper methods that can be used to simplify common web page tasks such as generating a set of pagination links. In this example, tag helpers were used to generate paging controls for the SportsStore application.

The `Paging` class was created to manage the state required for the paging functionality, including the current page and other relevant information.
:p How does the `Paging` class help with managing the paging state?
??x
The `Paging` class stores necessary details like the current page number and additional paging information, allowing tag helpers to generate the appropriate links. This separation of concerns helps in maintaining cleaner Razor views.

For instance, the `PagingTagHelper` class uses properties from the `Paging` model to determine which classes to apply to each link based on whether it is the currently selected page or not.
x??

---
#### Applying Styles with Tag Helpers
Tag helpers can be used to dynamically apply styles and other attributes to HTML elements based on data bound in your Razor views. The `PageClass`, `PageClassSelected`, and `PageClassNormal` properties of the `Paging` class are mapped to specific CSS classes.

:p How do tag helpers handle attribute mapping from C# property names to HTML attribute names?
??x
Tag helpers use conventions for mapping between C# property names and HTML attribute names. For instance, the `PageClassSelected` and `PageClassNormal` properties map to the `page-class-selected` and `page-class-normal` attributes respectively.

The following line of code in the tag helper demonstrates how these mapped values are used:
```csharp
tag.AddCssClass(i == PageModel.CurrentPage ? PageClassSelected : PageClassNormal);
```
x??

---
#### Creating a Partial View for Product Summaries
Partial views allow you to create reusable snippets of Razor markup that can be included in other views. In this example, the `ProductSummary` partial view was created and used within the main `Index.cshtml` file.

:p How does using a partial view benefit an application?
??x
Using a partial view helps reduce code duplication by defining common content once and reusing it across multiple views. This makes maintenance easier and keeps your views cleaner.

The `ProductSummary` partial view is defined in `Views/Shared/ProductSummary.cshtml`, and the main view includes it using the `<partial>` tag helper with the appropriate model passed to it.
x??

---
#### Razor Partial View Syntax
A partial view in ASP.NET Core can be included in a main view by using the `<partial>` tag helper, which requires specifying the name of the partial view and the model that should be passed to it.

:p How do you include a partial view within another Razor view?
??x
You include a partial view by using the `@partial` or `<partial>` tag helper in your Razor view. The syntax is as follows:

```html
<partial name="PartialViewName" model="yourModelObject">
```

In this context, the `name` attribute specifies the filename of the partial view (without the `.cshtml` extension), and the `model` attribute passes a model object to the partial view.
x??

---
#### Paging Tag Helper Example
The paging functionality in the SportsStore application was implemented using a tag helper that generates navigation links for different pages. The `PagingTagHelper` class uses the properties of the `Paging` model to conditionally apply CSS classes and generate HTML.

:p What does the `<partial>` element do in the `Index.cshtml` file?
??x
The `<partial>` element is used within a Razor view to include another partial view. In this example, it embeds the `ProductSummary` partial view for each product listed on the page.

Here's how it looks in code:
```razor
<partial name="ProductSummary" model="p" />
```
This line of code takes a model object and renders the `ProductSummary.cshtml` partial view with that specific model passed to it.
x??

---

---
#### Creating SportsStore ASP.NET Core Project
Background context: The SportsStore project is an example application created using ASP.NET Core, showcasing a typical web development setup. It demonstrates how to build and integrate various components like MVC architecture, Entity Framework for database operations, and HTML generation with CSS.

:p How is the SportsStore project created?
??x
The SportsStore project is created by utilizing the basic ASP.NET Core template provided by Visual Studio or any other development environment that supports .NET Core. This template sets up a web application structure that includes controllers, views, models, and configurations necessary for building an ASP.NET Core application.
```csharp
// Example code to create a new ASP.NET Core project using dotnet CLI:
dotnet new mvc -n SportsStore
cd SportsStore
```
x??
---

---
#### ASP.NET Core Integration with Entity Framework Core
Background context: Entity Framework Core is tightly integrated into the ASP.NET Core framework. It simplifies data access and management in .NET applications by providing an object-oriented API to work with relational databases.

:p What is the integration between ASP.NET Core and Entity Framework Core?
??x
ASP.NET Core has a seamless integration with Entity Framework Core, which allows developers to easily manage and interact with relational databases using an object-relational mapping (ORM) approach. This integration simplifies database operations such as creating, reading, updating, and deleting data.
```csharp
// Example code for setting up DbContext in ASP.NET Core:
public class ApplicationDbContext : DbContext {
    public DbSet<Product> Products { get; set; }
}
```
x??
---

---
#### Pagination with Data Requests
Background context: Paginating data is a common requirement in web applications to manage the display of large datasets. In the SportsStore project, pagination can be implemented by including the page number as part of the request URL or query string and using it when querying the database.

:p How can data pagination be achieved in an ASP.NET Core application?
??x
Data pagination is achieved in ASP.NET Core applications by adding a `page` parameter to the URL or query string. This value is then used within the controller to limit the number of records returned from the database, effectively splitting the large dataset into multiple pages.
```csharp
// Example code for paginating data:
int pageNumber = int.Parse(Request.Query["pageNumber"]);
int pageSize = 10; // Define page size

var products = await _context.Products.Skip((pageNumber - 1) * pageSize).Take(pageSize).ToListAsync();
```
x??
---

---
#### Styling with CSS Frameworks
Background context: To enhance the visual appearance of web applications, ASP.NET Core can leverage popular CSS frameworks like Bootstrap. These frameworks provide pre-designed styles and components that can be quickly applied to HTML content generated by the application.

:p How can CSS frameworks be used in an ASP.NET Core project?
??x
CSS frameworks such as Bootstrap can be used in ASP.NET Core projects by including their CSS files in the layout or specific views of the web application. This allows developers to apply a wide range of predefined styles and components without manually writing CSS code.

To include Bootstrap, you can use CDN links or download it locally:

```html
<!-- Example using CDN -->
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
```

Or, for local files:
```html
<!-- Example with local files -->
<link rel="stylesheet" href="~/css/bootstrap.min.css">
```
x??
---

