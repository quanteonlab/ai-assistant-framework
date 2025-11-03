# Flashcards: Pro-ASPNET-Core-7_processed (Part 153)

**Starting Chapter:** 31.3 Creating a Razor Pages forms application

---

#### Adding a Method to ViewModelFactory.cs for Deletion

**Background context:** In the MVC application, adding a method to handle the deletion of objects from the database is necessary. This involves creating a view model that will be used to display the data and prompt the user before removing it.

The `ViewModelFactory` class in the `Models` folder contains methods to generate different view models based on the operation needed (e.g., edit, delete). The `Delete` method specifically handles generating a view model for deletion operations.

:p What is the purpose of the `Delete` method in the `ViewModelFactory`?

??x
The purpose of the `Delete` method in the `ViewModelFactory` is to generate a `ProductViewModel` that includes information about the product to be deleted, along with contextual data like categories and suppliers. This view model is used to present the deletion form to the user.

```csharp
public static class ViewModelFactory {
    public static ProductViewModel Delete(Product p, IEnumerable<Category> categories,
                                          IEnumerable<Supplier> suppliers) {
        return new ProductViewModel { 
            Product = p, 
            Action = "Delete", 
            ReadOnly = true, 
            Theme = "danger", 
            Categories = categories, 
            Suppliers = suppliers 
        };
    }
}
```

x??

---

#### Adding Action Methods to Handle Deletion in HomeController.cs

**Background context:** In the `HomeController`, action methods are added to handle both the GET and POST requests for deleting a product. The GET request displays the form to confirm the deletion, while the POST request processes the actual removal of the data from the database.

The `Delete` method retrieves the product by its ID and returns it as part of a view model if found. If not found, it returns a 404 error. The `HttpPost` method removes the product from the database using Entity Framework Core and redirects to the index action.

:p What are the two actions methods added in the HomeController for deletion?

??x
The two action methods added in the `HomeController` for handling deletion are:

1. **GET Method (`Delete(long id)`):** This method retrieves a product by its ID from the database, checks if it exists, and then returns a view model with the necessary data to display the delete form.

```csharp
public async Task<IActionResult> Delete(long id) {
    Product? p = await context.Products.FindAsync(id);
    if (p != null) {
        ProductViewModel model = ViewModelFactory.Delete(p, Categories, Suppliers);
        return View("ProductEditor", model);
    }
    return NotFound();
}
```

2. **POST Method (`[HttpPost] Delete(Product product)`):** This method processes the form submission by removing the product from the database and saving changes.

```csharp
[HttpPost]
public async Task<IActionResult> Delete(Product product) {
    context.Products.Remove(product);
    await context.SaveChangesAsync();
    return RedirectToAction(nameof(Index));
}
```

x??

---

#### Creating a Razor Pages Forms Application

**Background context:** In Razor Pages, creating forms for CRUD operations (Create, Read, Update, and Delete) follows a similar pattern but is broken into smaller parts. The main challenge lies in maintaining the modular nature of the application without duplicating code or markup.

The `Index.cshtml` file serves as a starting point to display a list of products along with links to other CRUD operations like details, edit, and delete. Each operation is handled by separate Razor Pages (e.g., Details, Edit, Delete).

:p What does the `IndexModel: PageModel` class in the `Index.cshtml` do?

??x
The `IndexModel: PageModel` class in the `Index.cshtml` file serves as a model for the page. It initializes and manages the data needed to display the list of products, including any related categories or suppliers.

```csharp
public class IndexModel : PageModel {
    private DataContext context;

    public IndexModel(DataContext dbContext) {
        context = dbContext;
    }

    public IEnumerable<Product> Products { get; set; } 
      = Enumerable.Empty<Product>();

    public void OnGetAsync(long id = 1) {
        Products = context.Products
            .Include(p => p.Category).Include(p => p.Supplier);
    }
}
```

This class provides methods and properties to fetch data from the database, making it reusable for different operations.

x??

---

#### Partial Views in Razor Pages
Background context explaining how partial views are used to share common UI elements across different Razor pages. This allows for modular and reusable components.

:p How do partial views facilitate code reuse in Razor Pages?
??x
Partial views allow you to define reusable UI components that can be included in multiple Razor pages without duplicating the HTML code. For example, a `_ProductEditor.cshtml` partial view can be shared among different operations like creating or editing products.

Here's an excerpt from the provided text:

```html
@model ProductViewModel

<partial name="_Validation" />
<h5 class="bg-@Model?.Theme text-white text-center p-2">@Model?.Action</h5>
<form asp-page="@Model?.Action" method="post">
    <div class="form-group">
        <label asp-for="Product.ProductId"></label>
        <input class="form-control" asp-for="Product.ProductId" readonly />
    </div>
    <!-- more form fields -->
</form>
```

The partial view includes a form and validation elements, which can be included in different Razor pages by using the `@partial` directive.
x??

---

#### Page Model Base Class
Background context explaining why defining a base class for page models is beneficial. It allows you to consolidate common functionality such as data access or business logic.

:p Why would you define a base class for your page models?
??x
Defining a base class for Razor Page models helps in consolidating common code, such as data access methods and properties that are shared across multiple pages. This approach promotes modularity and reduces code duplication. For instance, the `EditorPageModel` class defines common properties like `DataContext`, `Categories`, and `Suppliers`.

Here's an excerpt from the provided text:

```csharp
public class EditorPageModel : PageModel {
    public EditorPageModel(DataContext dbContext) {
        DataContext = dbContext;
    }

    public DataContext DataContext { get; set; }
    public IEnumerable<Category> Categories => DataContext.Categories;
    public IEnumerable<Supplier> Suppliers => DataContext.Suppliers;
    public ProductViewModel? ViewModel { get; set; }
}
```

This base class provides a common starting point for other page models, ensuring that they have access to shared data and functionality.
x??

---

#### Handling Multiple Page Operations
Background context explaining the use of `asp-page-handler` attribute in Razor Pages. This allows a single page to handle multiple operations by specifying different handler methods.

:p How does using `asp-page-handler` benefit Razor Pages?
??x
Using the `asp-page-handler` attribute in Razor Pages enables you to define pages that can perform more than one operation, such as handling both GET and POST requests with the same page. This is useful for actions like creating or editing entities where the same form might be used but different handlers process the data.

Here's an example of how it could be used:

```html
<form asp-page="@(Model?.Action)" method="post" asp-page-handler="Save">
    <!-- form fields -->
</form>
```

The `asp-page-handler` attribute allows you to specify which handler method should handle the request, making the page more versatile and flexible.
x??

---

#### Common ViewModel Usage
Background context explaining how ViewModels are used in Razor Pages. They help separate concerns by encapsulating data and logic specific to a view.

:p How do ViewModels enhance the separation of concerns in Razor Pages?
??x
ViewModels in Razor Pages help maintain the separation of concerns by encapsulating the model-specific data that a view needs to display or modify. This keeps the UI logic clean and decoupled from business logic, making the application easier to manage.

Here's an excerpt from the provided text:

```csharp
@model ProductViewModel

<partial name="_Validation" />
<h5 class="bg-@Model?.Theme text-white text-center p-2">@Model?.Action</h5>
<form asp-page="@(Model?.Action)" method="post">
    <div class="form-group">
        <label asp-for="Product.ProductId"></label>
        <input class="form-control" asp-for="Product.ProductId" readonly />
    </div>
    <!-- more form fields -->
</form>
```

The `ProductViewModel` defines the structure of data that is passed to and from the Razor page, ensuring that only relevant properties are exposed to the view.
x??

---

#### Consolidating Common Content
Background context explaining how consolidating common content in partial views and a shared base class helps in reducing code duplication.

:p How does consolidating common content in partial views and base classes help?
??x
Consolidating common content in partial views and a shared base class reduces code duplication across multiple Razor pages. This approach makes the application more maintainable and easier to update, as changes can be made in one place rather than scattered throughout several files.

For instance, the `EditorPageModel` defines common properties and methods that are then used by specific page models, ensuring consistency and reducing redundancy.

```csharp
public class EditorPageModel : PageModel {
    public EditorPageModel(DataContext dbContext) {
        DataContext = dbContext;
    }

    public DataContext DataContext { get; set; }
    public IEnumerable<Category> Categories => DataContext.Categories;
    public IEnumerable<Supplier> Suppliers => DataContext.Suppliers;
    public ProductViewModel? ViewModel { get; set; }
}
```

This base class provides a common foundation for handling CRUD operations, making the application more modular and easier to manage.
x??

---

#### Razor Pages for CRUD Operations
Background context: The text describes how to create a set of Razor Pages (Details, Create, Edit, Delete) that handle different CRUD operations using a shared base class and partial views. This approach leverages inheritance and reusability to simplify the implementation.

:p What is the purpose of creating multiple Razor Pages for handling different CRUD operations?
??x
The purpose is to modularize the code by separating each operation (viewing, creating, editing, deleting) into its own page while maintaining a consistent structure and logic through inheritance. This approach promotes reusability, maintainability, and adherence to the DRY (Don't Repeat Yourself) principle.

For example, using a base class `EditorPageModel` allows shared code for common tasks like setting up the context and handling model binding.
```csharp
public abstract class EditorPageModel : PageModel
{
    protected readonly DataContext DbContext;

    public EditorPageModel(DataContext dbContext)
    {
        DbContext = dbContext;
    }
}
```
x??

---

#### Details.cshtml - Product Detail View
Background context: The `Details.cshtml` page is responsible for displaying the details of a product. It uses a partial view (`_ProductEditor`) to present the data.

:p What does the `OnGetAsync` method in the `DetailsModel` class do?
??x
The `OnGetAsync` method retrieves the product from the database based on its ID and populates a `ViewModel` object using the `ViewModelFactory`. If no product is found, it creates an empty product with a default name.

```csharp
public async Task OnGetAsync(long id)
{
    Product? p = await DataContext.Products.
        Include(p => p.Category).Include(p => p.Supplier)
        .FirstOrDefaultAsync(p => p.ProductId == id);
    
    ViewModel = ViewModelFactory.Details(
        p ?? new () { Name = string.Empty });
}
```
x??

---

#### Create.cshtml - Product Creation Form
Background context: The `Create.cshtml` page is used to create a new product. It uses the same partial view (`_ProductEditor`) for form rendering.

:p What happens when the user submits the creation form on the `Create.cshtml` page?
??x
When the user submits the creation form, the `OnPostAsync` method processes the submitted data. If the model state is valid, it adds a new product to the database and redirects to the index page. Otherwise, it repopulates the form with error messages.

```csharp
public async Task<IActionResult> OnPostAsync(
    [FromForm] Product product)
{
    if (ModelState.IsValid)
    {
        product.ProductId = default;
        product.Category = default;
        product.Supplier = default;

        DataContext.Products.Add(product);
        await DataContext.SaveChangesAsync();
        return RedirectToPage(nameof(Index));
    }

    ViewModel = ViewModelFactory.Create(
        product, Categories, Suppliers);
    return Page();
}
```
x??

---

#### Edit.cshtml - Product Editing Form
Background context: The `Edit.cshtml` page is used to edit an existing product. It uses the partial view (`_ProductEditor`) for form rendering and updates the product in the database.

:p What does the `OnPostAsync` method do when editing a product on the `Edit.cshtml` page?
??x
The `OnPostAsync` method handles the update of a product based on user input. If the model state is valid, it updates the specified product and saves changes to the database. After saving, it redirects to the index page.

```csharp
public async Task<IActionResult> OnPostAsync(
    [FromForm] Product product)
{
    if (ModelState.IsValid)
    {
        product.Category = default;
        product.Supplier = default;

        DataContext.Products.Update(product);
        await DataContext.SaveChangesAsync();
        return RedirectToPage(nameof(Index));
    }

    ViewModel = ViewModelFactory.Edit(
        product, Categories, Suppliers);
    return Page();
}
```
x??

---

#### Delete.cshtml - Product Deletion Form
Background context: The `Delete.cshtml` page is used to delete an existing product. It uses the partial view (`_ProductEditor`) for form rendering and removes the product from the database.

:p What does the `OnPostAsync` method do when deleting a product on the `Delete.cshtml` page?
??x
The `OnPostAsync` method processes the deletion of a product based on user input. It first finds the specified product, then removes it from the database and saves changes to the database. After saving, it redirects to the index page.

```csharp
public async Task<IActionResult> OnPostAsync(
    [FromForm] Product product)
{
    DataContext.Products.Remove(product);
    await DataContext.SaveChangesAsync();
    return RedirectToPage(nameof(Index));
}
```
x??

---

