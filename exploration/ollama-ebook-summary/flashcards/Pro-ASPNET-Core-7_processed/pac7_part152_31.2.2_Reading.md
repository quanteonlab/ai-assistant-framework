# Flashcards: Pro-ASPNET-Core-7_processed (Part 152)

**Starting Chapter:** 31.2.2 Reading data

---

#### MVC Forms Application Overview
In a typical MVC (Model-View-Controller) application, the **ProductViewModel** object serves as a bridge between the model and the view. It holds both data and metadata necessary to display or edit product information. Each property of the Product class corresponds to an HTML element such as labels, inputs, selects, and spans for validation messages.

The `asp-for` attribute ensures that tag helpers transform these elements properly based on the underlying property.
:p What is the primary role of a **ProductViewModel** in an MVC application?
??x
A **ProductViewModel** serves as a container that holds both data (from the model) and instructions (such as validation messages, input types) for rendering in the view. This allows for dynamic and interactive user interfaces where data can be displayed, edited, or validated.
x??

---
#### ViewModelFactory Class
The `ViewModelFactory` class is used to create different variations of a `ProductViewModel` depending on the task at hand. For instance, when displaying product details, it ensures that related categories and suppliers are fetched from the database.

Here's how you can implement the `Details` method in `ViewModelFactory.cs`:
```csharp
public static class ViewModelFactory {
    public static ProductViewModel Details(Product p) {
        return new ProductViewModel {
            Product = p,
            Action = "Details",
            ReadOnly = true,
            Theme = "info",
            ShowAction = false,
            Categories = p == null || p.Category == null 
                ? Enumerable.Empty<Category>() 
                : new List<Category> { p.Category },
            Suppliers = p == null || p.Supplier == null 
                ? Enumerable.Empty<Supplier>() 
                : new List<Supplier> { p.Supplier },
        };
    }
}
```
:p How does the `ViewModelFactory` class help in creating a `ProductViewModel` for viewing product details?
??x
The `ViewModelFactory` class provides methods like `Details` to generate a `ProductViewModel` that is specifically configured for displaying product details. This includes setting properties such as `ReadOnly`, `Theme`, and `ShowAction` appropriately, along with fetching related categories and suppliers from the database.
x??

---
#### HomeController Action Method
In the `HomeController`, an action method named `Details` retrieves a specific product based on its ID and then uses the `ViewModelFactory` to generate a read-only view model. This view model is then passed to the `ProductEditor` view for rendering.

Here’s how you can implement the `Details` action in `HomeController.cs`:
```csharp
[AutoValidateAntiforgeryToken]
public class HomeController : Controller {
    private DataContext context;

    public HomeController(DataContext data) {
        context = data;
    }

    [HttpGet("details/{id}")]
    public async Task<IActionResult> Details(long id) {
        Product? p = await context.Products
            .Include(p => p.Category)
            .Include(p => p.Supplier)
            .FirstOrDefaultAsync(p => p.ProductId == id)
            ?? new () { Name = string.Empty };
        
        ProductViewModel model = ViewModelFactory.Details(p);
        return View("ProductEditor", model);
    }
}
```
:p How does the `Details` action method in `HomeController.cs` work?
??x
The `Details` action method fetches a product by its ID from the database, including related categories and suppliers. It then uses the `ViewModelFactory.Details` method to create a read-only `ProductViewModel`. This view model is passed to the `ProductEditor` view for rendering, ensuring that only relevant details are displayed.
x??

---
#### Displaying Product Details
When you request the `Details` action with an ID via the URL (e.g., `http://localhost:5000/details/123`), it returns a read-only form where product details can be viewed. This is achieved by setting the `ReadOnly` property to true in the `ProductViewModel`.

:p How do you test the functionality of displaying product details?
??x
You can test this by restarting your ASP.NET Core application and navigating to the URL that corresponds to a specific product ID, such as `http://localhost:5000/details/123`. This will trigger the `Details` action method in the controller, which retrieves the product and its related categories/suppliers. The read-only view model is then passed to the `ProductEditor` view for rendering.
x??

---
#### Summary of Concepts
- **ProductViewModel**: A class that holds data and instructions for displaying or editing products in a form.
- **ViewModelFactory**: A utility class for generating different configurations of ProductViewModels based on context (e.g., details, edit).
- **HomeController**: Contains action methods to handle user requests and interact with the database.
- **Details Action Method**: Fetches a product by ID, sets up a read-only view model, and renders it in the `ProductEditor` view.

These concepts work together to enable dynamic data presentation and interaction within an MVC application.

#### Creating Data in MVC Forms Application
Background context: This section describes how to create data within an MVC forms application, focusing on model binding and validation. It involves adding methods to handle both GET and POST requests for creating a new product.

:p What is the purpose of the `Create` method with the `[HttpPost]` attribute?
??x
The purpose of the `Create` method with the `[HttpPost]` attribute is to handle the submission of the form data when the user clicks the "Create" button. This method uses model binding to receive the data from the request and performs validation on it before saving it to the database.

```csharp
[HttpPost]
public async Task<IActionResult> Create(
    [FromForm] Product product)
{
    if (ModelState.IsValid) {
        // Prepare the object for storage in the database by resetting properties.
        product.ProductId = default;
        product.Category = default;
        product.Supplier = default;

        context.Products.Add(product);
        await context.SaveChangesAsync();

        return RedirectToAction(nameof(Index));
    }

    // If validation fails, create a new view model and pass it to the view.
    return View("ProductEditor", ViewModelFactory.Create(
        product,
        Categories,
        Suppliers
    ));
}
```
x??

---

#### ViewModelFactory for Creating Data
Background context: The `ViewModelFactory` class provides methods to create `ProductViewModel` objects, which are used in views to display and edit products. Different view models are created based on whether the user is viewing or creating a new product.

:p How does the `Create` method of `ViewModelFactory` prepare a `ProductViewModel` for creating data?
??x
The `Create` method of `ViewModelFactory` prepares a `ProductViewModel` with default values and appropriate properties to be used when creating a new product. It sets the `Action` property to "Create" and initializes the `Categories`, `Suppliers`, and `Product` properties.

```csharp
public static ProductViewModel Create(
    Product product,
    IEnumerable<Category> categories,
    IEnumerable<Supplier> suppliers)
{
    return new ProductViewModel 
    { 
        Product = product, 
        Categories = categories, 
        Suppliers = suppliers 
    };
}
```
x??

---

#### Handling GET Requests for Creating Data
Background context: The `Create` method without the `[HttpPost]` attribute handles GET requests to display the form for creating a new product. It retrieves necessary data (categories and suppliers) from the database and passes it along with an empty `ProductViewModel`.

:p What does the `Create` action method return when handling GET requests?
??x
When handling GET requests, the `Create` action method returns the "ProductEditor" view with an initial `ProductViewModel` that has an empty product (with a default name) and includes categories and suppliers retrieved from the database.

```csharp
public IActionResult Create()
{
    return View("ProductEditor", 
        ViewModelFactory.Create(
            new() { Name = string.Empty }, 
            Categories, Suppliers
        ));
}
```
x??

---

#### Index Action Method for Displaying Products
Background context: The `Index` action method retrieves all products from the database along with their categories and suppliers. This data is passed to a view to display a list of products.

:p What does the `Index` action method return?
??x
The `Index` action method returns the "Product" view, passing it a list of products that are retrieved from the database and include their associated category and supplier information.

```csharp
public IActionResult Index()
{
    return View(
        context.Products 
            .Include(p => p.Category) 
            .Include(p => p.Supplier)
    );
}
```
x??

---

#### Handling POST Requests for Details Action
Background context: The `Details` action method handles GET requests to display the details of a specific product. If the product with the given ID is found, it passes a view model containing the product and related data; otherwise, it returns an empty product.

:p What does the `Details` action method do when handling GET requests?
??x
The `Details` action method handles GET requests to display the details of a specific product. It checks if a product with the given ID exists in the database. If found, it populates and returns a view model containing the product and related data (categories and suppliers). If not found, it returns an empty product.

```csharp
public async Task<IActionResult> Details(long id)
{
    Product? p = await context.Products 
        .Include(p => p.Category) 
        .Include(p => p.Supplier) 
        .FirstOrDefaultAsync(p => p.ProductId == id) 
        ?? new() { Name = string.Empty };

    ProductViewModel model = ViewModelFactory.Details(p);
    return View("ProductEditor", model);
}
```
x??

---

#### Redirecting After Data Creation
Background context: After successfully creating a new product and saving it to the database, the application redirects the user back to the `Index` action method. This ensures that users see the updated list of products.

:p How does the application handle redirection after successful data creation?
??x
After successfully creating a new product and saving it to the database, the application redirects the user back to the `Index` action method using `RedirectToAction`. This ensures that users see the updated list of products on the index page.

```csharp
if (ModelState.IsValid) {
    // Prepare the object for storage in the database by resetting properties.
    product.ProductId = default;
    product.Category = default;
    product.Supplier = default;

    context.Products.Add(product);
    await context.SaveChangesAsync();

    return RedirectToAction(nameof(Index));
}
```
x??

---

#### Validation and Error Handling
Background context: The application ensures that data is validated before it is stored in the database. If validation fails, an appropriate view model with the model-bound data is passed back to the view.

:p What happens if the `ModelState` is not valid after handling a POST request?
??x
If the `ModelState` is not valid after handling a POST request for creating a new product, the application creates a new `ProductViewModel` object that incorporates the model-bound data and passes this to the view. This ensures that any validation errors are displayed on the form.

```csharp
if (ModelState.IsValid) {
    // Prepare the object for storage in the database by resetting properties.
    product.ProductId = default;
    product.Category = default;
    product.Supplier = default;

    context.Products.Add(product);
    await context.SaveChangesAsync();

    return RedirectToAction(nameof(Index));
}

// If validation fails, create a new view model and pass it to the view.
return View("ProductEditor", 
    ViewModelFactory.Create(
        product,
        Categories,
        Suppliers
    ));
```
x??

---

---
#### Restricting User Choices for Data Entry
Background context: In real applications, providing users with restricted choices can enhance usability and reduce input errors. For example, when a user needs to select a supplier or category, it's more efficient to present them with a dropdown list of available options rather than requiring them to enter the primary key directly.
:p How does using a dropdown list help in restricting user choices?
??x
Using a dropdown list helps by providing users with a pre-defined set of valid choices. This approach is more intuitive and reduces the chance of input errors, such as entering incorrect or non-existent primary keys. For instance, when editing a product, the user can choose from existing suppliers rather than having to type in the supplier's ID.
```html
<select asp-for="Product.SupplierId" class="form-control"
        disabled="@Model?.ReadOnly"
        asp-items="@(new SelectList(Model?.Suppliers, "SupplierId", "Name"))">
    <option value="" disabled selected>Choose a Supplier</option>
</select>
```
x?
---

---
#### ViewModelFactory for Different Actions
Background context: The `ViewModelFactory` class is used to configure the view models for different actions in an MVC application. This allows the application to present data in a way that matches the intended action (e.g., viewing details, creating new objects, editing existing ones).
:p How does the `ViewModelFactory` help in configuring view models?
??x
The `ViewModelFactory` helps by providing methods tailored for different actions. For example:
- `Details`: Configures the view model for displaying detailed information.
- `Create`: Sets up the view model for creating new objects, often allowing full user input.
- `Edit`: Prepares the view model for editing existing objects, ensuring that only relevant fields are editable while keeping primary keys read-only.

Here is a simplified example of how these methods might be used:
```csharp
namespace WebApp.Models {
    public static class ViewModelFactory {
        public static ProductViewModel Details(Product p) {
            return new ProductViewModel { /* configuration */ };
        }
        
        public static ProductViewModel Create(Product product, 
                                              IEnumerable<Category> categories,
                                              IEnumerable<Supplier> suppliers) {
            return new ProductViewModel { /* configuration */ };
        }

        public static ProductViewModel Edit(Product product, 
                                            IEnumerable<Category> categories,
                                            IEnumerable<Supplier> suppliers) {
            return new ProductViewModel { /* configuration */ };
        }
    }
}
```
x?
---

---
#### Editing Data in MVC Applications
Background context: Editing data involves displaying the current state of an entity to the user and allowing them to make changes. In ASP.NET Core, this is typically done by adding action methods that fetch the entity, present it for editing, and handle the updated values when the form is submitted.
:p What steps are involved in implementing data editing in MVC applications?
??x
To implement data editing in an MVC application, follow these steps:
1. Add a method to the `ViewModelFactory` to configure the view model for editing.
2. Create action methods in the controller to handle displaying the entity and processing the updated values.

For example:
```csharp
public async Task<IActionResult> Edit(long id) {
    Product? p = await context.Products.FindAsync(id);
    if (p == null) {
        ProductViewModel model = ViewModelFactory.Edit(p, Categories, Suppliers);
        return View("ProductEditor", model);
    }
    return NotFound();
}

[HttpPost]
public async Task<IActionResult> Edit(
    [FromForm] Product product) {
    if (ModelState.IsValid) {
        context.Products.Update(product);
        await context.SaveChangesAsync();
        return RedirectToAction(nameof(Index));
    }
    return View("ProductEditor", ViewModelFactory.Edit(product, Categories, Suppliers));
}
```
x?
---

