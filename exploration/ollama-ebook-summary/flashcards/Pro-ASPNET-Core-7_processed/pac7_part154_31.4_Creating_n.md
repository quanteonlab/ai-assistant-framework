# Flashcards: Pro-ASPNET-Core-7_processed (Part 154)

**Starting Chapter:** 31.4 Creating new related data objects. 31.4.1 Providing the related data in the same request

---

---
#### Providing Related Data in the Same Request
Background context: When creating form applications, sometimes users need to create related data alongside main objects. For example, a user might want to add both a new `Product` and its associated `Category`. This can be achieved by collecting all necessary details for related objects within the same request.

:p How is it possible to collect details for a related object in the same form as the main object?
??x
It's feasible by including hidden fields or input elements specifically for the related data. In this example, a `Category` name can be collected using an `<input>` field while ensuring that these elements are initially hidden until necessary.

For instance:
```html
<div class="form-group bg-info mt-2 p-1" id="categoryGroup">
    <label class="text-white" asp-for="Category.Name">New Category Name</label>
    <input class="form-control" asp-for="Category.Name" value="" id="categoryInput" />
</div>
```
The JavaScript ensures that the category input field is hidden until needed. The `CategoryId` of the product can trigger this visibility change.

??x
In what way does the provided JavaScript handle the visibility of the new category fields?
??x
The JavaScript listens for changes in the dropdown menu (`select` element) that represents categories. If a user selects an option with a value of `-1`, indicating they want to create a new category, the script unhides and enables the input field for entering the new category's name:

```javascript
window.addEventListener("DOMContentLoaded", () => {
    function setVisibility(visible) {
        document.getElementById("categoryGroup").hidden = !visible;
        const input = document.getElementById("categoryInput");
        if (visible) {
            input.removeAttribute("disabled");
        } else {
            input.setAttribute("disabled", "disabled");
        }
    }

    setVisibility(false);
    document.querySelector("select[name='Product.CategoryId']")
        .addEventListener("change", (event) => 
            setVisibility(event.target.value === "-1")
        );
});
```
This way, the user can only enter a new category when they select `-1` from the dropdown.

??x
How does this method ensure that the `CategoryId` of the product is updated correctly?
??x
When a user selects `-1` as the value for `Product.CategoryId`, it triggers the addition of a new `Category`. The JavaScript sets the visibility of the category input field, and once the user inputs a name, the script adds this new category to the database:

```csharp
protected async Task CheckNewCategory(Product product) {
    if (product.CategoryId == -1 && string.IsNullOrEmpty(product.Category?.Name)) {
        DataContext.Categories.Add(product.Category);
        await DataContext.SaveChangesAsync();
        product.CategoryId = product.Category.CategoryId;
        ModelState.Clear();
        TryValidateModel(product);
    }
}
```
This method checks if the `CategoryId` is `-1`, indicating a new category. If so, it adds this new category to the context and saves changes. After saving, it updates the `CategoryId` of the product with the newly assigned ID.

??x
How does integrating this functionality into multiple pages avoid code duplication?
??x
By adding a common method in the base class (`EditorPageModel.cs`), you can handle related data creation uniformly across different pages without duplicating code. For instance, `CheckNewCategory` is added to manage the addition of new categories:

```csharp
using Microsoft.AspNetCore.Mvc.RazorPages;
using WebApp.Models;

namespace WebApp.Pages {
    public class EditorPageModel : PageModel {
        // other properties and methods...
        
        protected async Task CheckNewCategory(Product product) {
            if (product.CategoryId == -1 && string.IsNullOrEmpty(product.Category?.Name)) {
                DataContext.Categories.Add(product.Category);
                await DataContext.SaveChangesAsync();
                product.CategoryId = product.Category.CategoryId;
                ModelState.Clear();
                TryValidateModel(product);
            }
        }
    }
}
```
This method is then used in specific page models like `Create` and `Edit`, ensuring consistency.

??x
How does the `OnPostAsync` method handle creating a new related object during form submission?
??x
In the `OnPostAsync` method of the `CreateModel`, it checks if the `CategoryId` is `-1` and there's no existing category name. If so, it creates a new `Category`, saves it to the database, and updates the product's `CategoryId`. This ensures that the form data validates correctly after adding the related object.

```csharp
public async Task<IActionResult> OnPostAsync([FromForm] Product product) {
    await CheckNewCategory(product);
    if (ModelState.IsValid) {
        product.ProductId = default;
        product.Category = default;
        product.Supplier = default;
        DataContext.Products.Add(product);
        await DataContext.SaveChangesAsync();
        return RedirectToPage(nameof(Index));
    }
    ViewModel = ViewModelFactory.Create(product, Categories, Suppliers);
    return Page();
}
```
The `CheckNewCategory` method is called to handle the addition of a new category if necessary. If validation passes, it adds and saves the product.

??x
How does the integration into the `EditModel` ensure that changes are correctly handled?
??x
In the `EditModel`, the same `CheckNewCategory` method is used to manage related data creation when editing a product. This ensures consistency across different actions (`Create`, `Edit`) and prevents code duplication:

```csharp
@page "/pages/edit/{id}"
@model EditModel

<div class="m-2">
    <partial name="_ProductEditor" model="@Model.ViewModel" />
</div>

@functions {
    public class EditModel : EditorPageModel {
        // constructor...

        protected async Task CheckNewCategory(Product product) {
            if (product.CategoryId == -1 && string.IsNullOrEmpty(product.Category?.Name)) {
                DataContext.Categories.Add(product.Category);
                await DataContext.SaveChangesAsync();
                product.CategoryId = product.Category.CategoryId;
                ModelState.Clear();
                TryValidateModel(product);
            }
        }

        // other methods...
    }
}
```
This approach ensures that the `CheckNewCategory` method is called whenever a related object needs to be created, maintaining a clean and consistent codebase.
x??

---

#### OnGetAsync Method
This method is part of an ASP.NET Core Razor Page that handles the initial GET request for editing a product. It retrieves the specified `Product` from the database or initializes it if not found, and sets up the view model for the edit form.

:p What does this method do when handling the initial GET request for editing a product?
??x
This method retrieves the product with the given ID from the database using `FindAsync`. If no product is found, it creates a new anonymous object to initialize the ViewModel. It then populates the ViewModel with the necessary data and categories/suppliers.
```csharp
public async Task OnGetAsync(long id)
{
    Product p = await this.DataContext.Products.FindAsync(id) ?? new () { Name = string.Empty };
    ViewModel = ViewModelFactory.Edit(p, Categories, Suppliers);
}
```
x??

---

#### OnPostAsync Method
This method is responsible for handling the POST request when the user submits changes to a product. It checks if any new categories are created during this process and updates the product's details before saving them back to the database.

:p What does this method do when handling the POST request for updating a product?
??x
This method processes the submitted form data, checks for any new category creation, validates the model state, updates the product with default values (likely meaning setting Category and Supplier to `null`), saves the changes to the database, and redirects back to the index page.
```csharp
public async Task<IActionResult> OnPostAsync(
    [FromForm] Product product)
{
    await CheckNewCategory(product);
    
    if (ModelState.IsValid)
    {
        product.Category = default;
        product.Supplier = default;
        
        DataContext.Products.Update(product);
        await DataContext.SaveChangesAsync();
        
        return RedirectToPage(nameof(Index));
    }
    
    ViewModel = ViewModelFactory.Edit(product, Categories, Suppliers);
    return Page();
}
```
x??

---

#### Creating Related Data
This section discusses the process of handling related data in form applications. For more complex scenarios, it suggests breaking out to another page or controller to create new objects instead of cluttering the main form.

:p How does this approach help manage complex related data creation?
??x
Breaking out into a separate page for creating related data, like suppliers, helps keep the main edit form simpler and less overwhelming. This approach allows users to focus on the primary task while providing an easy path to create or modify associated objects.
```csharp
// Example of a simplified SupplierBreakOut.cshtml file snippet
@page "/pages/supplier"
@model SupplierPageModel

<div class="m-2">
    <h5 class="bg-secondary text-white text-center p-2">New Supplier</h5>
    <form asp-page="SupplierBreakOut" method="post">
        <div class="form-group">
            <label asp-for="Supplier.Name"></label>
            <input class="form-control" asp-for="Supplier.Name" />
        </div>
        <div class="form-group">
            <label asp-for="Supplier.City"></label>
            <input class="form-control" asp-for="Supplier.City" />
        </div>
        <button class="btn btn-secondary mt-2" type="submit">Create</button>
        <a class="btn btn-outline-secondary mt-2"
           asp-page="@Model.ReturnPage"
           asp-route-id="@Model.ProductId">
            Cancel
        </a>
    </form>
</div>

@functions {
    public class SupplierPageModel : PageModel
    {
        private DataContext context;
        public SupplierPageModel(DataContext dbContext)
        {
            context = dbContext;
        }

        [BindProperty]
        public Supplier? Supplier { get; set; }
    }
}
```
x??

---

#### Supplier Breakout Page
This specific example demonstrates how to create a separate Razor page for creating new supplier objects, allowing users to fill out the necessary information and then return to the main form.

:p How does this SupplierBreakOut.cshtml file facilitate the creation of a new supplier?
??x
The `SupplierBreakOut.cshtml` file provides an easy-to-use form where users can enter the name and city of a new supplier. Upon submission, it creates or updates the supplier object in the database without cluttering the main edit form.

```csharp
@page "/pages/supplier"
@model SupplierPageModel

<div class="m-2">
    <h5 class="bg-secondary text-white text-center p-2">New Supplier</h5>
    <form asp-page="SupplierBreakOut" method="post">
        <div class="form-group">
            <label asp-for="Supplier.Name"></label>
            <input class="form-control" asp-for="Supplier.Name" />
        </div>
        <div class="form-group">
            <label asp-for="Supplier.City"></label>
            <input class="form-control" asp-for="Supplier.City" />
        </div>
        <button class="btn btn-secondary mt-2" type="submit">Create</button>
        <a class="btn btn-outline-secondary mt-2"
           asp-page="@Model.ReturnPage"
           asp-route-id="@Model.ProductId">
            Cancel
        </a>
    </form>
</div>

@functions {
    public class SupplierPageModel : PageModel
    {
        private DataContext context;
        public SupplierPageModel(DataContext dbContext)
        {
            context = dbContext;
        }

        [BindProperty]
        public Supplier? Supplier { get; set; }
    }
}
```
x??

---

#### Product and Return Page Handling
Background context: This section explains how to handle a product and return page using GET and POST requests. The `OnGet` method processes incoming queries, while the `OnPostAsync` method handles form submissions.

:p How does the `OnGet` method handle the incoming query and store data temporarily?
??x
The `OnGet` method extracts a `Product` from the query parameters and stores its serialized version in `TempData`. It also sets `ReturnPage`, retrieves or creates `ProductId`, and assigns these values to `TempData`.

```csharp
public void OnGet([FromQuery(Name = "Product")] Product product, string returnPage)
{
    TempData["product"] = Serialize(product);
    TempData["returnAction"] = ReturnPage = returnPage;
    TempData["productId"] = ProductId = product.ProductId.ToString();
}
```
x??

---
#### Supplier Creation and Product Update
Background context: This section describes the process of creating a new `Supplier` and updating an existing `Product` with the newly created supplier's ID.

:p How does the `OnPostAsync` method handle the POST request to create a new supplier?
??x
The `OnPostAsync` method first checks if the model state is valid and if the `Supplier` variable is not null. If these conditions are met, it adds the supplier to the database context and saves changes asynchronously. It then deserializes the stored product data from `TempData`, sets its `SupplierId`, and redirects back to the original page with the updated product.

```csharp
public async Task<IActionResult> OnPostAsync()
{
    if (ModelState.IsValid && Supplier != null)
    {
        context.Suppliers.Add(Supplier);
        await context.SaveChangesAsync();
        
        Product? product = Deserialize(TempData["product"] as string);
        if (product != null)
        {
            product.SupplierId = Supplier.SupplierId;
            TempData["product"] = Serialize(product);
            string? id = TempData["productId"] as string;
            return RedirectToPage(TempData["returnAction"] as string, new { id = id });
        }
    }
    return Page();
}
```
x??

---
#### Partial View for Product Editor
Background context: This section explains the modifications made to a partial view (`_ProductEditor`) to include elements that allow creating or selecting suppliers.

:p What additional elements are added in `_ProductEditor` to handle supplier creation?
??x
In `_ProductEditor`, two new elements were added:
1. A hidden input element to capture the page to return to.
2. A button element to submit form data using a GET request to `SupplierBreakOut`.

```html
<div class="form-group">
    <label asp-for="Product.SupplierId">
        Supplier
        @if (Model?.ReadOnly == false) {
            <input type="hidden" name="returnPage" value="@Model?.Action" />
            <button class="btn btn-sm btn-outline-primary ml-3 my-1"
                    asp-page="SupplierBreakOut" formmethod="get" formnovalidate>
                Create New Supplier
            </button>
        }
    </label>
    <div>
        <span asp-validation-for="Product.SupplierId" class="text-danger"></span>
    </div>
    <select asp-for="Product.SupplierId" class="form-control"
            disabled="@Model?.ReadOnly" 
            asp-items="@(new SelectList(Model?.Suppliers, "SupplierId", "Name"))">
        <option value="" disabled selected>Choose a Supplier</option>
    </select>
</div>
```
x??

---
#### Retrieving Temp Data on Create Page
Background context: This section describes how to use `TempData` in the `Create.cshtml` file to retrieve and populate the product form.

:p How does the `CreateModel` class handle retrieving data from `TempData`?
??x
The `CreateModel` class checks if the "product" key exists in `TempData`. If it does, it retrieves the serialized product data. Otherwise, it initializes a new `Product`.

```csharp
public void OnGet()
{
    Product p = TempData.ContainsKey("product")
        ? Deserialize(TempData["product"] as string)
        : new Product();
}
```
x??

---

#### Retrieving Data for Creating and Editing Products
Background context: The provided code snippets show how to handle creating and editing `Product` entities using ASP.NET Core Razor Pages. The key points are:
- Handling initial data retrieval from `TempData`.
- Using a factory method (`ViewModelFactory`) to create view models.
- Performing validation and saving changes in the database.

The code also demonstrates handling related entities such as `Categories` and `Suppliers`.

:p How does the code retrieve product data for creating or editing?
??x
To retrieve the initial product data, the code checks if a serialized product is present in `TempData`. If not, it fetches the product from the database. This approach ensures that the form can be initialized with existing product details or create new ones.

```csharp
Product? p = TempData.ContainsKey("product")
    ? JsonSerializer.Deserialize<Product>((TempData["product"] as string))
    : await this.DataContext.Products.FindAsync(id);
```
x??

---
#### ViewModel Factory for Product Editing
Background context: The `ViewModelFactory` is used to initialize view models with either existing product data or default values when creating a new product.

:p How does the `ViewModelFactory.Create` method handle both creation and editing of products?
??x
The `ViewModelFactory.Create` method initializes a `ProductViewModel` based on whether it's for creating or editing. For creating, it initializes an empty product; for editing, it uses the provided `Product`.

```csharp
public class ViewModelFactory {
    public static ProductViewModel Create(Product? p, List<Category> categories, List<Supplier> suppliers) {
        if (p != null) {
            // Edit mode: initialize with existing data
            return new ProductViewModel(p.Name, p.Description, ...);
        } else {
            // Create mode: use default values
            return new ProductViewModel(string.Empty, string.Empty, ...);
        }
    }

    public static ProductViewModel Edit(Product? p, List<Category> categories, List<Supplier> suppliers) {
        if (p != null) {
            // Edit mode: initialize with existing data
            return new ProductViewModel(p.Name, p.Description, ...);
        } else {
            // Create mode: use default values
            return new ProductViewModel(string.Empty, string.Empty, ...);
        }
    }
}
```
x??

---
#### Handling Form Data for Creating and Editing Products
Background context: The `OnPostAsync` method handles form submission for both creating and editing products. It checks model state validity before saving changes to the database.

:p What does the code do when a product is submitted via a POST request?
??x
When a product is submitted, the code first checks if the model state is valid. If valid, it sets default values for `ProductId`, `Category`, and `Supplier` (to ensure they are not null). Then, it adds or updates the product in the database and saves changes.

```csharp
if (ModelState.IsValid) {
    product.ProductId = default;
    product.Category = default;
    product.Supplier = default;

    if (p == null) {
        DataContext.Products.Add(product);
    } else {
        DataContext.Products.Update(product);
    }

    await DataContext.SaveChangesAsync();
}
```
x??

---
#### Adding a Supplier during Product Creation
Background context: The code allows users to create new suppliers while creating products. If the user creates a supplier, they are redirected back to the product creation page with the newly created supplier selected.

:p How does the code handle adding a new supplier when creating a product?
??x
When creating a product and a new supplier is added, the code stores the supplier in the database, redirects to the product form, and pre-fills the supplier select element with the newly created supplier's ID.

```csharp
if (ModelState.IsValid) {
    await CheckNewCategory(product);
    if (product.Supplier != default && product.Supplier.Id == Guid.Empty) {
        // Supplier is a new one, create it
        var newSupplier = new Supplier { /* set properties */ };
        await DataContext.Suppliers.AddAsync(newSupplier);
        await DataContext.SaveChangesAsync();
        product.SupplierId = newSupplier.Id;
    }
    DataContext.Products.Add(product);
    await DataContext.SaveChangesAsync();
}
```
x??

---
#### Redirecting to Product Index after Saving
Background context: After saving the product, the application redirects the user back to the product index page.

:p What is the redirection mechanism used when a product is saved?
??x
After successfully saving the product, the code redirects the user to the product index page using `RedirectToPage`.

```csharp
return RedirectToPage(nameof(Index));
```
x??

---

#### Creating an ASP.NET Core Project
Background context: This section explains how to create a new ASP.NET Core project using the `dotnet` command-line tool. The purpose is to set up a basic structure for developing web applications with .NET Core.

:p What are the steps involved in creating an ASP.NET Core project?
??x
The steps involve using the `dotnet new` and `dotnet sln` commands to initialize a new project and solution, respectively. Here’s how it is done:
```bash
# Step 1: Create global.json for .NET SDK version
dotnet new globaljson --sdk-version 7.0.100 --output Advanced

# Step 2: Create the ASP.NET Core web application
dotnet new web --no-https --output Advanced --framework net7.0

# Step 3: Initialize a solution file
dotnet new sln -o Advanced

# Step 4: Add the project to the solution
dotnet sln Advanced add Advanced
```
x??

---
#### Creating a Solution File
Background context: A solution file in .NET Core is used to manage multiple projects within a single development environment. The `dotnet new sln` command creates this file.

:p How do you create a solution file for an ASP.NET Core project?
??x
You use the `dotnet new sln` command followed by the name of the output directory where the solution file will be created.
```bash
dotnet new sln -o Advanced
```
This command creates a `.sln` file in the specified directory, which can now hold multiple projects.

x??

---
#### Adding Project to Solution
Background context: Once you have initialized the solution with `dotnet new sln`, you need to add your project files to this solution. This is done using the `dotnet sln` command.

:p How do you add a project to an existing .NET Core solution?
??x
You use the `dotnet sln` command to include the project in the solution file.
```bash
dotnet sln Advanced add Advanced
```
This command adds the specified project (in this case, `Advanced`) to the `.sln` file.

x??

---
#### Configuring HTTP Port and Launch Settings
Background context: The `launchSettings.json` file controls how your application is launched during debugging. It specifies settings like the port number where the app will run, enabling/disabling browser launching, etc.

:p What changes are recommended for the `launchSettings.json` file?
??x
You should change the HTTP port to a different value and disable automatic browser launch. Here’s an example of how you can do this in `launchSettings.json`:
```json
{
  "profiles": {
    "WebApp": {
      "commandName": "Project",
      "dotnetRunMessages": true,
      "launchBrowser": false,  // Disable browser launching
      "applicationUrl": "http://localhost:5001"  // Change the port number
    }
  }
}
```
x??

---
#### Open Solution in Visual Studio/Visual Studio Code
Background context: After setting up your project and solution, you need to open them in either Visual Studio or Visual Studio Code. This allows for easier development and debugging.

:p How do you open an ASP.NET Core project in Visual Studio or Visual Studio Code?
??x
In Visual Studio:
1. Open the `Advanced.sln` file located in the `Advanced` folder.
Alternatively, using Visual Studio Code:
1. Open the `Advanced` folder containing your project files.

x??

---
#### Setting Environment Variables for Development
Background context: The `ASPNETCORE_ENVIRONMENT` variable is used to define the environment (e.g., development, production) in which the application runs. This affects how certain features and settings are configured.

:p What is the purpose of setting the `ASPNETCORE_ENVIRONMENT` variable?
??x
Setting the `ASPNETCORE_ENVIRONMENT` variable to "Development" ensures that the application runs with development-specific configurations, such as enabling detailed error messages and hot module replacement. This can be done in the `launchSettings.json` file under the environmentVariables section:
```json
{
  "profiles": {
    "WebApp": {
      "environmentVariables": {
        "ASPNETCORE_ENVIRONMENT": "Development"
      }
    }
  }
}
```
x??

---

