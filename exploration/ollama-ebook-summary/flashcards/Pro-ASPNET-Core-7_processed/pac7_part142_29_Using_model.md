# Flashcards: Pro-ASPNET-Core-7_processed (Part 142)

**Starting Chapter:** 29 Using model validation

---

---
#### Model Validation Introduction
Background context explaining model validation and its importance. ASP.NET Core supports extensive model validation features to ensure data integrity and provide useful feedback to users.

:p What is model validation?
??x
Model validation ensures that the data received from a user is suitable for use in an application. It checks if the provided data can be bound to a model and, if not, it provides clear feedback to the user about how to correct any issues.
x??

---
#### Purpose of Model Validation
Explains why model validation is essential. Unvalidated data can lead to unexpected errors and poor user experiences.

:p Why is model validation useful?
??x
Using unvalidated data can result in application states that are odd or undesirable, leading to errors. Model validation helps prevent these issues by ensuring the data is correct before it's used in the application. Additionally, providing clear feedback to users allows them to correct their input and improve the overall user experience.
x??

---
#### Validation Process
Describes the two main parts of the model validation process: checking the data received and providing feedback to the user.

:p What are the key components of the model validation process?
??x
The model validation process consists of:
1. Checking that the data provided in a request is valid for use.
2. Providing useful information to the user if the data is not valid, helping them correct any issues.
x??

---
#### Controller and Razor Page Validation
Explains how controllers and Razor Pages can check the outcome of the validation process.

:p How do controllers and Razor Pages handle validation?
??x
Controllers and Razor Pages use `ModelState` features or validation attributes to ensure that data is validated before it's used. If validation fails, tag helpers are used in views to display error messages to the user.
x??

---
#### Tag Helpers for Validation Feedback
Describes how tag helpers can be used to include validation feedback in views.

:p How do tag helpers help with validation?
??x
Tag helpers are used in Razor Pages and Views to display validation errors directly on form elements. This helps users understand what went wrong and how to correct their input.
x??

---
#### Client-Side Validation
Explains the role of client-side validation and remote validation.

:p What is client-side and remote validation?
??x
Client-side validation checks data in the browser before sending it to the server, providing instant feedback to the user. Remote validation checks data against a database or external service, ensuring that data meets specific criteria on both client and server sides.
x??

---
#### Summary of Key Points
Summarizes the main points covered in the chapter.

:p What are the key takeaways from this chapter?
??x
Key takeaways include understanding model validation, using `ModelState` features and attributes to validate data, displaying validation errors to users, performing custom validations if needed, and utilizing both client-side and remote validation techniques.
x??

---

#### Changing Form.cshtml View
Background context: This section describes how to modify the `Form` view file (`Form.cshtml`) within the `Views/Form` folder to include input fields for each of the properties defined by the `Product` class, excluding navigation properties used by Entity Framework Core.

:p What needs to be modified in the `Form.cshtml` file?
??x
The content of the `Form.cshtml` file should be updated to include input elements for the non-navigation properties of the `Product` class. Specifically, it should contain fields for `Name`, `Price`, `CategoryId`, and `SupplierId`.

```csharp
@model Product

@{
    Layout = "_SimpleLayout";
}

<h5 class="bg-primary text-white text-center p-2">HTML Form</h5>
<form asp-action="submitform" method="post" id="htmlform">
    <div class="form-group">
        <label asp-for="Name"></label>
        <input class="form-control" asp-for="Name" />
    </div>
    <div class="form-group">
        <label asp-for="Price"></label>
        <input class="form-control" asp-for="Price" />
    </div>
    <div class="form-group">
        <label>CategoryId</label>
        <input class="form-control" asp-for="CategoryId"  />
    </div>
    <div class="form-group">
        <label>SupplierId</label>
        <input class="form-control" asp-for="SupplierId"  />
    </div>
    <button type="submit" class="btn btn-primary mt-2">Submit</button>
</form>
```
x??

---

#### Updating FormController.cs
Background context: This section describes how to update the `FormController` file (`FormController.cs`) to include support for displaying the properties defined in the modified `Form.cshtml` and remove unnecessary model binding attributes and action methods.

:p What needs to be done with the `FormController.cs` file?
??x
The contents of the `FormController.cs` file should be updated to handle the form submission for the `Product` class. This involves removing any unnecessary actions or model binding attributes and ensuring that the controller can properly handle the form data.

```csharp
using Microsoft.AspNetCore.Mvc;
using WebApp.Models;
using Microsoft.EntityFrameworkCore;

namespace WebApp.Controllers {
    [AutoValidateAntiforgeryToken]
    public class FormController : Controller {
        private readonly ApplicationDbContext _context; // Assuming this is your DbContext

        public FormController(ApplicationDbContext context) {
            _context = context;
        }

        public IActionResult Index() {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> SubmitForm(Product product) {
            if (ModelState.IsValid) {
                await _context.Products.AddAsync(product);
                await _context.SaveChangesAsync();
                return RedirectToAction("Index");
            }
            return View(product);
        }
    }
}
```
x??

---

#### Explanation of the FormController.cs Changes
Background context: This section explains the specific changes made to the `FormController` file, including the addition of an action method for handling form submissions and ensuring that only valid model data is processed.

:p How does the updated `FormController` handle form submission?
??x
The updated `FormController` includes a new action method `[HttpPost] SubmitForm(Product product)` which processes the form data. This method checks if the submitted model state (`ModelState`) is valid, and if so, it adds the new `Product` instance to the database context and saves the changes.

```csharp
[HttpPost]
public async Task<IActionResult> SubmitForm(Product product) {
    if (ModelState.IsValid) {
        await _context.Products.AddAsync(product);
        await _context.SaveChangesAsync();
        return RedirectToAction("Index");
    }
    return View(product);
}
```
The `SubmitForm` method ensures that only valid data is processed and stored in the database, improving the reliability of the application.

x??

---

#### FormController Class Overview
Background context: The `FormController` class is part of a web application and handles the logic for displaying forms, submitting data, and showing results. It uses dependency injection to obtain an instance of `DataContext`, which represents the database context.

:p What does the `FormController` class do?
??x
The `FormController` class manages form operations by providing actions to display a form, submit form data, and show the submitted data. It includes methods for displaying the form based on a given ID or ordering products by their product IDs.
```csharp
public class FormController : Controller {
    private DataContext context;

    public FormController(DataContext dbContext) {
        context = dbContext;
    }

    public async Task<IActionResult> Index(long? id) { 
        // This method returns the form view with an optional product based on ID or ordered products.
        return View("Form", await context.Products
            .OrderBy(p => p.ProductId)
            .FirstOrDefaultAsync(p => id == null || p.ProductId == id));
    }

    [HttpPost]
    public IActionResult SubmitForm(Product product) {
        // This method saves the submitted form data to TempData and redirects to the Results view.
        TempData["name"] = product.Name;
        TempData["price"] = product.Price.ToString();
        TempData["categoryId"] = product.CategoryId.ToString();
        TempData["supplierId"] = product.SupplierId.ToString();
        return RedirectToAction(nameof(Results));
    }

    public IActionResult Results() {
        // This method returns the results view with the data from TempData.
        return View(TempData);
    }
}
```
x??

---

#### Database Context and Dependency Injection
Background context: The `FormController` uses dependency injection to get an instance of `DataContext`, which represents the database context. This allows for better separation of concerns, making the code more testable.

:p How does the FormController use dependency injection?
??x
The `FormController` uses constructor injection to receive an instance of `DataContext`. This pattern promotes better testability and maintainability by decoupling the controller from concrete implementations.
```csharp
public class FormController : Controller {
    private DataContext context;

    public FormController(DataContext dbContext) { 
        context = dbContext;
    }
}
```
x??

---

#### Dropping the Database with EF Core
Background context: To prepare for running the example application, you need to drop an existing database. Entity Framework Core provides a command-line tool to execute this task.

:p How do you use the command-line tool to drop the database?
??x
You can use the `dotnet ef database drop --force` command in the PowerShell window to drop the database. The `--force` option ensures that any existing database is dropped without prompting for confirmation.
```powershell
dotnet ef database drop --force
```
x??

---

#### Running the Example Application
Background context: After dropping the database, you need to run the application using the `dotnet run` command in PowerShell. This will start the web server and make your application accessible.

:p How do you run the example application?
??x
You can use the `dotnet run` command in the PowerShell window to launch the application. This command starts a development server, making your application available at `http://localhost:5000`.
```powershell
dotnet run
```
x??

---

#### Displaying an HTML Form
Background context: The `Index` method of the `FormController` returns an action result that renders a view named "Form" and passes a product object or null based on the given ID.

:p How does the Index method handle displaying the form?
??x
The `Index` method uses dependency injection to get the `DataContext` context. It then queries for products, orders them by their product IDs, and selects the first one that matches the provided ID (or returns null if no match). The result is passed to the "Form" view.
```csharp
public async Task<IActionResult> Index(long? id) { 
    return View("Form", await context.Products
        .OrderBy(p => p.ProductId)
        .FirstOrDefaultAsync(p => id == null || p.ProductId == id));
}
```
x??

---

#### Handling Form Submission and Redirecting to Results
Background context: The `SubmitForm` method handles the form submission by storing the submitted data in `TempData` and then redirecting to the `Results` view.

:p How does the SubmitForm method handle the form submission?
??x
The `SubmitForm` method is a POST action that receives a `Product` object from the form. It stores key properties of the product in `TempData`, which are later used by the `Results` view to display the submitted data.
```csharp
[HttpPost]
public IActionResult SubmitForm(Product product) {
    TempData["name"] = product.Name;
    TempData["price"] = product.Price.ToString();
    TempData["categoryId"] = product.CategoryId.ToString();
    TempData["supplierId"] = product.SupplierId.ToString();
    return RedirectToAction(nameof(Results));
}
```
x??

---

#### Viewing Submitted Form Data
Background context: The `Results` method uses the data stored in `TempData` to render a view that displays the submitted form data.

:p How does the Results method display the submitted data?
??x
The `Results` method retrieves the data from `TempData` and passes it directly to the "Results" view. This view is responsible for displaying the saved information.
```csharp
public IActionResult Results() {
    return View(TempData);
}
```
x??

---

