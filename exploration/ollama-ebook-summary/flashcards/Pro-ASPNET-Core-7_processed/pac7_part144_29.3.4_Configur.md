# Flashcards: Pro-ASPNET-Core-7_processed (Part 144)

**Starting Chapter:** 29.3.4 Configuring the default validation error messages

---

#### Explicit Validation Process Using Entity Framework Core
Background context: The process involves using Entity Framework Core to ensure that user-provided IDs (CategoryId and SupplierId) correspond to actual IDs stored in the database. This is done through explicit validation checks, and any errors are reported via `ModelState.IsValid`.
:p What is the process for validating CategoryId and SupplierId properties using Entity Framework Core?
??x
The process involves querying the database to check if the provided CategoryId or SupplierId exists. If not, a validation error will be added to `ModelState`, indicating that the value does not correspond to an ID stored in the database.

```csharp
public class ProductValidator : AbstractValidator<Product>
{
    private readonly ApplicationDbContext _context;

    public ProductValidator(ApplicationDbContext context)
    {
        _context = context;
        
        RuleFor(p => p.CategoryId).MustAsync(async (id, cancellationToken) =>
            await _context.Categories.AnyAsync(c => c.Id == id))
            .WithMessage("Invalid Category ID.");
        
        RuleFor(p => p.SupplierId).MustAsync(async (id, cancellationToken) =>
            await _context.Suppliers.AnyAsync(s => s.Id == id))
            .WithMessage("Invalid Supplier ID.");
    }
}
```
x??

---

#### Using ModelState.IsValid for Validation
Background context: After performing explicit validation checks on properties like CategoryId and SupplierId, the `ModelState.IsValid` property is used to check if there were any errors during the validation process. If there are no validation errors, then `ModelState.IsValid` will be true; otherwise, it will return false.
:p How does `ModelState.IsValid` help in determining whether the form data is valid?
??x
`ModelState.IsValid` helps by checking the validity of all model properties after performing explicit or implicit validation checks. If any property fails validation (e.g., due to missing values, incorrect types, etc.), `ModelState.IsValid` will be false.

```csharp
if (!modelState.IsValid)
{
    return View(model);
}
```
x??

---

#### Default Validation Error Messages in ASP.NET Core
Background context: By default, the validation process can produce messages that are not always user-friendly. For example, if a required field is empty, it might show "The value '' is invalid." This inconsistency can be resolved by configuring custom error messages using the `DefaultModelBindingMessageProvider` class.
:p How can you replace default validation messages in ASP.NET Core?
??x
You can replace default validation messages by configuring them through the options pattern in your `Program.cs` file. Specifically, you can use methods like `SetValueMustNotBeNullAccessor` to customize error messages.

```csharp
builder.Services.Configure<MvcOptions>(options =>
{
    options.ModelBindingMessageProvider.SetValueMustNotBeNullAccessor =
        value => "Please enter a value";
});
```
x??

---

#### Customizing Validation Error Messages for Decimal Fields
Background context: When dealing with decimal fields, the default validation error messages can be less helpful. For example, if a required decimal field is left empty or contains invalid input, it might show confusing messages. You can customize these messages to make them more user-friendly.
:p How do you customize validation error messages for decimal properties in ASP.NET Core?
??x
You can customize validation error messages by configuring the `DefaultModelBindingMessageProvider` methods specifically for decimal fields.

```csharp
builder.Services.Configure<MvcOptions>(options =>
{
    options.ModelBindingMessageProvider.SetValueIsInvalidAccessor =
        value => "Please enter a valid number";
});
```
x??

---

#### Configuring Validation Error Messages in Program.cs
Background context: In the `Program.cs` file, you can configure default validation error messages to be more user-friendly. This is done using the options pattern and the methods of the `DefaultModelBindingMessageProvider`.
:p How do you change the validation message for a null value in ASP.NET Core?
??x
To change the validation message for a null value, you use the `SetValueMustNotBeNullAccessor` method.

```csharp
builder.Services.Configure<MvcOptions>(options =>
{
    options.ModelBindingMessageProvider.SetValueMustNotBeNullAccessor =
        value => "Please enter a value";
});
```
x??

---

---
#### Custom Error Messages for Model Validation
In ASP.NET Core, by default, validation messages can sometimes be generic and not directly tied to specific fields. The example provided uses a custom error message for missing Name fields, which shows how non-nullable model properties are validated as if they had the `Required` attribute applied.
:p How does ASP.NET Core handle validation messages for non-nullable properties?
??x
By default, ASP.NET Core treats non-nullable properties as required and provides generic validation messages. However, you can customize these messages using the `asp-validation-for` tag helper in your views. For example:
```html
<span asp-validation-for="Name" class="text-danger"></span>
```
This line of code generates a span element that will display the custom error message if there's an issue with the `Name` property.
x?
---
#### Displaying Property-Level Validation Messages
To make validation messages more user-friendly, you can show them next to the corresponding input fields. This is achieved using the `ValidationMessageTag` tag helper and adding it as a sibling element to the input field that requires validation.
:p How can you display custom validation error messages near the respective input elements?
??x
You can use the `asp-validation-for` attribute on a span element within the form group, which will bind to a specific property. Here is an example:
```html
<div class="form-group">
    <label asp-for="Name"></label>
    <input class="form-control" asp-for="Name" />
    <span asp-validation-for="Name" class="text-danger"></span>
</div>
```
This code ensures that if the `Name` property fails validation, a custom error message will be shown next to the input field.
x?
---
#### Displaying Model-Level Validation Messages
Model-level validation can handle errors that arise from combinations of properties. The example provided demonstrates how to add a model-level validation message when the Price exceeds 100 and the Name starts with "Small".
:p How does ASP.NET Core allow you to set up model-level validation?
??x
You can use `ModelState.AddModelError` method in your controller action to add model-level validation errors. Here is an example of how this might look:
```csharp
if (ModelState.GetValidationState(nameof(Product.Name)) == ModelValidationState.Valid &&
    ModelState.GetValidationState(nameof(Product.Price)) == ModelValidationState.Valid &&
    product.Name.ToLower().StartsWith("small") && product.Price > 100)
{
    ModelState.AddModelError("", "Small products cannot cost more than $100");
}
```
This checks if both Name and Price are valid, but the name starts with "Small" and price is greater than 100 before adding a model-level error.
x?
---
#### Configuring Validation Summary for Model Only
The validation summary can be configured to display only messages related to the entire model rather than individual properties. The example uses `asp-validation-summary="ModelOnly"` in the view to exclude property-level errors and show only model-wide issues.
:p How do you configure the validation summary to display only model-level errors?
??x
You set the `asp-validation-summary` attribute on your form to `"ModelOnly"`. This ensures that only messages applicable to the entire model are shown, rather than individual property errors. Here is how it looks in a view:
```html
<div asp-validation-summary="ModelOnly" class="text-danger"></div>
```
This line of code will display any validation issues related to the overall model structure.
x?
---

#### Razor Page Validation
Background context: In a Razor Page application, validation is crucial for ensuring that data entered by users meets certain criteria before it can be saved or processed. The `ModelState` object is used to store and manage these validation errors.

Relevant code snippets:
```csharp
@page "/pages/form/{id:long?}"
@model FormHandlerModel
@functions {
    public class FormHandlerModel : PageModel {
        private DataContext context;
        
        public FormHandlerModel(DataContext dbContext) {
            context = dbContext;
        }

        [BindProperty]
        public Product Product { get; set; } = new() { Name = string.Empty };

        public async Task OnGetAsync(long id = 1) {
            Product = await context.Products
                .OrderBy(p => p.ProductId)
                .FirstAsync(p => p.ProductId == id);
        }

        public IActionResult OnPost() {
            if (ModelState.GetValidationState("Product.Price")
                == ModelValidationState.Valid && Product.Price < 1) {
                ModelState.AddModelError("Product.Price", "Enter a positive price");
            }
            // Other validation checks
            if (ModelState.IsValid) {
                TempData[\"name\"] = Product.Name;
                return RedirectToPage(\"FormResults\");
            } else {
                return Page();
            }
        }
    }
}
```
:p How is explicit data validation performed in a Razor Page?
??x
Explicit data validation in a Razor Page involves checking the state of model properties and adding errors to the `ModelState` object if the data does not meet specified criteria. This ensures that any issues are displayed to the user before form submission.

For example, validating the price field:
```csharp
if (ModelState.GetValidationState("Product.Price") 
    == ModelValidationState.Valid && Product.Price < 1) {
    ModelState.AddModelError("Product.Price", "Enter a positive price");
}
```
This code checks if the price is valid but less than $1, and if so, adds an error message to `ModelState`.

Other validation checks include:
- Ensuring the name does not start with 'small' and has a high price.
- Checking for existing category and supplier IDs.

:x??
---
#### Model Binding in Razor Pages
Background context: Model binding is a mechanism that automatically binds form input data to model properties. In Razor Pages, this is managed by attributes like `[BindProperty]`.

Relevant code snippet:
```csharp
[BindProperty]
public Product Product { get; set; } = new() { Name = string.Empty };
```
:p What does the `[BindProperty]` attribute do in a Razor Page?
??x
The `[BindProperty]` attribute binds form input data to model properties automatically. In this case, it ensures that the `Product` object is populated with values from the form when the page is posted.

Example:
```csharp
[BindProperty]
public Product Product { get; set; } = new() { Name = string.Empty };
```
This attribute makes sure that any input named "Name", "Price", etc., in the form is automatically assigned to the corresponding properties of the `Product` object.

:x??
---
#### Displaying Validation Messages
Background context: Validation messages are displayed using the `<span asp-validation-for="..."></span>` and `<div asp-validation-summary="ModelOnly"></div>` tags. These tags help in displaying detailed validation errors next to each field or at the top of the form.

Relevant code snippet:
```html
<div class="m-2">
    <h5 class="bg-primary text-white text-center p-2">HTML Form</h5>
    <form asp-page="/pages/form/{id:long?}" method="post" id="htmlform">
        <div asp-validation-summary="ModelOnly" class="text-danger"></div>
        <!-- Other form fields and validation tags -->
    </form>
</div>
```
:p How do you display model-level validation messages in a Razor Page?
??x
To display model-level validation messages, use the `<div asp-validation-summary="ModelOnly" class="text-danger"></div>` tag. This tag will show summary-level errors for the entire model if any exist.

Additionally, specific field validations can be displayed using `<span asp-validation-for="FieldName" class="text-danger"></span>`, which shows error messages next to each field.

Example:
```html
<div asp-validation-summary="ModelOnly" class="text-danger"></div>
<div class="form-group">
    <label>Name</label>
    <div><span asp-validation-for="Product.Name" class="text-danger"></span></div>
    <input class="form-control" asp-for="Product.Name" />
</div>
```
:x??
---

