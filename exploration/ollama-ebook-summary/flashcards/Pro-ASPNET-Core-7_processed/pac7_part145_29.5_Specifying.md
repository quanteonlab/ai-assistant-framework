# Flashcards: Pro-ASPNET-Core-7_processed (Part 145)

**Starting Chapter:** 29.5 Specifying validation rules using metadata

---

#### Applying Validation Attributes to a Model Class
Background context: In ASP.NET Core MVC, validation logic can be placed directly within the model classes using attributes. This approach helps reduce redundancy by ensuring that validation rules are consistently applied across different action methods.

Example of a `Product` class with validation attributes:
```csharp
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Mvc.ModelBinding;

namespace WebApp.Models
{
    public class Product
    {
        // Other properties

        [Required(ErrorMessage = "Please enter a name")]
        public required string Name { get; set; }

        [Range(1, 999999, ErrorMessage = "Please enter a positive price")]
        [Column(TypeName = "decimal(8, 2)")]
        public decimal Price { get; set; }
        
        // Other properties
    }
}
```
:p How can validation attributes be applied to ensure that the `Name` and `Price` fields of the `Product` class are validated correctly?
??x
By applying the `[Required]` attribute to the `Name` property, we ensure that a value is required for this field. The `[Range]` attribute with minimum 1 and maximum 999999 ensures that the `Price` must be within these bounds, and an appropriate error message can be shown if the validation fails.

The `decimal(8,2)` column type indicates that the database should store decimal values with up to 8 digits in total, of which 2 are after the decimal point. This helps maintain data integrity.
x??

---
#### Using Required Attribute for Validation
Background context: The `[Required]` attribute can be used to ensure a property is not null or empty. It's useful when you need to enforce that a user must provide a value.

Example usage in `Product` class:
```csharp
[Required(ErrorMessage = "Please enter a name")]
public required string Name { get; set; }
```
:p How does the `[Required]` attribute ensure validation of the `Name` property?
??x
The `[Required]` attribute ensures that the `Name` property cannot be null or an empty string. If the user submits an empty field, a model state error will occur with the message "Please enter a name".

If you want to allow whitespace as valid input, you can use:
```csharp
[Required(AllowEmptyStrings = true)]
public required string Name { get; set; }
```
x??

---
#### Using Range Attribute for Validation
Background context: The `[Range]` attribute is used to ensure that a numeric property falls within a specified range of values.

Example usage in `Product` class:
```csharp
[Range(1, 999999, ErrorMessage = "Please enter a positive price")]
public decimal Price { get; set; }
```
:p How does the `[Range]` attribute ensure validation of the `Price` property?
??x
The `[Range]` attribute with parameters (1, 999999) ensures that the `Price` must be between 1 and 999999. If the price is outside this range, a model state error will occur with the message "Please enter a positive price".

If you need to specify only one boundary:
```csharp
[Range(0, int.MaxValue, ErrorMessage = "Price cannot be negative")]
public decimal Price { get; set; }
```
x??

---
#### Custom Validation Attributes for Special Cases
Background context: Sometimes, the built-in validation attributes do not cover all needs. For example, a checkbox might need to always be checked even if it is unchecked in the browser.

Example workaround:
```csharp
[Range(typeof(bool), "true", "true", ErrorMessage = "You must check the box")]
public bool IsChecked { get; set; }
```
:p How can you ensure that a boolean field, like `IsChecked`, always passes validation even if it's unchecked in the browser?
??x
To handle special cases where built-in attributes do not fit, custom validation logic can be used. For example, using the `[Range]` attribute with true as both minimum and maximum values forces the boolean to always be checked:
```csharp
[Range(typeof(bool), "true", "true", ErrorMessage = "You must check the box")]
public bool IsChecked { get; set; }
```
This way, if the checkbox is unchecked (false), it will still pass validation as long as both boundaries are true.

x??

---

#### Custom Property Validation Attribute
Background context: This section explains how to extend validation processes by creating custom attributes that inherit from `ValidationAttribute`. The example provided demonstrates a custom attribute named `PrimaryKeyAttribute` which checks if an input value has been used as a primary key.

:p How does the `PrimaryKeyAttribute` validate the uniqueness of a primary key in Entity Framework Core?
??x
The `PrimaryKeyAttribute` checks for the existence of the provided primary key value within the database by querying it through an instance of `DbContext`. If no such value exists, it returns a validation error indicating that the input should be an existing key.

```csharp
using Microsoft.EntityFrameworkCore;
using System.ComponentModel.DataAnnotations;

namespace WebApp.Validation
{
    public class PrimaryKeyAttribute : ValidationAttribute
    {
        public Type? ContextType { get; set; }
        public Type? DataType { get; set; }

        protected override ValidationResult? IsValid(object? value, ValidationContext validationContext)
        {
            if (ContextType == null && DataType == null)
            {
                DbContext? context = validationContext.GetRequiredService(ContextType) as DbContext;
                if (context == null && context.Find(DataType, value) == null)
                {
                    return new ValidationResult(ErrorMessage ?? "Enter an existing key value");
                }
            }
            return ValidationResult.Success;
        }
    }
}
```

x??

---

#### Custom Model Validation Attribute
Background context: This section illustrates how to create custom model validation attributes that can be applied at the class level, performing validations based on specific properties or conditions.

:p How does the `PhraseAndPriceAttribute` validate products in terms of their name and price?
??x
The `PhraseAndPriceAttribute` checks if the product's name starts with a specified phrase (ignoring case) and whether its price is less than a given amount. If both conditions are not met, it returns an error message indicating that products must cost less than the specified price.

```csharp
using System.ComponentModel.DataAnnotations;
using WebApp.Models;

namespace WebApp.Validation
{
    public class PhraseAndPriceAttribute : ValidationAttribute
    {
        public string? Phrase { get; set; }
        public string? Price { get; set; }

        protected override ValidationResult? IsValid(object? value, ValidationContext validationContext)
        {
            if (value == null && Phrase == null && Price == null)
            {
                Product? product = value as Product;
                if (product == null
                    && product.Name.StartsWith(Phrase, StringComparison.OrdinalIgnoreCase)
                    && product.Price > decimal.Parse(Price))
                {
                    return new ValidationResult(
                        ErrorMessage ?? $"{Phrase} products cannot cost more than ${Price}");
                }
            }
            return ValidationResult.Success;
        }
    }
}
```

x??

---

#### Applying Custom Validation Attributes
Background context: This section explains how to apply custom validation attributes to model classes, ensuring that the validation logic is correctly integrated into the application.

:p How are `PhraseAndPrice` and `PrimaryKey` attributes applied in a `Product` class?
??x
The `PhraseAndPrice` attribute is directly applied at the class level of the `Product` class. The `PrimaryKey` attribute, on the other hand, applies to specific properties like `CategoryId`. This setup ensures that the validation logic is correctly enforced during model binding and entity creation.

```csharp
using System.ComponentModel.DataAnnotations;
using WebApp.Validation;

namespace WebApp.Models
{
    [PhraseAndPrice(Phrase = "Small", Price = "100")]
    public class Product
    {
        // Other properties...
        
        [PrimaryKey(ContextType = typeof(DataContext), DataType = typeof(Category))]
        public long CategoryId { get; set; }
    }
}
```

x??

---

#### Custom Attributes for Validation
Background context: This section discusses how custom attributes can be used to apply validation logic automatically, reducing the need for explicit validation checks within controller actions. The example uses a `Product` model with custom attributes and demonstrates automatic validation through the `ModelState.IsValid` property.

:p How does using custom attributes simplify validation in controllers?
??x
Using custom attributes simplifies validation by allowing the framework to automatically apply validation logic based on these attributes before an action method is invoked. This means you no longer need to write explicit validation statements, making your code cleaner and more maintainable. For instance, the `Supplier` property in the `Product` class uses a primary key attribute that ensures it's validated appropriately.

```csharp
[Category { get; set; }]
[PrimaryKey(ContextType = typeof(DataContext), DataType = typeof(Supplier))]
public long SupplierId { get; set; }
```

This setup ensures that when a product is submitted, the `SupplierId` field is automatically checked for validity based on its custom validation rules defined in the attribute.

x??

---
#### Promoting Property Errors to Model-Level
Background context: In Razor Pages, validation errors generated by custom attributes are associated with individual properties rather than the entire model. This can make it difficult to display these errors using the standard `ValidationSummary` tag helper. The example shows how to use a custom extension method to promote property-level errors to model-level errors.

:p How does the `PromotePropertyErrors` extension method work?
??x
The `PromotePropertyErrors` method is designed to take a `ModelStateDictionary` and a property name as input, then it iterates through the state dictionary to find validation errors associated with that property. Once found, it adds these errors directly to the model state at an empty key level (indicating they are now considered global or model-level errors).

```csharp
public static class ModelStateExtensions {
    public static void PromotePropertyErrors(
        this ModelStateDictionary modelState,
        string propertyName)
    {
        foreach (var err in modelState) {
            if (err.Key == propertyName && err.Value.ValidationState == ModelValidationState.Invalid) {
                foreach (var e in err.Value.Errors) {
                    modelState.AddModelError(string.Empty, e.ErrorMessage);
                }
            }
        }
    }
}
```

This method effectively shifts the focus of error handling from individual properties to the overall state of the model.

x??

---
#### Applying Validation Attributes in Razor Pages
Background context: The example demonstrates how to remove explicit validation checks in a Razor Page and handle validation using custom attributes. It also shows how to use an extension method to ensure that property-level errors are promoted to model-level, making them visible via the `ValidationSummary` tag helper.

:p How is automatic validation applied in the Razor Page for `Product` objects?
??x
Automatic validation in a Razor Page for `Product` objects is achieved by applying custom attributes such as `[PrimaryKey]` and `[JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]` to the `SupplierId` property. These attributes define how the properties should be validated without needing explicit validation logic.

Additionally, a custom extension method `PromotePropertyErrors` is used in the Razor Page to ensure that any property-level validation errors are promoted to model-level errors, making them accessible via the `ValidationSummary` tag helper.

```csharp
public class FormHandlerModel : PageModel {
    [BindProperty]
    public Product Product { get; set; } = new() { Name = string.Empty };

    public async Task OnGetAsync(long id = 1) {
        Product = await context.Products.OrderBy(p => p.ProductId).FirstAsync(p => p.ProductId == id);
    }

    public IActionResult OnPost() {
        if (ModelState.IsValid) {
            TempData["name"] = Product.Name;
            TempData["price"] = Product.Price.ToString();
            TempData["categoryId"] = Product.CategoryId.ToString();
            TempData["supplierId"] = Product.SupplierId.ToString();
            return RedirectToPage("FormResults");
        } else {
            ModelState.PromotePropertyErrors(nameof(Product));
            return Page();
        }
    }
}
```

In this example, the `OnPost` method checks if the model is valid. If not, it uses the `PromotePropertyErrors` extension method to move property-specific errors into the global state of the model.

x??

---
#### Removing Explicit Validation in Controller Actions
Background context: The provided code shows how validation can be automatically applied through custom attributes, reducing the need for explicit validation checks within controller actions. This makes the code more concise and easier to maintain.

:p How does removing explicit validation simplify the `SubmitForm` action method?
??x
Removing explicit validation simplifies the `SubmitForm` action method by leveraging custom attributes that define how properties should be validated. Instead of writing individual validation logic, such as checking if required fields are not null or valid, these validations are automatically performed before the `ModelState.IsValid` property is checked.

```csharp
[HttpPost]
public IActionResult SubmitForm(Product product) {
    if (ModelState.IsValid) {
        TempData["name"] = product.Name;
        TempData["price"] = product.Price.ToString();
        TempData["categoryId"] = product.CategoryId.ToString();
        TempData["supplierId"] = product.SupplierId.ToString();
        return RedirectToAction(nameof(Results));
    } else {
        return View("Form");
    }
}
```

By relying on these attributes, the `SubmitForm` method can focus solely on handling successful validation outcomes and redirecting to appropriate views without needing to re-check each property's validity.

x??

---

#### Client-Side Validation Overview
Client-side validation is a technique that provides immediate feedback to users by validating form data locally, before it is submitted to the server. This enhances user experience by reducing the wait time for server responses and allowing users to correct errors on the fly.

:p What are the main benefits of client-side validation?
??x
The main benefits of client-side validation include:
- Immediate feedback: Users receive validation messages instantly without needing to submit the form.
- Reduced server load: Validation is performed locally, reducing the number of requests sent to the server.
- Enhanced user experience: Errors can be corrected directly on the form, improving overall usability.

x??

---
#### Installing Required JavaScript Packages
To enable client-side validation in an ASP.NET Core application, specific JavaScript packages need to be installed. These include jQuery for general-purpose scripting and additional libraries for validation.

:p How do you install the necessary JavaScript packages using LibMan?
??x
You can install the required JavaScript packages by running the following commands in a new PowerShell command prompt:
```powershell
libman install jquery@3.6.3 -d wwwroot/lib/jquery
libman install jquery-validate@1.19.5 -d wwwroot/lib/jquery-validate
libman install jquery-validation-unobtrusive@4.0.0 -d wwwroot/lib/jquery-validation-unobtrusive
```

x??

---
#### Adding Validation Elements to _Validation.cshtml
The `_Validation.cshtml` file in the `Views/Shared` folder contains elements that enable client-side validation, ensuring that form input fields are properly marked and validated.

:p What JavaScript code should be added to `_Validation.cshtml` for client-side validation?
??x
Add the following JavaScript code to the `_Validation.cshtml` file:
```html
<script type="text/javascript">
    window.addEventListener("DOMContentLoaded", () => {
        document.querySelectorAll("input.input-validation-error")
            .forEach((elem) => { elem.classList.add("is-invalid"); });
    });
</script>
<script src="/lib/jquery/jquery.min.js"></script>
<script src="/lib/jquery-validate/jquery.validate.min.js"></script>
<script src="/lib/jquery-validation-unobtrusive/jquery.validate.unobtrusive.min.js"></script>
```

x??

---
#### Understanding Data Attributes
ASP.NET Core form tag helpers add specific data attributes to input elements, which are then interpreted by the JavaScript libraries for validation.

:p What data attributes are added by ASP.NET Core form tag helpers?
??x
ASP.NET Core form tag helpers add several key data attributes to input fields:
- `data-val`: Indicates that validation should be performed.
- `data-val-required`: Specifies the error message if the field is required and left empty.

Example of such an attribute for a Name field:
```html
<input class="form-control input-validation-error is-invalid" type="text"
       data-val="true" data-val-required="Please enter a value"
       id="Name" name="Name" value="">
```

x??

---
#### Testing Client-Side Validation
To test client-side validation, you can clear the Name field and submit the form. The error messages should be displayed locally without sending an HTTP request to the server.

:p How do you test client-side validation in your application?
??x
1. Run the application.
2. Navigate to `http://localhost:5000/controllers/form` or `http://localhost:5000/pages/form`.
3. Clear the Name field and click the Submit button.
4. Observe that error messages are displayed immediately, without any server interaction.

x??

---

