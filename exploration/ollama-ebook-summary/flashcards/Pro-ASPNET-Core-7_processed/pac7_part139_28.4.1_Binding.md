# Flashcards: Pro-ASPNET-Core-7_processed (Part 139)

**Starting Chapter:** 28.4.1 Binding to a property

---

---
#### Using BindProperty for Model Binding
Background context: In Razor Pages, when binding to a complex type using parameters, it often duplicates properties defined by the page model class. This can lead to redundancy and less elegant code.

:p How does the `BindProperty` attribute help in avoiding redundant property definitions?
??x
The `BindProperty` attribute is used to indicate that the properties of a class should be subject to the model binding process. By decorating a property with this attribute, you instruct the framework to use the value of this property directly from the form data without needing an explicit parameter.

```csharp
[BindProperty]
public Product Product { get; set; } = new() { Name = string.Empty };
```

In the example above, the `Product` property is decorated with `[BindProperty]`, allowing its properties to be automatically bound when a POST request is made. This means that in the `OnPost` method, you can directly access the `Product` object without needing an additional parameter.

x??
---
#### Handling GET Requests with BindProperty
Background context: By default, the `BindProperty` attribute does not bind data for GET requests. However, this behavior can be changed by setting the `SupportsGet` argument to true.

:p How do you modify the `BindProperty` attribute so that it supports binding during a GET request?
??x
To enable model binding for both GET and POST requests with the `BindProperty` attribute, you need to set the `SupportsGet` parameter to `true`.

```csharp
[BindProperty(SupportsGet = true)]
public Product Product { get; set; } = new() { Name = string.Empty };
```

This configuration allows the `Product` property to be populated with data during a GET request, making it easier to pre-fill form fields in cases where you might want to display an existing entity before allowing edits.

x??
---
#### Excluding Properties from Model Binding
Background context: Sometimes, you may want certain properties to be excluded from model binding. The `BindNever` attribute can be used for this purpose.

:p How do you exclude a property from being bound using model binding in Razor Pages?
??x
To exclude a specific property from being bound by the model binder, use the `BindNever` attribute on that property.

```csharp
[BindNever]
public string ReadOnlyField { get; set; }
```

In this example, the `ReadOnlyField` is marked with `[BindNever]`, meaning its value will not be automatically populated during form submissions. This can be useful for fields like IDs or timestamps that should remain unchanged when a user submits a form.

x??
---

#### Nested Complex Types and Model Binding
Background context: This section explains how to handle nested complex types within a model binding scenario. When properties of a complex type are bound, the model binder uses property names as prefixes for the HTML form elements. This allows for accurate binding of data from forms back into nested objects.

:p How does the model binder handle nested complex types in an HTML form?
??x
The model binder processes nested properties by using the parent object's name as a prefix. For instance, if `Category` is a property within the `Product` class, and you want to bind the `Name` of the `Category`, the input element for this property should have its `name` attribute set as "Category.Name". This ensures that when the form data is submitted, the model binder can correctly map these values back to their respective properties.

Example:
```html
<input class="form-control" name="Category.Name" value="@Model.Category.Name" />
```
x??

---
#### Custom Prefixes for Nested Properties
Background context: Sometimes, you might want to bind form data to a different structure than what is used in your view model. This can lead to mismatches between the form's `name` attributes and the expected property names by the model binder.

:p What happens if the form's name attribute does not match the expected prefix?
??x
If the form's `name` attributes do not match the expected prefixes, the model binder may fail to correctly bind the data. For example, if your view is expecting a `Category` object but the form uses "cat.name" instead of "Category.Name", the model binder will not be able to find and map these values accurately.

Example:
```html
<input class="form-control" asp-for="Category.Name" name="cat.name" />
```
x??

---
#### Using the Bind Attribute for Model Binding Prefixes
Background context: To resolve issues where form data names do not match expected prefixes, you can use the `Bind` attribute with a custom prefix. This allows you to specify which parts of the form data should be bound to your model.

:p How does the `Bind` attribute help in resolving naming mismatches?
??x
The `Bind` attribute is used to explicitly bind specific parts of the form data based on a custom prefix. By setting this attribute, you can ensure that the model binder correctly identifies and binds the data from the form to your model properties.

Example:
```csharp
[HttpPost]
public IActionResult SubmitForm([Bind(Prefix = "cat")] Category category)
{
    // Code for processing the form submission
}
```
x??

---
#### Using BindProperty Attribute for Model Binding Prefixes in Razor Pages
Background context: In Razor Pages, you might want to bind complex objects with nested properties using specific prefixes. The `BindProperty` attribute can be used along with the `Name` argument to specify these custom prefixes.

:p How does the `BindProperty` attribute help in specifying model binding prefixes in Razor Pages?
??x
The `BindProperty` attribute, when combined with the `Name` argument, allows you to bind form data to your model properties using specific names. This is particularly useful when dealing with complex objects and nested properties where default naming conventions might not match.

Example:
```csharp
[BindProperty(Name = "Product.Category")]
public Category Category { get; set; } = new() { Name = string.Empty };
```
x??

---

#### Binding Complex Types
Background context: The text discusses how to handle complex types and selectively bind properties during model binding. This is important for security and ensuring that only certain fields are updated or bound.

:p What does `System.Text.Json.JsonSerializer.Serialize` do in this context?
??x
`System.Text.Json.JsonSerializer.Serialize` converts the `Category` object into a JSON string and stores it in `TempData`. This allows the category information to be passed between pages.
```csharp
TempData["category"] = System.Text.Json.JsonSerializer.Serialize(Category);
```
x??

---
#### Redirecting to Another Page After Binding
Background context: The example shows how to redirect to another page after handling a form submission. TempData is used to store temporary data that can be accessed on the target page.

:p How does `RedirectToPage` work in this scenario?
??x
`RedirectToPage` redirects to the specified page and passes any `TempData` items as query string parameters or as model data if there's a view model. In this case, it redirects to "FormResults" while passing category information.
```csharp
return RedirectToPage("FormResults");
```
x??

---
#### Specifying Model Binding Prefix in Razor Page
Background context: This example demonstrates how to use the `asp-for` attribute with a specific prefix for binding properties.

:p What is the purpose of using `asp-for` with a model binding prefix?
??x
Using `asp-for` with a model binding prefix ensures that only certain properties are bound when the form is submitted. Here, it binds the "Name" and "Category" properties.
```html
<input class="form-control" asp-for="Category.Name" />
```
x??

---
#### Selectively Binding Properties in Controller
Background context: The code shows how to selectively bind specific properties by using the `Bind` attribute with a list of property names.

:p How does the `Bind` attribute help in selectively binding properties?
??x
The `Bind` attribute is used to specify which properties should be included in the model binding process. In this case, it binds only "Name" and "Category", excluding other properties.
```csharp
[HttpPost]
public IActionResult SubmitForm([Bind("Name", "Category")] Product product)
```
x??

---
#### Preventing Over-Binding Attacks
Background context: The example illustrates how to prevent over-binding attacks by selectively binding sensitive properties using the `BindNever` attribute.

:p What is the purpose of the `BindNever` attribute in model classes?
??x
The `BindNever` attribute prevents the model binder from considering certain properties, thus preventing users from setting values for those fields. In this case, it excludes the "Price" property.
```csharp
[Column(TypeName = "decimal(8, 2)")]
[BindNever]
public decimal Price { get; set; }
```
x??

---
#### Decorating Properties with BindNever
Background context: This example shows how to apply the `BindNever` attribute directly on properties in model classes.

:p How does applying `BindNever` affect property binding?
??x
Applying `BindNever` to a property ensures that it is not included in the model binding process, preventing users from setting values for those fields. Here, "Price" cannot be set via form submission.
```csharp
public decimal Price { get; set; } // Initially bindable
[Column(TypeName = "decimal(8, 2)")]
[BindNever]
public decimal Price { get; set; } // Now not bindable
```
x??

---

