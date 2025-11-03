# Flashcards: Pro-ASPNET-Core-7_processed (Part 143)

**Starting Chapter:** 29.3 Validating data

---

---
#### Model Validation Introduction
Background context: ASP.NET Core performs basic data validation during model binding but discards these errors since it hasn't been instructed on how to respond. The `ModelStateDictionary` class is used to track details of the state of the model object, especially validation errors.

:p What does the `ModelState` property in ASP.NET Core do?
??x
The `ModelState` property returns a `ModelStateDictionary` that tracks the state of each model property. It provides methods like `IsValid`, which checks if all properties are valid and contains validation error messages.
x??

---
#### Checking Outcome of Validation Process
:p How does one check if data provided by the user is valid in ASP.NET Core?
??x
One can use the `ModelState.IsValid` property to check if all the model properties are valid. If there are no validation errors, `IsValid` returns true; otherwise, it returns false.
```csharp
if (ModelState.IsValid)
{
    // Process valid data
}
else
{
    // Handle invalid data by redisplaying the form with error messages
    return View("Form");
}
```
x??

---
#### Handling Invalid Data
:p What happens if validation fails in the `SubmitForm` method?
??x
If validation fails, the `ModelState.IsValid` property returns false. The method then re-renders the form view with the model state errors included, allowing the user to correct them.
```csharp
if (ModelState.IsValid)
{
    // Save valid data and redirect to results page
}
else
{
    return View("Form");
}
```
x??

---
#### Displaying Validation Error Feedback
:p How does ASP.NET Core provide feedback when validation fails?
??x
ASP.NET Core uses tag helpers to modify the input elements based on validation errors. When a field has an error, it gets added to the `input-validation-error` class, which can be styled by CSS libraries like Bootstrap.
```html
<input class="form-control" asp-for="Name" data-val-required="The Name field is required." />
```
x??

---
#### Customizing Validation Error Styling with JavaScript and Bootstrap
:p How does one ensure that validation error classes are applied correctly when using a framework like Bootstrap?
??x
To apply the built-in validation styles from frameworks like Bootstrap, you can use JavaScript to map between ASP.NET Core's class names and Bootstrap's error classes. The example uses an event listener to add the `is-invalid` class.
```javascript
<script type="text/javascript">
    window.addEventListener("DOMContentLoaded", () => {
        document.querySelectorAll("input.input-validation-error")
            .forEach((elem) => { elem.classList.add("is-invalid"); });
    });
</script>
```
x??

---

#### Displaying Validation Summary
Background context: In ASP.NET Core, validation summaries are a crucial part of form handling that provide users with feedback about any errors encountered during data submission. The `ValidationSummaryTagHelper` is used to display these error messages.

:p How does the `asp-validation-summary` tag helper help in displaying validation errors?
??x
The `asp-validation-summary` tag helper detects the presence of validation errors and generates a summary message, which is displayed on the form when an invalid submission occurs. The attribute value `"All"` is used to display all detected validation errors.

```html
<div asp-validation-summary="All" class="text-danger"></div>
```
This tag will add a `<div>` element with the `text-danger` class if there are any validation errors, displaying them in red text.

x??

---

#### Highlighting Validation Error Input Fields
Background context: In addition to showing summary messages, it's often necessary to highlight specific input fields that have validation errors. This can be done using CSS classes and JavaScript.

:p How does the provided HTML snippet highlight input fields with validation errors?
??x
The provided HTML uses a custom JavaScript function or event listener to add the `input-validation-error` class to input fields that are invalid upon form submission. This is detected by the presence of the `data-valmsg-for` attribute, which matches the name of the input field.

```html
<div class="form-group">
    <label asp-for="Name"></label>
    <input class="form-control" asp-for="Name" data-valmsg-for="Name" />
</div>

<script>
    // Example JavaScript to highlight errors
    document.addEventListener("DOMContentLoaded", function() {
        const form = document.getElementById('htmlform');
        form.onsubmit = (e) => {
            if (!form.checkValidity()) {
                const inputs = form.querySelectorAll('.input-validation-error');
                for (let input of inputs) {
                    input.classList.add('highlight');
                }
            }
        };
    });
</script>
```
x??

---

#### ASP.NET Core Validation Summary Attributes
Background context: The `asp-validation-summary` attribute can take different values to control how and when validation summary messages are displayed.

:p What does the `asp-validation-summary="All"` attribute do in the given example?
??x
The `asp-validation-summary="All"` attribute ensures that all validation errors for the form are displayed as a single message. This is useful when you want to inform users about multiple errors at once.

```html
<div asp-validation-summary="All" class="text-danger"></div>
```
This tag will generate an error summary if there are any validation issues, showing them in red text.

x??

---

#### ASP.NET Core Validation Messages
Background context: Validation messages can be customized and displayed using the `ValidationSummary` enumeration. Different values within this enumeration control how errors are presented to the user.

:p What does the `ValidationSummary` enum include?
??x
The `ValidationSummary` enum includes several values that determine how validation summary messages are displayed:
- `"All"`: Displays all validation errors.
- `"ModelOnly"`: Displays only model-level validation errors, excluding individual property-level errors.
- `"None"`: Disables the tag helper.

```csharp
// Example usage in C#
public class MyViewModel {
    [Required(ErrorMessage = "Name is required")]
    public string Name { get; set; }
}
```
The `asp-validation-summary` attribute can be used to specify which type of validation errors should be displayed.

x??

---

#### ASP.NET Core Validation on Form Submission
Background context: When a form is submitted, ASP.NET Core validates the data before processing it. If validation fails, specific actions are taken based on the configuration and requirements.

:p What happens when you submit an invalid form in ASP.NET Core?
??x
When you submit an invalid form in ASP.NET Core, the `ValidationSummary` tag helper detects that not all required fields have been filled or other validation rules have been violated. The system then displays a summary of these errors to the user.

```html
<form asp-action="submitform" method="post" id="htmlform">
    <!-- Form fields here -->
    <div asp-validation-summary="All" class="text-danger"></div>
</form>
```
Upon submission, if validation fails, an error summary is displayed in a red text box at the top of the form.

x??

---

---
#### Required Property Validation
Background context: In ASP.NET Core, required properties are automatically validated to ensure a value is provided. This is part of implicit validation performed during model binding. The `required` keyword in C# ensures that a property must have a non-null value.

:p What does the `required` keyword do for properties in C#?
??x
The `required` keyword enforces that a property cannot be null, and if it is not provided by the user, an implicit validation error will be generated. This is useful for ensuring fields like names or IDs are always filled out.
```csharp
public class Product {
    public required string Name { get; set; } // Ensures 'Name' must have a value
}
```
x?
---
#### Price Parsing Validation
Background context: Implicit validation also checks if the incoming HTTP request can be parsed into the expected data type. For properties with non-nullable types, ASP.NET Core attempts to parse the string values from the request body.

:p What happens when submitting an invalid price in the form?
??x
When a user submits an invalid price (e.g., "ten") for the `Price` property, ASP.NET Core will generate a parsing validation error because it cannot convert this string into a decimal value. This results in displaying an error message indicating that the input is not valid.

```csharp
// Example of the error handling logic:
if (ModelState.GetValidationState(nameof(Product.Price)) == ModelValidationState.Valid && product.Price <= 0) {
    ModelState.AddModelError(nameof(Product.Price), "Enter a positive price");
}
```
x?
---
#### Explicit Validation Checks
Background context: While implicit validation takes care of basic requirements like non-null values and type conversions, explicit validation is necessary to perform more detailed checks. This involves using `ModelStateDictionary` methods to add custom error messages.

:p How does one check the validation state for a property in ASP.NET Core?
??x
To check the validation state of a property in ASP.NET Core, you can use the `GetValidationState` method from `ModelStateDictionary`. It returns an enumeration value indicating whether the property is valid or has any issues. For instance:

```csharp
if (ModelState.GetValidationState(nameof(Product.Price)) == ModelValidationState.Valid && product.Price <= 0) {
    ModelState.AddModelError(nameof(Product.Price), "Enter a positive price");
}
```

This code checks if the `Price` property passed implicit validation and then adds an error message if it's not greater than zero.

x?
---
#### Validation States
Background context: The `ModelStateDictionary.GetValidationState` method returns one of several values from the `ModelValidationState` enumeration, which describes whether a model property is valid or has issues. This helps in determining the next steps when dealing with form validation.

:p What are the possible values returned by `GetValidationState`?
??x
The `GetValidationState` method can return four possible values from the `ModelValidationState` enumeration:
- Unvalidated: No validation has been performed.
- Valid: The request value associated with the property is valid.
- Invalid: The request value associated with the property is invalid and should not be used.
- Skipped: The model property has not been processed.

These states help in making decisions on whether to add or remove error messages based on the validation results.

x?
---

