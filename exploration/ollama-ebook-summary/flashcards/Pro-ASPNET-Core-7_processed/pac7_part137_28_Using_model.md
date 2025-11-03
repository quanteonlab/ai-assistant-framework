# Flashcards: Pro-ASPNET-Core-7_processed (Part 137)

**Starting Chapter:** 28 Using model binding

---

---
#### Model Binding Overview
Model binding is the process of creating .NET objects using data values from HTTP requests to provide easy access to the data required by action methods and Razor Pages. It automates the task of extracting necessary information from incoming requests, making it simpler for developers to work with complex data structures.
:p What is model binding?
??x
Model binding is a process that takes data from an HTTP request and automatically maps it into .NET objects used in controller actions or page handlers. This makes handling and processing user input more straightforward by leveraging C# types directly as parameters or properties.
x??
---
#### Simple Type Binding
In its simplest form, methods declare parameters with primitive types such as `int`, `string`, etc., which are then automatically populated from the HTTP request based on their names.
:p How does model binding handle simple type binding?
??x
Simple type binding involves declaring method parameters using basic data types like `int` or `string`. These parameter names correspond to form field names in the request, allowing for automatic population of these values without manual parsing.
```csharp
public IActionResult ExampleAction(string name, int age)
{
    // Code here uses 'name' and 'age'
}
```
x??
---
#### Complex Type Binding
Model binding can also bind complex types by using class properties as method parameters. The names of the properties are used to retrieve data from the HTTP request.
:p How does model binding handle complex type binding?
??x
Complex type binding involves declaring method parameters as objects with properties, which correspond to form fields or JSON keys in the request body. This allows for structured data handling within action methods.
```csharp
public IActionResult ExampleAction(MyComplexType model)
{
    // Code here uses 'model' which contains multiple properties
}

public class MyComplexType {
    public string Name { get; set; }
    public int Age { get; set; }
}
```
x??
---
#### Binding to a Property
You can use the `BindProperty` attribute to bind data directly to a property in a model.
:p How does the `BindProperty` attribute work for binding properties?
??x
The `BindProperty` attribute allows you to specify which parts of the request should be used to populate certain properties in your models. By applying this attribute, you can easily bind form fields or JSON properties directly to specific class properties.
```csharp
public IActionResult ExampleAction([BindProperty] MyComplexType model)
{
    // Code here uses 'model' with bound properties
}
```
x??
---
#### Binding Nested Types
When dealing with nested types, ensure that the form value names follow a dotted notation that matches the property hierarchy in your class.
:p How does model binding handle nested types?
??x
For nested types, use a hierarchical naming convention in the request data. For instance, if you have a `Person` class containing a `Address` object, the address fields should be named as `person.address.fieldName`.
```csharp
public class Person {
    public Address Address { get; set; }
}

public class Address {
    public string Street { get; set; }
}
```
In the request:
```
person.address.street=123 Main St
```
x??
---
#### Selecting Properties for Binding
Use `Bind` and `BindNever` attributes to specify which properties should be bound or ignored during model binding.
:p How do you control which properties are bound during model binding?
??x
You can use the `Bind` and `BindNever` attributes to selectively bind or ignore specific properties. The `Bind` attribute binds specified properties, while `BindNever` excludes them from being populated by the binder.
```csharp
public class MyComplexType {
    [BindProperty]
    public string Name { get; set; }

    [BindNever]
    public int Age { get; set; }
}
```
x??
---
#### Binding Collections
Follow sequence binding conventions to bind collections such as lists or arrays. Use `[FromForm]` or `[FromBody]` attributes as necessary.
:p How do you handle collection binding in model binding?
??x
Binding collections involves following specific conventions like using `List<T>` for form inputs with the same name repeated, or using JSON arrays when binding from request bodies. The `FromForm` and `FromBody` attributes are used to specify the source of the data.
```csharp
public IActionResult ExampleAction([FromForm] List<MyComplexType> models)
{
    // Code here uses 'models' which is a list of objects
}

public class MyComplexType {
    public string Name { get; set; }
}
```
x??
---
#### Specifying the Source for Binding
Use source attributes like `FromForm`, `FromBody`, etc., to explicitly specify where data values should be retrieved from in the request.
:p How do you specify the source of data during model binding?
??x
Source attributes such as `FromForm` and `FromBody` allow you to dictate whether the model binder should look for form fields or JSON content in the request body. This ensures that the correct part of the request is used to populate your models.
```csharp
public IActionResult ExampleAction([FromForm] MyComplexType model)
{
    // Code here uses 'model' with data from form fields
}
```
x??
---
#### Manually Performing Binding
Use `TryUpdateModel` method when you need more control over the binding process, such as applying custom logic or excluding certain properties.
:p How do you perform manual model binding using the `TryUpdateModel` method?
??x
The `TryUpdateModel` method provides a way to manually bind values from an HTTP request to a model. This allows for additional flexibility and customization beyond what automatic bindings can offer.
```csharp
public IActionResult ExampleAction()
{
    var model = new MyComplexType();
    if (TryUpdateModel(model)) {
        // Code here after successful binding
    } else {
        // Handle binding failure
    }
}
```
x??
---

---
#### Replacing Form.cshtml Content
Background context: This section describes how to prepare for using model binding by replacing the content of a specific file. The goal is to set up a form that can submit data to an action method, and use Razor syntax to bind the form fields to the model.

:p What changes are required in the `Form.cshtml` file located in the `Views/Form` folder?

??x
The contents of the `Form.cshtml` file should be replaced with the code shown in Listing 28.1. This involves updating the HTML structure and Razor syntax for creating an input form that binds to a model named `Product`.

```html
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
    <button type="submit" class="btn btn-primary mt-2">Submit</button>
</form>
```
x??

---
#### Commenting Out DisplayFormat Attribute
Background context: The `DisplayFormat` attribute was previously applied to the `Product` model. This attribute controls how data is displayed in views, but it can interfere with model binding during form submission.

:p What should be done with the `DisplayFormat` attribute for the `Price` property in the `Product.cs` file?

??x
The `DisplayFormat` attribute that has been applied to the `Price` property in the `Product.cs` file should be commented out, as shown in Listing 28.2. This is done to ensure proper model binding during form submission.

```csharp
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json.Serialization;

namespace WebApp.Models {
    public class Product {
        public long ProductId { get; set; }
        public required string Name { get; set; }
        [Column(TypeName = "decimal(8, 2)")]
        // [DisplayFormat(DataFormatString = "{0:c2}", ApplyFormatInEditMode = true)]
        public decimal Price { get; set; }
        public long CategoryId { get; set; }
        public Category? Category { get; set; }
    }
}
```
x??

---
#### Using Model Binding with HTML Forms
Background context: This section explains how to use model binding in ASP.NET Core applications, specifically focusing on setting up a form that binds to a `Product` model. Model binding automatically maps form data to the model properties.

:p What does model binding do in an ASP.NET Core application?

??x
Model binding automatically maps form data submitted via HTTP requests (like POST) to model properties. This means that when a user submits a form, the values entered in the form fields are mapped to the corresponding properties of the `Product` model.

For example, if you have a form with input fields for `Name` and `Price`, the values from these fields will be automatically assigned to the `Name` and `Price` properties of the `Product` object when the form is submitted.

Example:
```html
<form asp-action="submitform" method="post" id="htmlform">
    <div class="form-group">
        <label asp-for="Name"></label>
        <input class="form-control" asp-for="Name" />
    </div>
    <div class="form-group">
        <label asp-for="Price"></label>
        <input class="form-control" asp-for="Price" />
    </div>
    <button type="submit" class="btn btn-primary mt-2">Submit</button>
</form>
```

When this form is submitted, the `Name` and `Price` values from the input fields will be bound to a `Product` model instance that can be used in your action method.

x??

---
#### Action Method for Form Submission
Background context: This section explains how to handle form submissions using an action method. The action method will receive the data submitted via the form and process it accordingly.

:p How should you define an action method to handle form submissions?

??x
An action method should be defined in your controller to handle form submissions. The method signature should match the model type (in this case, `Product`) and include necessary parameters for the HTTP request.

Example:
```csharp
using Microsoft.AspNetCore.Mvc;

namespace WebApp.Controllers {
    public class FormController : Controller {
        [HttpPost]
        public IActionResult SubmitForm(Product product) {
            // Process the submitted form data
            if (ModelState.IsValid) {
                // Save to database or perform other actions
                return RedirectToAction("Success");
            }
            else {
                // Handle invalid model state
                return View("Form", product);
            }
        }
    }
}
```

In this example, the `SubmitForm` action method accepts a `Product` object as its parameter. If the model state is valid, you can process the data (e.g., save it to a database). If not, you might need to return the form view again with error messages.

x??

---

---
#### Dropping the Database
Background context: The process of dropping a database in ASP.NET Core using Entity Framework Command Line Tools is described. This involves executing specific commands to ensure that the application's local database is properly set up or reset, which can be particularly useful when starting new examples or troubleshooting issues.
:p How do you drop the database for an ASP.NET Core application using Entity Framework?
??x
To drop the database for an ASP.NET Core application using Entity Framework Command Line Tools, you need to open a PowerShell command prompt and navigate to the project folder. Then, run the following command:
```
dotnet ef database drop --force
```
This command drops the existing database associated with your project.
x??

---
#### Running the Example Application
Background context: Instructions on how to start an ASP.NET Core application using the `dotnet run` command and navigate to a specific URL in a web browser are provided. This process allows users to interact with the application's forms and observe model binding in action.
:p How do you run the example application?
??x
To run the example application, open a PowerShell command prompt, navigate to the project folder containing the `WebApp.csproj` file, and execute the following command:
```
dotnet run
```
This command starts the development server for your ASP.NET Core application. You can then use a web browser to access it by navigating to `http://localhost:5000/controllers/form`.
x??

---
#### Understanding Model Binding
Background context: Model binding in ASP.NET Core acts as an intermediary between HTTP requests and action methods, facilitating the population of model objects with data from various sources such as form fields, route parameters, query strings, or request bodies.
:p What is model binding in ASP.NET Core?
??x
Model binding in ASP.NET Core is a mechanism that automatically binds incoming HTTP requests to action method parameters. It simplifies the process of populating models and view models with data from various sources like form fields, routing segment variables, query strings, or request bodies.
:x??

---
#### Model Binding Sources
Background context: The model binding system in ASP.NET Core looks for values based on a specific order: form data, request body (for `ApiController` decorated controllers), routing segment variables, and query strings. This process is crucial to understanding how data gets passed between the application and the user.
:p What sources does the model binding system check for parameter values?
??x
The model binding system in ASP.NET Core checks the following sources in order when looking for parameter values:
1. Form data
2. Request body (only for controllers decorated with `ApiController`)
3. Routing segment variables
4. Query strings

For instance, if a request includes form fields and also has a query string, the model binding system will first check for form data, then routing data, before finally checking the query string.
:x??

---
#### Handling Multiple Values in Model Binding
Background context: When multiple values are present (e.g., both in route segments and query strings), the order of preference matters. The system prioritizes routing segment variables over other sources. This can lead to different outcomes based on the URL structure.
:p How does model binding handle URLs with both routing segment values and query string parameters?
??x
When a URL contains both routing segment values (e.g., `http://localhost:5000/controllers/Form/Index/5`) and query string parameters (e.g., `http://localhost:5000/controllers/Form/Index/5?id=1`), the model binding system follows this order of preference:
1. Routing segment variables
2. Form data
3. Request body (only for controllers decorated with `ApiController`)
4. Query strings

In the given example, if a request is made to `http://localhost:5000/controllers/Form/Index/5`, the model binding system will use the value from the routing segment (`id = 5`). If instead, you request `http://localhost:5000/controllers/Form/Index?id=4`, it would use the query string value because no routing segment is present.
:x??

---

