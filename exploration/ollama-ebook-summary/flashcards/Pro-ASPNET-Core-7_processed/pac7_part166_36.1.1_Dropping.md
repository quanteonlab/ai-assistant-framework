# Flashcards: Pro-ASPNET-Core-7_processed (Part 166)

**Starting Chapter:** 36.1.1 Dropping the database and running the application. 36.2 Using the Blazor form components

---

#### Dropping and Running the Database
Background context explaining how to drop and run a database using the Entity Framework (EF) Core. This is crucial for ensuring that your application starts with a clean database.

:p How do you drop the database using EF Core?

??x
To drop the database, you use the `dotnet ef` command-line tool provided by .NET Core. The specific command is:
```shell
dotnet ef database drop --force
```
This command will delete the existing database and prepare for a new one to be created when the application runs again.

x??

---

#### Running the Example Application
Context about running the example application in a local development environment using .NET Core. This involves navigating to the correct directory and starting the server.

:p How do you run the Blazor example application?

??x
To run the example application, open a PowerShell command prompt, navigate to the folder containing the `Advanced.csproj` file, and execute:
```shell
dotnet run
```
This command will start the development server. You can then access the application by navigating to `http://localhost:5000/forms` in your web browser.

x??

---

#### Using Blazor Form Components
Overview of Blazor form components, their purpose, and how they are used for data validation and user interaction. This includes understanding the roles of various components like `EditForm`, `InputText`, etc.

:p What is the primary component for rendering a form in Blazor?

??x
The primary component for rendering a form in Blazor is the `EditForm`. It provides the foundational structure needed for data validation and ensures that server-side properties are updated after user interaction. 

For example, consider an `EditForm` used to edit a person's details:
```razor
<EditForm Model="PersonData">
    <div class="form-group">
        <label>Firstname</label>
        <InputText class="form-control" @bind-Value="PersonData.Firstname" />
    </div>
</EditForm>
```

x??

---

#### Binding and Validation in Blazor Forms
Explanation of how to bind data using the `@bind-Value` attribute and perform validation with `EditForm`.

:p How do you bind a property value to an input element in Blazor?

??x
You can bind a property value to an input element by using the `@bind-Value` attribute. For example, to bind the `Firstname` property of a `Person` object to an `InputText` component:
```razor
<InputText class="form-control" @bind-Value="PersonData.Firstname" />
```
This ensures that any changes made in the input field are reflected back into the `Firstname` property.

x??

---

#### Disabling Input Fields in Blazor Forms
Explanation of how to disable an input field using attributes and attribute splatting.

:p How do you disable a specific input field within an `EditForm`?

??x
To disable a specific input field, you can use the `disabled` attribute on the corresponding component. For example, to disable the `PersonId` input field:
```razor
<InputNumber class="form-control" @bind-Value="PersonData.PersonId" disabled />
```
This will prevent users from modifying the `PersonId`, while still allowing them to interact with other parts of the form.

x??

---

#### Handling Parameter Initialization in Blazor Forms
Explanation of how to initialize and retrieve data for a specific form instance using parameters.

:p How do you initialize a model object within an `EditForm` component?

??x
You can initialize a model object by setting up properties and handling parameter initialization. For example, in the provided code:
```razor
public Person PersonData { get; set; } = new();
```
And then using the `OnParametersSetAsync` method to fetch data from the context if available:
```razor
protected async override Task OnParametersSetAsync() {
    if (Context != null) {
        PersonData = await Context.People.FindAsync(Id)
            ?? new Person();
    }
}
```

x??

---

#### Summary and Data Table Display
Overview of displaying summary data and a form for editing.

:p How does the summary section update in response to changes made in an `EditForm`?

??x
The summary section updates automatically when changes are made within the `EditForm`. For example, if you have:
```razor
<summary>Current values: ID = @PersonData.PersonId, Name = @PersonData.Firstname</summary>
```
Any changes to properties like `Firstname` or `PersonId` will be reflected in the summary section as soon as the user interacts with the form.

x??

---

#### Custom Form Components in Blazor
Background context: Blazor provides built-in components for common input elements like `<InputNumber>` and `<InputText>`. However, for more complex scenarios or custom requirements, developers need to create their own form components. This is demonstrated through creating a `CustomSelect` component that integrates into the Blazor forms feature.

:p What is the purpose of the `CustomSelect` component in Blazor?
??x
The `CustomSelect` component serves as a reusable and customizable dropdown select element for Blazor applications, allowing developers to map between user-visible string values and model property values. This component leverages the built-in `InputBase<TValue>` class to handle most of the form binding logic.

```csharp
@typeparam TValue
@inherits InputBase<TValue>
```
This snippet shows how the component is defined with a generic type parameter `TValue`, which represents the type of the model property value. The component inherits from `InputBase<TValue>`, providing it with necessary properties and methods for handling form values.

??x
The answer explains that the `CustomSelect` component is designed to be flexible, allowing developers to handle different types of data without needing to implement specific parsing logic within each custom component.
x??

---
#### CustomForm TryParseValueFromString Method
Background context: The `CustomSelect` component includes a method called `TryParseValueFromString`, which is crucial for mapping between the string values used by HTML elements and the corresponding value for the C# model property. This method must be implemented to ensure proper data validation during form submission.

:p What is the purpose of implementing the `TryParseValueFromString` method in the `CustomSelect` component?
??x
The purpose of implementing the `TryParseValueFromString` method is to allow Blazor to correctly parse string values from user input into C# model property values. This method ensures that when a user selects an option from the dropdown, the selected value is properly mapped and validated before being bound to the underlying model.

```csharp
protected override bool TryParseValueFromString(string? value,
    [MaybeNullWhen(false)] out TValue? result,
    [NotNullWhen(false)] out string? validationErrorMessage)
{
    try {
        if (Parser != null && value != null) {
            result = Parser(value);
            validationErrorMessage = null;
            return true;
        }
        result = default(TValue);
        validationErrorMessage = "Value or parser not defined";
        return false;
    } catch {
        result = default(TValue);
        validationErrorMessage = "The value is not valid";
        return false;
    }
}
```
This code snippet demonstrates the implementation of `TryParseValueFromString`. It checks if a parser function and non-null value are provided, then attempts to parse the string into a model property value. If parsing fails or necessary parameters are missing, appropriate validation error messages are returned.

??x
The answer explains that this method is essential for ensuring data integrity and proper binding between user input and the underlying model properties.
x??

---
#### Using Custom Components in Blazor Forms
Background context: To utilize custom components like `CustomSelect` within a form, they need to be properly integrated. This involves defining the component and then using it within an `<EditForm>` element to handle data validation and binding.

:p How can you use the `CustomSelect` component to bind values for properties defined by the `Person` class?
??x
To use the `CustomSelect` component, you need to define a dictionary mapping string values (displayed options) to their corresponding `TValue` (model property values). This is then passed as a parameter to the `CustomSelect` component along with the `Parser` function.

```razor
<CustomSelect TValue="long" Values="Departments"
             Parser="@((string str) => long.Parse(str))"
             @bind-Value="PersonData.DepartmentId">
    <option selected disabled value="0">Choose a Department</option>
</CustomSelect>

<CustomSelect TValue="long" Values="Locations"
             Parser="@((string str) => long.Parse(str))"
             @bind-Value="PersonData.LocationId">
    <option selected disabled value="0">Choose a Location</option>
</CustomSelect>
```
Here, the `@bind-Value` attribute is used to bind the selected value from the custom select element directly to the `DepartmentId` and `LocationId` properties of the `Person` model. The `Parser` function ensures that user input (a string) is correctly converted into a long integer.

??x
The answer highlights how to integrate custom components like `CustomSelect` into Blazor forms, emphasizing the importance of data binding and validation through proper component definition and usage.
x??

---
#### Validation in Custom Components
Background context: Custom form components need to handle validation logic for user input. This is particularly important when dealing with generic types and ensuring that string values are correctly mapped to model properties.

:p How does the `CustomSelect` component ensure proper data validation during binding?
??x
The `CustomSelect` component ensures proper data validation by implementing the `TryParseValueFromString` method, which handles the conversion of user input strings into model property values. This method checks if a parser function is provided and if the value to be parsed is not null. If valid, it converts the string to the appropriate type; otherwise, it returns an error message.

```csharp
protected override bool TryParseValueFromString(string? value,
    [MaybeNullWhen(false)] out TValue? result,
    [NotNullWhen(false)] out string? validationErrorMessage)
{
    try {
        if (Parser != null && value != null) {
            result = Parser(value);
            validationErrorMessage = null;
            return true;
        }
        result = default(TValue);
        validationErrorMessage = "Value or parser not defined";
        return false;
    } catch {
        result = default(TValue);
        validationErrorMessage = "The value is not valid";
        return false;
    }
}
```
This method provides a mechanism to validate and parse user input, ensuring that only correctly formatted strings are bound to the model properties. If parsing fails or necessary parameters are missing, appropriate error messages are returned.

??x
The answer explains how the `CustomSelect` component uses the `TryParseValueFromString` method to ensure proper data validation during binding.
x??

---
#### Displaying and Disabling Elements in Forms
Background context: In a Blazor form, it's often necessary to display or disable certain input elements based on the state of the application. This can be achieved by using attributes like `disabled`.

:p How do you make an `<InputNumber>` element disabled within a form?
??x
To make an `<InputNumber>` element disabled within a form, you can use the `@bind-Value` attribute along with the `disabled` attribute to control its state. The `disabled` attribute should be set conditionally based on whether the field should be editable or not.

```razor
<InputNumber class="form-control" @bind-Value="PersonData.PersonId" disabled />
```
In this example, the `disabled` attribute is used directly within the `<InputNumber>` element. This ensures that the input number field is read-only and cannot be edited by the user. The value of `PersonId` can still be bound to the model but will not accept new user inputs.

??x
The answer explains how to make an input element disabled in a Blazor form, ensuring it remains non-editable while maintaining data binding.
x??

---

#### Blazor Validation Components Overview
Blazor provides validation components and classes to handle form validation, making it easy to integrate error handling into forms. These components include `DataAnnotationsValidator`, `ValidationMessage`, and `ValidationSummary`. The latter two generate elements styled with CSS for better user experience.

:p What are the main Blazor components used for form validation?
??x
The main Blazor components used for form validation are `DataAnnotationsValidator`, which integrates model validation attributes, `ValidationMessage` for property-level error messages, and `ValidationSummary` for summary error messages.

```razor
<FormSpy PersonData="PersonData">
    <EditForm Model="PersonData">
        <DataAnnotationsValidator />
        <ValidationSummary />
        <!-- Other form components -->
    </EditForm>
</FormSpy>
```
x??

---

#### CSS Classes for Validation in Blazor
Blazor uses specific CSS classes to indicate the validation status of form elements. These include `modified`, `valid`, and `invalid` classes, which are added or removed based on the user's interaction with the form.

:p What are the key CSS classes used by Blazor for indicating validation status?
??x
The key CSS classes used by Blazor for indicating validation status are:
- `modified`: Applied to elements once the value has been edited.
- `valid`: Added when a valid value is entered.
- `invalid`: Applied if the entered value fails validation.

These classes can be styled in a CSS file, as shown below:

```css
.validation-errors {
    background-color: rgb(220, 53, 69);
    color: white;
    padding: 8px;
    text-align: center;
    font-size: 16px;
    font-weight: 500;
}

div.validation-message {
    color: rgb(220, 53, 69);
    font-weight: 500
}

.modified.valid {
    border: solid 3px rgb(40, 167, 69);
}

.modified.invalid {
    border: solid 3px rgb(220, 53, 69);
}
```
x??

---

#### Applying Validation in the Editor Component
To apply validation in a Blazor form, you need to use components like `DataAnnotationsValidator` and `ValidationSummary`. These components generate appropriate HTML elements styled with CSS for better user interaction.

:p How do you apply validation using Blazor's built-in components?
??x
To apply validation using Blazor's built-in components, you need to include the following in your form:

- `DataAnnotationsValidator`: Integrates model validation attributes.
- `ValidationSummary`: Displays summary error messages.
- `ValidationMessage` for property-level error messages.

Here is an example of how these components are used in the Editor component:

```razor
@page "/forms/edit/{id:long}"
@layout EmptyLayout

<link href="/blazorValidation.css" rel="stylesheet" />

<h4 class="bg-primary text-center text-white p-2">Edit</h4>

<FormSpy PersonData="PersonData">
    <EditForm Model="PersonData">
        <DataAnnotationsValidator />
        <ValidationSummary />
        <!-- Other form components -->
    </EditForm>
</FormSpy>
```
x??

---

#### Custom Validation Using Data Annotations
To enable validation on a model, you need to apply attributes like `Required`, `MinLength`, and `Range` using the `[Attribute]` syntax. These attributes help in defining specific rules for data validation.

:p How do you apply data validation attributes to a model class?
??x
To apply data validation attributes to a model class, you use attributes such as `Required`, `MinLength`, and `Range`. For example:

```csharp
using System.ComponentModel.DataAnnotations;

namespace Advanced.Models
{
    public class Person
    {
        // Other properties

        [Required(ErrorMessage = "A firstname is required")]
        [MinLength(3, ErrorMessage = "Firstnames must be 3 or more characters")]
        public string Firstname { get; set; } = String.Empty;

        [Required(ErrorMessage = "A surname is required")]
        [MinLength(3, ErrorMessage = "Surnames must be 3 or more characters")]
        public string Surname { get; set; } = String.Empty;

        // Other properties
    }
}
```
x??

---

#### Handling Validation Errors in Blazor Forms
Blazor handles validation errors by displaying messages based on the state of the form. When a user tabs out of an input field, if it fails validation, an error message is shown.

:p What happens when a validation error occurs in a Blazor form?
??x
When a validation error occurs in a Blazor form:
- The `ValidationMessage` component displays an error message for the specific property.
- The `ValidationSummary` component shows summary messages if any properties fail validation.
- The form elements are styled with CSS classes like `modified.invalid`, which indicate that the input is invalid.

For example, if the user deletes characters from a required field and tabs out of it:
```razor
<ValidationMessage For="() => PersonData.Firstname" />
```
x??

---

