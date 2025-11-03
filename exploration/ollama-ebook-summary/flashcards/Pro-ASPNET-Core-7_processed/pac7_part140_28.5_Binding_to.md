# Flashcards: Pro-ASPNET-Core-7_processed (Part 140)

**Starting Chapter:** 28.5 Binding to arrays and collections. 28.5.1 Binding to arrays

---

#### Binding to Arrays and Collections
Background context explaining how ASP.NET Core handles binding request data to arrays and collections. The default model binder supports populating array properties based on form inputs.

:p How does the default model binder handle binding to arrays?
??x
The default model binder automatically creates an array property in your model class and fills it with values from the corresponding input elements in the form. This is achieved by setting the `name` attribute of all input elements to be the same, which allows the model binder to map them to a single array property.

For example:
```csharp
public class BindingsModel : PageModel {
    [BindProperty(Name = "Data")]
    public string[] Data { get; set; } = Array.Empty<string>();
}
```

Here, all input elements with `name="Data"` will be bound to the `Data` property of the `BindingsModel`.
x??

---
#### Name Attribute for Specifying Position
Background context explaining that by default, arrays are populated in the order received from the browser. However, you can override this behavior using the `name` attribute.

:p How can you specify the position of values in an array when binding with the model binder?
??x
You can use the index notation within the `name` attribute to specify the exact position of each value in the array. This allows you to reorder or reposition values as needed.

For example, in the form:
```html
<form asp-page="Bindings" method="post">
    <div class="form-group">
        <label>Value #1</label>
        <input class="form-control" name="Data[1]" value="Item 1" />
    </div>
    <div class="form-group">
        <label>Value #2</label>
        <input class="form-control" name="Data[0]" value="Item 2" />
    </div>
    <div class="form-group">
        <label>Value #3</label>
        <input class="form-control" name="Data[2]" value="Item 3" />
    </div>
    <button type="submit" class="btn btn-primary">Submit</button>
    <a class="btn btn-secondary" asp-page="Bindings">Reset</a>
</form>
```

In this example, even though the inputs are defined in a specific order, setting `name` attributes with indices `[1]`, `[0]`, and `[2]` ensures that "Item 2" is bound to index 0, "Item 1" to index 1, and "Item 3" to index 2.
x??

---
#### Filtering Null Values in Arrays
Background context explaining the default behavior of handling empty fields as `null` values in arrays. This can lead to unwanted null values in your data model.

:p How do you handle null values in array properties when binding with the model binder?
??x
To filter out null values from being bound into your array, you can use LINQ's `Where` method or conditional statements within Razor syntax to exclude them before displaying or processing the data.

For example:
```razor
@functions {
    public class BindingsModel : PageModel {
        [BindProperty(Name = "Data")]
        public string[] Data { get; set; } = Array.Empty<string>();
    }
}

<ul class="list-group">
    @foreach (string s in Model.Data.Where(s => s != null)) {
        <li class="list-group-item">@s</li>
    }
</ul>
```

This ensures that any `null` values are not included in the displayed list, giving you a cleaner output.
x??

---
#### Binding Required Attribute
Background context explaining the usage of the `BindRequired` attribute to enforce that required properties must have non-null values.

:p What is the purpose of using the `BindRequired` attribute when binding model data?
??x
The `BindRequired` attribute can be used on a property in your model class to indicate that it should not accept null or empty values. If such a value is submitted and there is no default value, ASP.NET Core will produce a validation error.

For example:
```csharp
public class BindingsModel : PageModel {
    [BindProperty(Name = "Data", BindRequired = true)]
    public string[] Data { get; set; } = Array.Empty<string>();
}
```

This ensures that if the `Data` property is not provided in the request, ASP.NET Core will throw a validation error.
x??

---

---
#### Binding to a SortedSet
Background context: The model binding process can populate collections, including `SortedSet<string>`, with input values from form elements. The values will be automatically sorted based on the collection's implementation.

:p How does ASP.NET Core model binding handle population of a `SortedSet<string>`?
??x
The model binder processes form inputs and populates the `Data` property (of type `SortedSet<string>`) with the input values, which are then sorted alphabetically. This happens without needing to manually sort the set after population.

```csharp
public class BindingsModel : PageModel {
    [BindProperty(Name = "Data")]
    public SortedSet<string> Data { get; set; } = new SortedSet<string>();
}
```

x??
---
#### Binding to a Dictionary with Index Notation
Background context: When binding elements to a dictionary using index notation in the `name` attribute of form inputs, the model binder uses these indices as keys for the key-value pairs.

:p How does ASP.NET Core model binding handle population of a `Dictionary<string, string>` using index notation?
??x
The model binder processes form inputs and populates the `Data` dictionary with the input values, using the index parts in the `name` attribute as keys. This allows multiple elements to be transformed into key-value pairs.

```csharp
public class BindingsModel : PageModel {
    [BindProperty(Name = "Data")]
    public Dictionary<string, string> Data { get; set; } 
        = new Dictionary<string, string>();
}
```

Example form:
```html
<form asp-page="/pages/bindings" method="post">
    <div class="form-group">
        <label>Value #1</label>
        <input class="form-control" name="Data[first]" value="Item 1" />
    </div>
    <div class="form-group">
        <label>Value #2</label>
        <input class="form-control" name="Data[second]" value="Item 2" />
    </div>
    <div class="form-group">
        <label>Value #3</label>
        <input class="form-control" name="Data[third]" value="Item 3" />
    </div>
    <button type="submit" class="btn btn-primary">Submit</button>
    <a class="btn btn-secondary" asp-page="/pages/bindings">Reset</a>
</form>
```

x??
---

#### Model Binding to Collections and Complex Types
In ASP.NET Core, model binding is a mechanism that automatically binds form data or query string values to action method parameters. This process simplifies handling collections of simple types or complex types as input.

:p What is model binding used for?
??x
Model binding in ASP.NET Core helps in automatically mapping the incoming HTTP request's data (form fields, URL parameters) to the properties of your application’s models. This avoids manual parsing and validation, making the code cleaner and easier to maintain.
x??

---

#### Common Prefix in Model Binding
In model binding, when using collections or complex types, elements that provide values must share a common prefix followed by an index or key.

:p What is required for model binding with collections?
??x
For model binding with collections, all input fields should use the same prefix (common part of their names) and append an index or key. For example, if you have a collection named `Data`, each form field might look like `Data[0].PropertyName` or `Data[first]`.

Example:
```html
<input class="form-control" name="Data[0].Name" />
<input class="form-control" name="Data[1].Price" />
```
x??

---

#### Binding to a Dictionary
When binding to a dictionary, the keys and values from form data are used directly as dictionary entries.

:p How does model binding work with dictionaries?
??x
Model binding for dictionaries works by using a common prefix followed by the key. The key-value pairs in the form data are then mapped directly to the dictionary's structure.

For example:
- Form field name: `Data[first]`
- Value: "First value"
- Resulting dictionary entry: `Dictionary["first"] = "First value"`

:p How is this demonstrated in the given code?
??x
In the provided code, form fields are named using a common prefix (`Data`) followed by keys (e.g., `first`, `second`, `third`). When these values are submitted, they map directly to dictionary entries.

Example:
```html
<input class="form-control" name="Data[first]" value="First Value" />
```
x??

---

#### Binding to Complex Types in Arrays
Model binding can also handle complex types stored in arrays. Each element in the array is bound based on its index and properties.

:p How does model binding work with arrays of complex objects?
??x
When using an array of complex objects, each input field should be named according to the object's property paths. The model binder will map these values to the corresponding properties of the objects in the array.

Example:
```html
<input class="form-control" name="Data[0].Name" value="Product-0" />
<input class="form-control" name="Data[1].Price" value="101" />
```
x??

---

#### Removing Attributes for Excluded Properties
Sometimes, certain properties need to be excluded from the binding process using attributes like `BindNever`.

:p How does one exclude a property from model binding?
??x
To exclude a property from model binding, you can use the `[BindNever]` attribute on the property definition. This ensures that the property's value is not bound during form submission.

Example:
```csharp
public class Product {
    [BindNever]
    public decimal Price { get; set; }
}
```
x??

---

#### Re-enabling Excluded Properties in Model Binding
If you need to include a previously excluded property, you can remove the `BindNever` attribute and ensure it is bound correctly.

:p How does one re-enable binding for an excluded property?
??x
To re-enable binding for a previously excluded property, simply remove or comment out the `[BindNever]` attribute from the property definition. The model binder will then include this property in its binding process.

Example:
```csharp
public class Product {
    // [BindNever]
    public decimal Price { get; set; }
}
```
x??

---

#### Summary of Key Concepts
- **Model Binding**: Mechanism to automatically map incoming request data to model properties.
- **Common Prefix**: Required for collections and complex types, ensuring correct binding.
- **Dictionary Binding**: Direct mapping of key-value pairs in form data.
- **Complex Type Arrays**: Binding individual properties based on their index positions.
- **Excluding Properties**: Using `[BindNever]` attribute, and re-enabling it.

:p How do you handle model binding with collections and complex types?
??x
To handle model binding with collections and complex types:

1. Use a common prefix for all input fields that belong to the collection or complex type.
2. Ensure each form field is named correctly according to its position in the array or dictionary key-value pair.
3. For complex types, name the inputs based on their property paths.
4. Use attributes like `[BindNever]` to exclude properties from binding and remove them when needed.

Example:
```html
<!-- Form fields -->
<input class="form-control" name="Data[0].Name" value="Product-0" />
<input class="form-control" name="Data[1].Price" value="101" />
```

Example Model Class:
```csharp
public class Product {
    public string Name { get; set; }
    [BindNever]
    public decimal Price { get; set; } // Excluded initially, then re-enabled.
}
```
x??

