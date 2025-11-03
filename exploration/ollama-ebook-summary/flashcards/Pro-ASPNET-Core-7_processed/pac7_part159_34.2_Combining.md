# Flashcards: Pro-ASPNET-Core-7_processed (Part 159)

**Starting Chapter:** 34.2 Combining components

---

#### Combining Components in Blazor
Background context: In this section, you will learn how to combine components to create more complex features. This involves adding new components and integrating them with existing ones to enhance functionality.

:p How does one apply a component within another component?
??x
To apply a component within another component, you use the name of the component as an element instead of the `<ComponentName>` tag used in controller views or Razor Pages. For example:

```razor
<SelectFilter />
```

This integrates `SelectFilter` directly into the current component's template.
x??

---
#### Creating SelectFilter Component
Background context: The `SelectFilter` component is created to allow users to choose a city from a dropdown list.

:p What does the `SelectFilter.razor` file contain?
??x
The `SelectFilter.razor` file contains the following content:

```razor
<div class="form-group">
    <label for="select-@Title">@Title</label>
    <select name="select-@Title" class="form-control" @bind="SelectedValue">
        <option disabled selected value="">Select @Title</option>
        @foreach (string val in Values) {
            <option value="@val" selected="@(val == SelectedValue)">
                @val
            </option>
        }
    </select>
</div>

@code {
    public IEnumerable<string> Values { get; set; } = Enumerable.Empty<string>();
    public string? SelectedValue { get; set; }
    public string Title { get; set; } = "Placeholder";
}
```

It renders a `select` element allowing users to choose from the provided values. The `@bind` directive is used to bind the selected value.
x??

---
#### Applying SelectFilter in PeopleList
Background context: This example demonstrates how to integrate the `SelectFilter` component into an existing component (`PeopleList`) to filter people by city.

:p How does the `PeopleList.razor` file apply the `SelectFilter` component?
??x
In the `PeopleList.razor` file, the `SelectFilter` component is applied as follows:

```razor
<div class="form-group">
    <label for="city">City</label>
    <select name="city" class="form-control" @bind="SelectedCity">
        <option disabled selected value="">Select City</option>
        @foreach (string city in Cities ?? Enumerable.Empty<string>()) {
            <option value="@city" selected="@(city == SelectedCity)">
                @city
            </option>
        }
    </select>
</div>

<SelectFilter />
@code {
    [Inject]
    public DataContext? Context { get; set; }

    public IEnumerable<Person>? People => 
        Context?.People.Include(p => p.Department)
                       .Include(p => p.Location);

    public IEnumerable<string>? Cities =>
        Context?.Locations.Select(l => l.City);

    public string SelectedCity { get; set; } = string.Empty;

    public string GetClass(string? city) => 
        SelectedCity == city ? "bg-info text-white" : "";
}
```

The `SelectFilter` component is added directly within the template, and its properties are managed through the code-behind.
x??

---
#### Cascading Parameters for Configuration
Background context: This section explains how to use cascading parameters to distribute configuration settings widely.

:p What is a cascading parameter used for?
??x
A cascading parameter is used to pass values down to nested components, allowing them to access and utilize the same data. For example:

```razor
<CascadingParameter Name="CityFilter" Value="selectedCity">
    <PeopleList />
</CascadingParameter>
@code {
    public string selectedCity { get; set; } = "New York";
}
```

Here, `selectedCity` is passed down to any nested components that use the `CityFilter` parameter.
x??

---
#### Handling Connection Errors and Unhandled Errors
Background context: This part covers how to respond to connection errors and unhandled errors using Blazor's error handling mechanisms.

:p How do you handle connection errors in Blazor?
??x
Blazor provides the `<Connection>` element and related classes to manage connection states. For example:

```razor
<Connection Lost="OnConnectionLost" Reestablished="OnReconnection">
    <p>Application is connected.</p>
</Connection>

@code {
    private void OnConnectionLost()
    {
        // Handle lost connection logic here.
    }

    private void OnReconnection()
    {
        // Handle reconnected logic here.
    }
}
```

The `Lost` and `Reestablished` events are used to handle connection loss and re-establishment, respectively.
x??

---

#### Configuring Components with Attributes
Background context: In Razor components, attributes are used to configure component properties. This allows for flexible and reusable components that can be tailored to specific use cases through configuration.

:p How do you configure a component using attributes in Razor components?
??x
Attributes are added to the HTML element of the parent component to pass values down to the child component. These attribute values correspond to C# properties defined with the `[Parameter]` attribute within the child component.
```csharp
// Example in SelectFilter.razor
[Parameter]
public IEnumerable<string> Values { get; set; } = Enumerable.Empty<string>();
```
x??

---

#### Parent-Child Component Relationship
Background context: Components can delegate layout responsibilities to each other, forming a parent-child relationship where the parent component manages overall layout and functionality while delegating specific tasks to child components.

:p How does combining components using attributes affect their layout?
??x
When you combine components by adding an attribute like `<SelectFilter values="@Cities" title="City" />` in PeopleList.razor, it integrates the SelectFilter component into the parent's content. The `title` and `values` attributes are used to configure properties within the SelectFilter component, allowing for dynamic behavior based on the context.
```csharp
// Example of integrating components
<SelectFilter values="@Cities" title="City" />
```
x??

---

#### Using Attribute Splatting
Background context: When there are many configuration settings that need to be passed down to a child component, using individual properties can become error-prone. To handle bulk configurations efficiently, attribute splatting allows multiple attributes not matched by other properties to be captured in one property.

:p How does attribute splatting work?
??x
Attribute splatting uses the `@attributes` expression within the child component’s code to capture any unmatched attributes from the parent element. This is achieved by setting the `[Parameter]` attribute with `CaptureUnmatchedValues = true`. For instance, in SelectFilter.razor:
```csharp
// Example of using attribute splatting
[Parameter(CaptureUnmatchedValues = true)]
public Dictionary<string, object>? Attrs { get; set; }
```
x??

---

#### Combining Components to Create Reusable Features
Background context: By creating reusable components and configuring them through attributes, you can build complex user interfaces in a modular way. This approach enhances maintainability and reusability of code.

:p How does combining SelectFilter with PeopleList achieve better modularity?
??x
Combining SelectFilter with PeopleList allows the filtering functionality to be isolated into its own component (`SelectFilter`), making it reusable throughout the application. The `PeopleList` can then focus on displaying data, while `SelectFilter` handles user input for filtering. This separation of concerns makes both components easier to manage and modify independently.
```razor
// Example in PeopleList.razor using SelectFilter
<SelectFilter values="@Cities" title="City" />
```
x??

---

#### Bulk Configuration Settings via a Single Property
Background context: For complex configurations, using individual properties for each setting can be cumbersome. A single property that captures all unmatched attributes simplifies the configuration process.

:p How does configuring components with bulk settings work?
??x
You define a property to capture any unmatched attributes from the parent component’s element. This is done by adding `CaptureUnmatchedValues = true` to the `[Parameter]` attribute and defining the property as a dictionary (`Dictionary<string, object>`). For example:
```csharp
// Example in SelectFilter.razor using bulk configuration
[Parameter(CaptureUnmatchedValues = true)]
public Dictionary<string, object>? Attrs { get; set; }
```
x??

---

#### Applying Configuration Attributes to Components
Background context: To integrate a component like `SelectFilter` into another (`PeopleList`), you use attributes within the parent’s HTML element. These attributes correspond to properties in the child component.

:p How do you add configuration attributes to a component?
??x
You specify configuration values as attributes on the `<SelectFilter>` element, such as `title` and `values`. For example:
```razor
// Example of adding configuration attributes
<SelectFilter title="City" values="@Cities" />
```
x??

---

#### Configuring Components in Controllers or Razor Pages
Background context: When components are used within controller views or Razor pages, they can be configured using similar attribute-based methods as described above.

:p How do you configure a component when it is applied using the `component` element?
??x
You use attributes that start with `param-` followed by the property name to pass values down to the component. For instance:
```razor
// Example of configuring components in PeopleList
<SelectFilter title="City" values="@Cities" />
```
The `title` and `values` attributes configure properties within the `SelectFilter` component.
x??

---

---
#### Using Attributes in Blazor Server Components
Background context: In this section, we learn how to use attributes to pass parameters and configuration settings to custom Blazor components. This is useful for modifying component behavior without changing the component itself.

:p What are some ways to configure a Blazor component using attributes?
??x
Attributes allow you to pass values and configurations directly into a Blazor component. For example, in Listing 34.10, `param-itemcount` and `param-selecttitle` are used to set the `ItemCount` and `SelectTitle` properties of the `PeopleList` component.

```html
<component type="typeof(Advanced.Blazor.PeopleList)" render-mode="Server" 
           param-itemcount="5" param-selecttitle="@("Location")" />
```

x??
---
#### Handling Custom Events in Blazor Components
Background context: When a Blazor component needs to notify its parent of user actions, custom events are used. In the `SelectFilter` component, a custom event is created to handle changes made by users.

:p How do you define and use a custom event in a Blazor component?
??x
To define a custom event, you add a property with the type `EventCallback<T>` where `T` represents the data type that will be passed. The parent can then register an event handler for this custom event.

In Listing 34.11, `CustomEvent` is defined as follows:
```csharp
[Parameter]
public EventCallback<string> CustomEvent { get; set; }
```

The `HandleSelect` method invokes the `CustomEvent` when a selection is made:
```csharp
public async Task HandleSelect(ChangeEventArgs e) {
    SelectedValue = e.Value as string;
    await CustomEvent.InvokeAsync(SelectedValue);
}
```

x??
---

---
#### Handling Custom Events in Blazor Components
Background context: In this scenario, we are dealing with custom events in a Blazor application. We have two components interacting through an event mechanism. The `SelectFilter` component triggers a custom event when its selected value changes, and the `PeopleList` component listens for this event to update the UI.

:p How does the `SelectFilter` component notify its parent when a selection is made?
??x
The `SelectFilter` component notifies its parent by invoking a custom event. This is done through an `EventCallback<string>` property that the parent can subscribe to and handle appropriately.
```razor
<SelectFilter values="@Cities" title="@SelectTitle" CustomEvent="HandleCustom" />
```

```csharp
public void HandleCustom(string newValue) {
    SelectedCity = newValue;
}
```
x??

---
#### EventCallback<T> in Blazor Components
Background context: The `EventCallback<T>` is a way for child components to notify their parent about events. It allows the parent to react to changes in the child component's state or user interactions.

:p How does `HandleCustom` handle the custom event from the `SelectFilter`?
??x
The `HandleCustom` method updates the `SelectedCity` property with the new value received from the `SelectFilter`. This change triggers a re-render of the parent component, updating the UI based on the selected city.
```razor
public void HandleCustom(string newValue) {
    SelectedCity = newValue;
}
```
x??

---
#### Data Binding and Custom Events in Blazor Components
Background context: The `PeopleList` component binds to the data provided by the `SelectFilter` component. When a selection is made, it updates its internal state (`SelectedCity`) through an event handler.

:p How does the `PeopleList` component use custom events?
??x
The `PeopleList` component uses the `CustomEvent` attribute of the `SelectFilter` to subscribe to and handle the selected value change event. When this event is triggered, it updates the `SelectedCity` property.
```razor
<SelectFilter values="@Cities" title="@SelectTitle" CustomEvent="HandleCustom" />
```

```csharp
public void HandleCustom(string newValue) {
    SelectedCity = newValue;
}
```
x??

---
#### Conditional Styling in Razor Components
Background context: The `PeopleList` component uses conditional styling to highlight rows based on the selected city. This is achieved using a custom CSS class that changes depending on whether the row's city matches the selected city.

:p How does the `GetClass` method determine which style class to apply?
??x
The `GetClass` method returns a CSS class name based on whether the city of the person in the table row matches the currently selected city. If they match, it applies the "bg-info text-white" class for highlighting; otherwise, it returns an empty string.
```razor
public string GetClass(string? city) => 
    SelectedCity == city ? "bg-info text-white" : "";
```
x??

---
#### Injecting Services into Components in Blazor
Background context: The `PeopleList` component needs access to a data context (`DataContext`) provided by the application. This is achieved through dependency injection, which allows components to request services via properties decorated with `[Inject]`.

:p How does the `PeopleList` component get its data?
??x
The `PeopleList` component gets its data from a service injected into it using the `[Inject]` attribute. The `Context` property is used to access this service, which provides necessary data such as people and locations.
```razor
[Inject]
public DataContext? Context { get; set; }
```

```csharp
public IEnumerable<Person>? People => 
    Context?.People.Include(p => p.Department).Include(p => p.Location).Take(ItemCount);
```
x??

---

#### Update to SelectFilter Component
Background context: The provided code snippet updates the `SelectFilter` component to include properties for binding, making it more flexible and reusable. This update is essential for creating dynamic UI elements that can be used across different parts of an application.

:p What changes were made to the `SelectFilter` component to enhance its functionality?
??x
The changes included adding parameters for `Values`, `SelectedValue`, and `Title`. The `@onchange` event handler was added to handle value selection, and the `SelectedValueChanged` property was introduced as an EventCallback. This setup allows the parent component to bind to and control the selected value of the dropdown.

```razor
<div class="form-group">
    <label for="select-@Title">@Title</label>
    <select name="select-@Title" class="form-control"
            @onchange="HandleSelect" value="@SelectedValue">
        <option disabled selected value="">Select @Title</option>
        @foreach (string val in Values) {
            <option value="@val" selected="@(val == SelectedValue)">
                @val
            </option>
        }
    </select>
</div>

@code {
    [Parameter] public IEnumerable<string> Values { get; set; } = Enumerable.Empty<string>();
    [Parameter] public string? SelectedValue { get; set; }
    [Parameter] public string Title { get; set; } = "Placeholder";
    [Parameter(CaptureUnmatchedValues = true)] public Dictionary<string, object>? Attrs { get; set; }
    [Parameter] public EventCallback<string> SelectedValueChanged { get; set; }

    public async Task HandleSelect(ChangeEventArgs e) {
        SelectedValue = e.Value as string;
        await SelectedValueChanged.InvokeAsync(SelectedValue);
    }
}
```
x??

---

#### Using Custom Binding in PeopleList
Background context: The `PeopleList` component uses a custom binding to update the selected city value when an option is chosen from the `SelectFilter` component. This setup illustrates how child components can be used within parent components, and how data bindings work in Blazor.

:p How does the `PeopleList` component use a custom binding with the `SelectFilter` component?
??x
The `PeopleList` component uses the `@bind-SelectedValue` attribute to bind the selected city value from the `SelectFilter` component. This allows the parent component to listen for changes and update its state accordingly.

```razor
<SelectFilter values="@Cities" title="Select City" @bind-SelectedValue="SelectedCity" />
```

Here, `SelectedCity` is a property in the parent component that gets updated whenever an option is selected in the `SelectFilter`.

x??

---

#### Handling EventCallback in SelectFilter Component
Background context: The `HandleSelect` method within the `SelectFilter` component handles the change event and updates the `SelectedValue`. It also triggers the `SelectedValueChanged` EventCallback to notify the parent component of the new value.

:p What is the purpose of the `HandleSelect` method in the `SelectFilter` component?
??x
The `HandleSelect` method is used to handle the `onchange` event of the dropdown. When a user selects an option, it updates the `SelectedValue` property and invokes the `SelectedValueChanged` EventCallback with the new value. This allows the parent component to react to the selection change.

```razor
public async Task HandleSelect(ChangeEventArgs e) {
    SelectedValue = e.Value as string;
    await SelectedValueChanged.InvokeAsync(SelectedValue);
}
```

The method ensures that any parent components using this `SelectFilter` can be notified about the new selected value, facilitating two-way data binding.

x??

---

#### Defining Parameters and Properties in SelectFilter Component
Background context: The `SelectFilter` component defines several parameters and properties to make it flexible and reusable. These include `Values`, `SelectedValue`, `Title`, and `SelectedValueChanged`. Each of these serves a specific purpose, allowing the component to be easily integrated into different parts of an application.

:p What are some key parameters and properties defined in the `SelectFilter` component?
??x
Key parameters and properties defined in the `SelectFilter` component include:
- `Values`: A collection of strings representing the options for the dropdown.
- `SelectedValue`: The currently selected value from the dropdown, which can be bound by the parent component.
- `Title`: A descriptive title for the dropdown.
- `SelectedValueChanged`: An EventCallback that notifies the parent component when the selected value changes.

```razor
[Parameter] public IEnumerable<string> Values { get; set; } = Enumerable.Empty<string>();
[Parameter] public string? SelectedValue { get; set; }
[Parameter] public string Title { get; set; } = "Placeholder";
[Parameter(CaptureUnmatchedValues = true)] public Dictionary<string, object>? Attrs { get; set; }
[Parameter] public EventCallback<string> SelectedValueChanged { get; set; }
```

These parameters and properties make the component highly flexible and reusable across different parts of an application.

x??

---

