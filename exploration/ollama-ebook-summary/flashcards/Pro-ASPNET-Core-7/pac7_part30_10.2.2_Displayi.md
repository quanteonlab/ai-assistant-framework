# Flashcards: Pro-ASPNET-Core-7_processed (Part 30)

**Starting Chapter:** 10.2.2 Displaying orders to the administrator

---

#### Blazor Layout and Navigation
Blazor uses Razor syntax to generate HTML but introduces its own directives and features. This layout renders a two-column display with product and order navigation buttons, which are created using `NavLink` elements.

:p What does the layout structure for the admin section look like?
??x
The layout consists of a two-column display where the left column contains navigation links for products and orders, while the right column displays the content based on the selected link. The `NavLink` elements apply a built-in Razor Component that changes the URL without triggering a new HTTP request.

Example:
```razor
<div class="row">
    <div class="col-3">
        <div class="nav flex-column nav-pills" role="tablist" aria-orientation="vertical">
            <NavLink class="btn btn-primary" href="/admin/products"
                     ActiveClass="btn-primary text-white"
                     Match="NavLinkMatch.Prefix">Products</NavLink>
            <NavLink class="btn btn-outline-primary" href="/admin/orders"
                     ActiveClass="btn-primary text-white"
                     Match="NavLinkMatch.Prefix">Orders</NavLink>
        </div>
    </div>
    <div class="col">
        @Body
    </div>
</div>
```
x??

---

#### Products and Orders Razor Components
To provide the administration tools, placeholder components `Products.razor` and `Orders.razor` are created within the `Pages/Admin` folder. These components specify URLs that will trigger their display.

:p What do the `Products.razor` and `Orders.razor` files contain initially?
??x
Initially, these files contain simple placeholder messages:

For `Products.razor`:
```razor
@page "/admin/products"
@page "/admin"

<h4>This is the products component</h4>
```

For `Orders.razor`:
```razor
@page "/admin/orders"

<h4>This is the orders component</h4>
```
The `@page` directives specify the URLs for which these components will be displayed.

x??

---

#### Blazor Initial Setup Verification
After setting up and adding the necessary components, it's essential to verify that everything works as expected. This involves starting ASP.NET Core and navigating to a specific URL to check if the Blazor application is functioning correctly.

:p How can you ensure that Blazor is working correctly?
??x
To ensure that Blazor is working correctly, follow these steps:
1. Start your ASP.NET Core project.
2. Navigate to `http://localhost:5000/admin`.
3. The Index Razor Page in the `Pages/Admin` folder will handle this request and include the necessary Blazor JavaScript files.
4. The JavaScript code will open a persistent HTTP connection to the server, allowing the initial Blazor content to be rendered without requiring a new HTTP request each time you navigate between pages.

Example of navigating:
1. Start your project in Visual Studio or run `dotnet run` in the terminal.
2. Open your browser and go to `http://localhost:5000/admin`.
3. You should see the navigation buttons for products and orders, and clicking on them will display the respective components without a new HTTP request.

x??

---

#### Implementing Administration Features
Now that Blazor is set up and tested, you can start implementing administration features such as viewing orders and marking them as shipped.

:p What are some of the features being implemented in this section?
??x
In this section, you will implement basic administration tools to view received orders and mark them as shipped. This involves creating or modifying Razor components within the `Pages/Admin` folder to provide these functionalities.

Example:
You might create a component like `Orders.razor` that fetches order data from a database and displays it in a table, along with buttons to mark an order as shipped.

```razor
@page "/admin/orders"

<h4>Order Management</h4>
<table class="table">
    <thead>
        <tr>
            <th>Order ID</th>
            <th>Date</th>
            <th>Status</th>
        </tr>
    </thead>
    <tbody>
        @foreach (var order in Orders)
        {
            <tr>
                <td>@order.Id</td>
                <td>@order.Date.ToShortDateString()</td>
                <td>
                    @if (order.IsShipped)
                    {
                        <span class="badge badge-success">Shipped</span>
                    }
                    else
                    {
                        <button onclick="@(() => MarkAsShipped(order))" class="btn btn-primary">Ship</button>
                    }
                </td>
            </tr>
        }
    </tbody>
</table>

@code {
    private List<Order> Orders { get; set; } = new List<Order>();

    protected override async Task OnInitializedAsync()
    {
        // Fetch orders from the database
        Orders = await OrderService.GetOrdersAsync();
    }

    private async Task MarkAsShipped(Order order)
    {
        // Update the order status and save it back to the database
        order.IsShipped = true;
        await OrderService.SaveOrderAsync(order);
    }
}
```

x??

---

#### Adding a Property to the Order Class
Background context explaining the need to add the `Shipped` property to the `Order` class. This change is part of enhancing the data model to track which orders have been shipped. The property helps in differentiating between orders that are still pending and those that have already been dispatched.

:p What new property was added to the Order class for tracking order shipping status?
??x
The new property `Shipped` was added to the `Order` class with a boolean type, allowing it to track whether an order has been shipped or not. This is important for administrative purposes and providing updated information to the user interface.

```csharp
public bool Shipped { get; set; }
```
x??

---

#### Using Entity Framework Core Migrations
Background context explaining how Entity Framework Core migrations can be used to update the database schema without manually writing SQL commands. The `dotnet ef migrations add` command is used to create a migration for adding the `Shipped` property.

:p How do you use Entity Framework Core to update the database schema after modifying the model?
??x
To update the database schema using Entity Framework Core, you can run the following command in a new PowerShell window:

```powershell
dotnet ef migrations add ShippedOrders
```

This command creates a new migration that adds the `Shipped` property to the `Order` class. When the application is started, this migration will be applied automatically by Entity Framework Core.

x??

---

#### Displaying Orders for Administrators
Background context explaining how two tables are displayed—one showing orders waiting to be shipped and another for shipped orders. This helps in managing the order status efficiently from an administrative perspective.

:p How are orders displayed to administrators?
??x
Orders are displayed using two tables: one for orders that need to be shipped, and another for those that have already been shipped. Each order is presented with a button allowing the administrator to change the shipping state.

```csharp
// Pseudocode for displaying orders
if (order.Shipped) {
    displayOrderAsShipped(order);
} else {
    displayOrderWaitingToBeShipped(order, shipmentButton);
}
```

The `shipmentButton` can trigger an action that updates the `Shipped` property to true.

x??

---

#### Iterative Development in ASP.NET Core
Background context explaining iterative development in software projects and how data models are typically extended over time as requirements evolve. This is a common practice in complex applications where the initial model may need adjustments based on user feedback or new features.

:p What does iterative development mean in the context of ASP.NET Core?
??x
Iterative development involves continuously enhancing and adapting the data model to support different features as understanding evolves throughout the project's lifecycle. For instance, adding a `Shipped` property to the `Order` class is an example of such an iterative change.

In practice, you may start with a simple initial model but find that it needs modifications over time to better fit the application’s requirements. Iterative development allows for flexibility and responsiveness to changing needs without requiring a complete redesign from scratch.

x??

---

#### Razor Component Introduction
Razor Components are a feature of Blazor that allow for server-side rendering using C# and HTML. They provide a way to create reusable UI components with logic embedded directly into the HTML, making it easier to manage complex user interfaces. The `@code` section within a Razor Component defines the properties and methods required for interactivity.

:p What is a key feature of Razor Components?
??x
Razor Components enable server-side rendering by combining C# code with HTML markup, providing a powerful way to build dynamic UIs in Blazor applications.
x??

---

#### Parameters in Razor Components
Parameters are used to pass data from the parent component to the child component. These parameters can be of various types and provide the necessary context for the component's functionality.

:p What is the purpose of using parameters in Razor Components?
??x
Parameters allow a parent component to pass configuration or data to a child component, enabling reusability and flexibility within the application.
x??

---

#### Table Display Logic
The provided code snippet demonstrates how to display a table with orders. The `@if` statement checks if there are any orders to display, and the `foreach` loops handle rendering each order and its associated lines.

:p How does the component ensure no orders are displayed when none exist?
??x
The component uses an `@if` statement to check if the `Orders` parameter has any items. If not, it displays a row indicating "No Orders" centered in the table.
x??

---

#### Button Click Handling
The button within the component triggers an event callback that sends the order ID to another part of the application when clicked.

:p What does the `@onclick` attribute do?
??x
The `@onclick` attribute sets up a click handler that invokes the `OrderSelected.InvokeAsync()` method, passing the current order's ID.
x??

---

#### EventCallback and Communication
The `EventCallback<int>` property allows for asynchronous communication between components. When the button is clicked, it sends data back to the parent component or another part of the application.

:p How does `EventCallback` facilitate communication in Blazor applications?
??x
`EventCallback` enables an event-driven approach where a child component can notify its parent (or other parts) about actions taken by the user. In this case, clicking the button triggers an asynchronous call to the parent component.
x??

---

#### Table Structure and Styling
The table is styled using Bootstrap classes such as `table`, `table-sm`, `table-striped`, and `table-bordered`. The header and body sections are structured to display order details.

:p What does the `@code` section define in this component?
??x
The `@code` section defines properties and methods that provide data, logic, and event handlers for the component. In this case, it defines parameters for title, orders, button label, and an order selected callback.
x??

---

#### Introduction to Orders Component
This section explains how to create a component that retrieves and displays order data from the database. The `Orders` component is responsible for fetching orders, categorizing them based on their shipping status, and updating the database when an order's status changes.

:p What is the purpose of the `Orders` component in the SportsStore example?
??x
The purpose of the `Orders` component is to manage and display orders from the database. It fetches all orders, categorizes them into shipped and unshipped based on their shipping status, and provides functionality to update the order's shipping status.

It uses a repository pattern for data access, ensuring that operations related to order management are encapsulated within this component. This approach helps in maintaining clean separation of concerns and ensures that the component operates independently of other user-facing components.
??x

---

#### Order Data Management
The `Orders` component manages orders by fetching them from a repository, categorizing them into shipped and unshipped, and providing methods to update their shipping status.

:p How does the `Orders` component fetch order data?
??x
The `Orders` component fetches order data using an asynchronous method called `UpdateData()`. This method is invoked during the initialization of the component (via `OnInitializedAsync()`).

```csharp
public async Task UpdateData()
{
    AllOrders = await Repository.Orders.ToListAsync();
    UnshippedOrders = AllOrders.Where(o => !o.Shipped);
    ShippedOrders = AllOrders.Where(o => o.Shipped);
}
```

This method fetches all orders from the database and then filters them into two categories: unshipped and shipped. The `AllOrders` property contains all the orders, while `UnshippedOrders` and `ShippedOrders` contain filtered results.
??x

---

#### Order Table Component
The `OrderTable` component is used to display and manage a table of orders. It takes several parameters such as `TableTitle`, `Orders`, `ButtonLabel`, and `OrderSelected`.

:p How does the `OrderTable` component handle button click events?
??x
When the user clicks a button in the `OrderTable` component, it invokes methods provided via the `OrderSelected` parameter. For example:

```razor
<OrderTable TableTitle="Unshipped Orders" 
            Orders="UnshippedOrders" 
            ButtonLabel="Ship" 
            OrderSelected="ShipOrder" />
```

The `OrderSelected` attribute is set to a method, such as `ShipOrder`. When the user clicks the "Ship" button for an unshipped order, it calls `ShipOrder(int id)` with the corresponding order ID.

```csharp
public void ShipOrder(int id) => UpdateOrder(id, true);
```

The `UpdateOrder` method updates the order's shipping status in the database:

```csharp
private void UpdateOrder(int id, bool shipValue)
{
    Order? o = Repository.Orders.FirstOrDefault(o => o.OrderID == id);
    if (o != null)
    {
        o.Shipped = shipValue;
        Repository.SaveOrder(o);
    }
}
```

This method retrieves the order from the repository and updates its `Shipped` property, then saves the changes back to the database.
??x

---

#### Component Initialization
The `Orders` component uses the `OnInitializedAsync()` method for initializing data. It ensures that the component fetches the latest order data whenever it initializes.

:p How does the `Orders` component initialize its data?
??x
The `Orders` component initializes its data by overriding the `OnInitializedAsync()` method and calling the `UpdateData()` method:

```csharp
protected async override Task OnInitializedAsync()
{
    await UpdateData();
}
```

This ensures that when the component is first rendered, it fetches all orders from the repository and categorizes them into shipped and unshipped states. This initialization happens automatically whenever the user navigates to the `/admin/orders` page.
??x

---

#### Repository Pattern Usage
The `Orders` component uses a repository pattern for data access. The repository provides a clean separation between the business logic of managing orders and the underlying data storage.

:p What is the role of the repository in the `Orders` component?
??x
The repository acts as an abstraction layer between the `Orders` component and the database. It encapsulates the data retrieval, manipulation, and persistence operations. The `Orders` component interacts with the repository to fetch orders and update their shipping status.

```csharp
public IOrderRepository Repository => Service;
```

This property ensures that the `Orders` component has its own instance of the repository, which is different from other components displayed to the same user. This separation helps in maintaining clean code and prevents data access conflicts.
??x

---

---
#### CRUD Interface Implementation for Product Catalog Management
Background context: This section discusses implementing a CRUD (Create, Read, Update, Delete) interface using Blazor for managing products in an e-commerce application. The implementation involves extending the repository to include methods that allow creation, modification, and deletion of product objects.
:p What is the purpose of adding `CreateProduct`, `DeleteProduct`, and `SaveProduct` methods to the `IStoreRepository` interface?
??x
The purpose is to provide a clear separation of concerns for data manipulation. These methods enable the application to perform CRUD operations on products stored in the database through an interface, making it easier to manage these operations independently.
```csharp
namespace SportsStore.Models {
    public interface IStoreRepository {
        // Existing properties and methods...
        
        void CreateProduct(Product p);
        void DeleteProduct(Product p);
        void SaveProduct(Product p);
    }
}
```
x??
---
#### Entity Framework Core Repository Implementation
Background context: The `EFStoreRepository` class implements the `IStoreRepository` interface, providing concrete implementations for CRUD operations using Entity Framework Core. This ensures that data persistence is managed through a database context.
:p How does the `EFStoreRepository` class implement the `CreateProduct`, `DeleteProduct`, and `SaveProduct` methods?
??x
The `EFStoreRepository` class implements these methods by leveraging Entity Framework Core's capabilities to manage changes in the database context. For `CreateProduct`, it adds a new product to the context, which will be saved later. For `DeleteProduct`, it removes an existing product from the context and saves the changes. The `SaveProduct` method is used to save any pending changes made to the entity framework context.
```csharp
namespace SportsStore.Models {
    public class EFStoreRepository : IStoreRepository {
        private StoreDbContext context;
        
        public EFStoreRepository(StoreDbContext ctx) { 
            context = ctx; 
        }

        // Existing implementations...
        
        public void CreateProduct(Product p) {
            context.Add(p);
            context.SaveChanges();
        }

        public void DeleteProduct(Product p) {
            context.Remove(p);
            context.SaveChanges();
        }

        public void SaveProduct(Product p) {
            context.SaveChanges();
        }
    }
}
```
x??
---
#### Validation Attributes in Data Models
Background context: Adding validation attributes to the `Product` data model ensures that user-provided values are validated before they are stored or updated. This is crucial for maintaining data integrity and providing meaningful feedback to users.
:p What changes were made to the `Product` class to include validation?
??x
The following validation attributes were added to the `Product` class:
- `[Required]`: Ensures that certain fields cannot be empty.
- `[Range(0.01, double.MaxValue)]`: Ensures that the price is a positive number.
- `[Column(TypeName = "decimal(8, 2)")]`: Specifies the database column type for the `Price` field.

```csharp
using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;

namespace SportsStore.Models {
    public class Product {
        // Existing properties...
        
        [Required(ErrorMessage = "Please enter a product name")]
        public string Name { get; set; } = String.Empty;
        
        [Required(ErrorMessage = "Please enter a description")]
        public string Description { get; set; } = String.Empty;
        
        [Required]
        [Range(0.01, double.MaxValue, ErrorMessage = "Please enter a positive price")]
        [Column(TypeName = "decimal(8, 2)")] 
        public decimal Price { get; set; }
        
        [Required(ErrorMessage = "Please specify a category")]
        public string Category { get; set; } = String.Empty;
    }
}
```
x??
---
#### Blazor List Component for Products
Background context: The `Products.razor` component is responsible for displaying a list of products and providing navigation links to their details, edit pages, and creation page. This component uses `NavLink` components to handle navigation based on the product's ID.
:p What does the `Products.razor` component do?
??x
The `Products.razor` component displays a table of products with columns for ID, Name, Category, Price, and Actions (Details and Edit). It also includes a button to navigate to the create page where new products can be added. The component uses LINQ to filter and display data asynchronously.
```csharp
@page "/admin/products"
@page "/admin"
@inherits OwningComponentBase<IStoreRepository>

<table class="table table-sm table-striped table-bordered">
    <thead>
        <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Category</th>
            <th>Price</th>
            <td />
        </tr>
    </thead>
    <tbody>
        @if (ProductData?.Count() > 0) {
            @foreach (Product p in ProductData) {
                <tr>
                    <td>@p.ProductID</td>
                    <td>@p.Name</td>
                    <td>@p.Category</td>
                    <td>@p.Price.ToString("c")</td>
                    <td>
                        <NavLink class="btn btn-info btn-sm" 
                                 href="@GetDetailsUrl(p.ProductID ?? 0)">
                            Details
                        </NavLink>
                        <NavLink class="btn btn-warning btn-sm" 
                                 href="@GetEditUrl(p.ProductID ?? 0)">
                            Edit
                        </NavLink>
                    </td>
                </tr>
            }
        } else {
            <tr>
                <td colspan="5" class="text-center">No Products</td>
            </tr>
        }
    </tbody>
</table>

<NavLink class="btn btn-primary" href="/admin/products/create">
    Create
</NavLink>

@code {
    public IStoreRepository Repository => Service;
    public IEnumerable<Product> ProductData { get; set; } = Enumerable.Empty<Product>();
    
    protected async override Task OnInitializedAsync() {
        await UpdateData();
    }
    
    public async Task UpdateData() {
        ProductData = await Repository.Products.ToListAsync();
    }
    
    public string GetDetailsUrl(long id) => 
        $"/admin/products/details/{id}";
    
    public string GetEditUrl(long id) => 
        $"/admin/products/edit/{id}";
}
```
x??

#### Details Component Overview
Background context: The `Details.razor` component is used to display details of a single product from the database. It retrieves the product data based on the provided ID and displays it in a tabular format.

:p What does the `Details.razor` component do?
??x
The `Details.razor` component displays detailed information about a specific product, including its ID, name, description, category, and price. The component uses dependency injection to access an implementation of the `IStoreRepository` interface to fetch the product data from the database.

```csharp
@page "/admin/products/details/{id:long}"
@inherits OwningComponentBase<IStoreRepository>

<h3 class="bg-info text-white text-center p-1">Details</h3>
<table class="table table-sm table-bordered table-striped">
    <tbody>
        <tr><th>ID</th><td>@Product?.ProductID</td></tr>
        <tr><th>Name</th><td>@Product?.Name</td></tr>
        <tr><th>Description</th><td>@Product?.Description</td></tr>
        <tr><th>Category</th><td>@Product?.Category</td></tr>
        <tr><th>Price</th><td>@Product?.Price.ToString("C")</td></tr>
    </tbody>
</table>

<NavLink class="btn btn-warning" href="@EditUrl">Edit</NavLink>
<NavLink class="btn btn-secondary" href="/admin/products">Back</NavLink>

@code {
    [Inject]
    public IStoreRepository? Repository { get; set; }

    [Parameter]
    public long Id { get; set; }

    public Product? Product { get; set; }

    protected override void OnParametersSet()
    {
        Product = Repository?.Products.FirstOrDefault(p => p.ProductID == Id);
    }

    public string EditUrl => $"admin/products/edit/{Product?.ProductID}";
}
```
x??

---

#### Editor Component Overview
Background context: The `Editor.razor` component is used both to create and edit product data. It uses Blazor's built-in form handling capabilities to validate and save the changes.

:p What does the `Editor.razor` component do?
??x
The `Editor.razor` component allows users to either create a new product or edit an existing one by rendering a form with input fields for product details like name, description, category, and price. It also handles data validation and saving changes back to the database.

```csharp
@page "/admin/products/edit/{id:long}"
@page "/admin/products/create"
@inherits OwningComponentBase<IStoreRepository>

<style>
    div.validation-message { color: rgb(220, 53, 69); font-weight: 500 }
</style>

<h3 class="bg-@ThemeColor text-white text-center p-1">@TitleText a Product </h3>

<EditForm Model="Product" OnValidSubmit="SaveProduct">
    <DataAnnotationsValidator />
    
    @if(Product.ProductID.HasValue && Product.ProductID.Value != 0)
    {
        <div class="form-group">
            <label>ID</label>
            <input class="form-control" disabled value="@Product.ProductID" />
        </div>
    }

    <div class="form-group">
        <label>Name</label>
        <ValidationMessage For="() => Product.Name" />
        <InputText class="form-control" @bind-Value="Product.Name" />
    </div>

    <div class="form-group">
        <label>Description</label>
        <ValidationMessage For="() => Product.Description" />
        <InputText class="form-control" @bind-Value="Product.Description" />
    </div>

    <div class="form-group">
        <label>Category</label>
        <ValidationMessage For="() => Product.Category" />
        <InputText class="form-control" @bind-Value="Product.Category" />
    </div>

    <div class="form-group">
        <label>Price</label>
        <ValidationMessage For="() => Product.Price" />
        <InputNumber class="form-control" @bind-Value="Product.Price" />
    </div>

    <div class="mt-2">
        <button type="submit" class="btn btn-@ThemeColor">Save</button>
        <NavLink class="btn btn-secondary" href="/admin/products">Cancel</NavLink>
    </div>
</EditForm>

@code {
    public IStoreRepository Repository => Service;
    
    [Inject]
    public NavigationManager? NavManager { get; set; }

    [Parameter]
    public long Id { get; set; } = 0;

    public Product Product { get; set; } = new Product();

    protected override void OnParametersSet()
    {
        if (Id != 0)
        {
            Product = Repository.Products.FirstOrDefault(p => p.ProductID == Id) ?? new();
        }
    }

    public void SaveProduct()
    {
        if (Id == 0)
        {
            Repository.CreateProduct(Product);
        }
        else
        {
            Repository.SaveProduct(Product);
        }
        
        NavManager?.NavigateTo("/admin/products");
    }

    public string ThemeColor => Id == 0 ? "primary" : "warning";
    public string TitleText => Id == 0 ? "Create" : "Edit";
}
```
x??

---

#### Dependency Injection in Components
Background context: The `Details.razor` and `Editor.razor` components use dependency injection to access the `IStoreRepository` interface, which provides methods for interacting with the database.

:p How do these components use dependency injection?
??x
These components use the `[Inject]` attribute to declare that they require an implementation of the `IStoreRepository` interface. This allows them to interact with the underlying data storage and retrieve or modify product information as needed.

```csharp
@page "/admin/products/details/{id:long}"
@inherits OwningComponentBase<IStoreRepository>

// Example usage within OnParametersSet method:
Product = Repository?.Products.FirstOrDefault(p => p.ProductID == Id);
```
x??

---

#### Form Validation in Blazor
Background context: The `Editor.razor` component uses built-in validation mechanisms provided by Blazor to ensure that form data is valid before it is submitted.

:p How does the `Editor.razor` handle form validation?
??x
The `Editor.razor` component handles form validation using the `<EditForm>`, `<DataAnnotationsValidator>`, and `<ValidationMessage>` components. When a user submits the form, only if the data entered into the form conforms to the rules defined by the validation attributes will the `OnValidSubmit` method be invoked.

```csharp
<EditForm Model="Product" OnValidSubmit="SaveProduct">
    <DataAnnotationsValidator />
    // Form fields with ValidationMessage and Input components
</EditForm>
```
x??

---

