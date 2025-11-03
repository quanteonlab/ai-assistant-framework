# Flashcards: Pro-ASPNET-Core-7_processed (Part 28)

**Starting Chapter:** 9.3 Submitting orders. 9.3.3 Creating the controller and view

---

#### Adding a Checkout Button to Cart View
Background context: The next step is to enable users to enter their shipping details and submit an order. To start, a `Checkout` button needs to be added to the cart view.

:p How do you add a `Checkout` button in the cart view?
??x
To add a `Checkout` button, you need to modify the `Cart.cshtml` file located in the `SportsStore/Pages` folder. Add an `<a>` tag with the appropriate class and link to call the `Checkout` action method of the `OrderController`. Here is how it looks:

```html
<div class="text-center">
    <a class="btn btn-primary" href="@Model.ReturnUrl">
        Continue shopping
    </a>
    <a class="btn btn-primary" asp-action="Checkout" asp-controller="Order">
        Checkout
    </a>
</div>
```

This change generates a button that, when clicked, calls the `Checkout` action method of the `OrderController`.
x??

---

#### Creating Order Model Class
Background context: The next step is to create a model class to capture shipping details from users. This involves adding validation attributes and ensuring sensitive data is not overwritten by HTTP requests.

:p What is the purpose of creating an `Order.cs` class file?
??x
The purpose of creating an `Order.cs` class file is to define a model that captures the shipping details for a customer. It includes properties like name, address lines, city, state, zip, country, and gift wrap options. Validation attributes ensure user inputs are required and correct.

Here's how the class looks:

```csharp
using System.ComponentModel.DataAnnotations;
using Microsoft.AspNetCore.Mvc.ModelBinding;

namespace SportsStore.Models {
    public class Order {
        [BindNever]
        public int OrderID { get; set; }
        
        [BindNever]
        public ICollection<CartLine> Lines { get; set; } = new List<CartLine>();
        
        [Required(ErrorMessage = "Please enter a name")]
        public string? Name { get; set; }

        // Other properties for address and country
    }
}
```

:p What is the `BindNever` attribute used for?
??x
The `BindNever` attribute prevents users from supplying values for sensitive or important model properties via HTTP requests. This stops ASP.NET Core from using values from the request to populate these properties.

Example:
```csharp
[BindNever]
public int OrderID { get; set; }
```

This means that the `OrderID` property will not be bound by the model binder during an HTTP POST or GET request.
x??

---

#### Creating Checkout Controller and View
Background context: To handle order processing, a controller named `OrderController` is needed. This controller should have a method to render a view where users can enter their shipping details.

:p How do you create the `OrderController` class?
??x
To create the `OrderController`, add a new class file called `OrderController.cs` in the `Controllers` folder and define it as follows:

```csharp
using Microsoft.AspNetCore.Mvc;
using SportsStore.Models;

namespace SportsStore.Controllers {
    public class OrderController : Controller {
        public ViewResult Checkout() => View(new Order());
    }
}
```

:p What does the `Checkout()` method do?
??x
The `Checkout()` method returns a view with an `Order` model as its view model. It initializes and passes a new `Order` object to the view, which will be used to capture user input for shipping details.

Example:
```csharp
public ViewResult Checkout() => View(new Order());
```

:p How do you create the `Checkout.cshtml` view?
??x
To create the `Checkout.cshtml` view, add a Razor view in the `Views/Order` folder. The markup should capture user inputs for shipping details and include validation attributes:

```html
@model Order

<h2>Check out now</h2>
<p>Please enter your details, and we'll ship your goods right away.</p>

<form asp-action="Checkout" method="post">
    <h3>Ship to</h3>
    <div class="form-group">
        <label>Name:</label>
        <input asp-for="Name" class="form-control" />
    </div>
    
    <h3>Address</h3>
    <div class="form-group">
        <label>Line 1:</label>
        <input asp-for="Line1" class="form-control" />
    </div>

    <!-- Other address fields -->

    <h3>Options</h3>
    <div class="checkbox">
        <label>
            <input asp-for="GiftWrap" /> Gift wrap these items
        </label>
    </div>

    <div class="text-center">
        <input class="btn btn-primary" type="submit" value="Complete Order" />
    </div>
</form>
```

:p How do you handle form submission in the `OrderController`?
??x
To handle form submission, you need to add a method in the `OrderController` that processes the POST request and saves the order details. Here is an example:

```csharp
[HttpPost]
public IActionResult Checkout([Bind("Name,Line1,Line2,Line3,City,State,Zip,Country,GiftWrap")] Order order) {
    // Save order to database or perform other actions
    return RedirectToAction("OrderSummary", "Order");
}
```

This method checks for the `Order` model and processes the form submission by saving the order details.
x??

---

#### Adding a Property to Database Context
Background context: The text explains how adding a property to the `StoreDbContext` class allows Entity Framework Core to manage orders in the database. This setup is essential for storing and retrieving order information.

:p How can you add a new property to the database context to support order processing?
??x
To add a new property to the database context, you modify the `StoreDbContext` class by adding a `DbSet<Order>` property as shown below:

```csharp
using Microsoft.EntityFrameworkCore;

namespace SportsStore.Models {
    public class StoreDbContext : DbContext {
        public StoreDbContext(DbContextOptions<StoreDbContext> options) 
            : base(options) { }
        
        public DbSet<Product> Products => Set<Product>();
        public DbSet<Order> Orders => Set<Order>(); // Added this line for order support
    }
}
```

This change enables Entity Framework Core to create a migration that adds the `Order` entity to the database schema.
x??

---

#### Creating Database Migrations
Background context: The text discusses creating and applying migrations in Entity Framework Core. These migrations help manage changes to the application's data model without losing existing data.

:p How do you create a migration for orders?
??x
To create a migration for orders, use the following command from the root of the SportsStore project:

```powershell
dotnet ef migrations add Orders
```

This command generates a new migration that updates the database schema to include order entities. The migration is applied automatically when the application starts or can be manually run using `dotnet ef database update`.
x??

---

#### Resetting the Database
Background context: During development, frequent changes to the model may lead to misalignment between migrations and the actual database schema. To resolve this, you might need to reset the database.

:p How do you delete and recreate the database in a development environment?
??x
To delete the existing database and recreate it, use the following command:

```powershell
dotnet ef database drop --force --context StoreDbContext
```

This command deletes the current database. Afterward, run this command to re-create the database and apply all migrations:

```powershell
dotnet ef database update --context StoreDbContext
```

These commands ensure that your application's data model accurately reflects the current codebase.
x??

---

#### Implementing Order Repository Interface
Background context: The text explains how to implement an `IOrderRepository` interface using Entity Framework Core. This repository provides methods for accessing and managing order data.

:p What is the purpose of implementing the `IOrderRepository` interface?
??x
The purpose of implementing the `IOrderRepository` interface is to provide a way to interact with the database for storing, retrieving, and modifying orders. By using this interface, you encapsulate the logic related to reading and writing order data, making your code more modular and testable.

Here is an example implementation of the `EFOrderRepository` class:

```csharp
using Microsoft.EntityFrameworkCore;
namespace SportsStore.Models {
    public class EFOrderRepository : IOrderRepository {
        private StoreDbContext context;

        public EFOrderRepository(StoreDbContext ctx) {
            context = ctx;
        }

        public IQueryable<Order> Orders => 
            context.Orders.Include(o => o.Lines).ThenInclude(l => l.Product);

        public void SaveOrder(Order order) {
            context.AttachRange(order.Lines.Select(l => l.Product));
            if (order.OrderID == 0) {
                context.Orders.Add(order);
            }
            context.SaveChanges();
        }
    }
}
```

This class includes methods to retrieve all orders and save new or modified orders. The `Include` and `ThenInclude` methods ensure that related data is loaded when querying the database.
x??

---

#### Attaching Related Entities
Background context: When saving an order, it's necessary to inform Entity Framework Core about related entities (like products) that should not be saved if they already exist in the database.

:p How do you prevent duplicate product entries when saving an order?
??x
When saving an order, you need to notify Entity Framework Core which products are already attached to the context. This prevents errors caused by trying to insert existing objects again.

Here is how you can attach related entities:

```csharp
public void SaveOrder(Order order) {
    context.AttachRange(order.Lines.Select(l => l.Product));
    if (order.OrderID == 0) {
        context.Orders.Add(order);
    }
    context.SaveChanges();
}
```

The `AttachRange` method ensures that Entity Framework Core recognizes the products in the order as existing entities and does not attempt to insert them again, preventing potential errors.
x??

---

#### Adding Services to Configure the Application

Background context: In this section, we are configuring services for an ASP.NET Core application. This involves setting up dependency injection (DI) and defining services that will be used throughout the application.

:p What is the purpose of adding `AddDbContext`, `AddScoped`, and `AddRazorPages` in the configuration?
??x
The purpose of these additions is to configure dependencies for the application, ensuring that they are properly initialized and available where needed. Specifically:
- `AddDbContext<StoreDbContext>` sets up a database context for managing entity framework operations.
- `AddScoped<IStoreRepository, EFStoreRepository>` registers a scoped service (repository) that will be instantiated per request.
- `AddRazorPages` enables Razor Pages support in the application.

```csharp
builder.Services.AddDbContext<StoreDbContext>(opts =>
{
    opts.UseSqlServer(builder.Configuration["ConnectionStrings:SportsStoreConnection"]);
});
builder.Services.AddScoped<IStoreRepository, EFStoreRepository>();
builder.Services.AddRazorPages();
```

x??

---
#### Configuring Middleware and Routing

Background context: The configuration of middleware and routing is necessary to define how HTTP requests are processed in the application. This involves setting up static file serving, sessions, and mapping routes to controllers.

:p What does `app.UseSession()` do in this context?
??x
`app.UseSession()` configures the session state for the application. It enables the use of sessions to maintain user-specific data across requests.

```csharp
app.UseSession();
```

x??

---
#### Completing the OrderController Class

Background context: The `OrderController` class is responsible for handling order processing in the application. To complete this controller, we need to ensure it can handle both GET and POST requests related to orders.

:p How does the `Checkout()` method handle different scenarios when an order is submitted?
??x
The `Checkout()` method handles different scenarios based on the presence of items in the cart and the validity of shipping details. It ensures that:
- If there are no items in the cart, it marks the model state as invalid.
- If the model state is valid, it processes the order by saving it to the repository and clearing the cart.

```csharp
[HttpPost]
public IActionResult Checkout(Order order)
{
    if (cart.Lines.Count() == 0) {
        ModelState.AddModelError("", "Sorry, your cart is empty.");
    }
    if (ModelState.IsValid) {
        order.Lines = cart.Lines.ToArray();
        repository.SaveOrder(order);
        cart.Clear();
        return RedirectToPage("/Completed", new { orderId = order.OrderID });
    } else {
        return View();
    }
}
```

x??

---
#### Unit Testing the OrderController

Background context: Unit testing ensures that the `OrderController` behaves as expected under different conditions. This involves creating test methods to validate various scenarios, such as processing an empty cart or providing invalid shipping details.

:p What does the first unit test (`Cannot_Checkout_Empty_Cart()`) verify?
??x
The first unit test verifies that attempting to checkout with an empty cart results in no order being saved and a view being returned where users can add items to their cart.

```csharp
[Fact]
public void Cannot_Checkout_Empty_Cart()
{
    // Arrange - create a mock repository, empty cart, and instance of the controller
    Mock<IOrderRepository> mock = new Mock<IOrderRepository>();
    Cart cart = new Cart();
    Order order = new Order();
    OrderController target = new OrderController(mock.Object, cart);

    // Act - try to checkout
    ViewResult? result = target.Checkout(order) as ViewResult;

    // Assert - no order is stored and view is returned
    mock.Verify(m => m.SaveOrder(It.IsAny<Order>()), Times.Never);
    Assert.True(string.IsNullOrEmpty(result?.ViewName));
    Assert.False(result?.ViewData.ModelState.IsValid);
}
```

x??

---

