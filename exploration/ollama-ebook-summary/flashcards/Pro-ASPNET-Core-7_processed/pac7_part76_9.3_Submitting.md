# Flashcards: Pro-ASPNET-Core-7_processed (Part 76)

**Starting Chapter:** 9.3 Submitting orders. 9.3.3 Creating the controller and view

---

#### Adding a Checkout Button to Cart View
Background context: In order to allow users to complete their orders by entering shipping details, we need to add functionality to capture this information. This involves adding a button on the cart view that will redirect to an order checkout page.

:p What does the added "Checkout" button in the cart view do?
??x
The Checkout button is styled as a link and points to the `Checkout` action method of the `Order` controller, allowing users to proceed to enter their shipping details. When clicked, it directs the user to the order checkout page where they can input necessary information.

```csharp
// Cart.cshtml (SportsStore/Pages)
<div class="text-center">
    <a class="btn btn-primary" href="@Model.ReturnUrl">
        Continue shopping
    </a>
    <a class="btn btn-primary" asp-action="Checkout" 
       asp-controller="Order">
        Checkout
    </a>
</div>
```
x??

---
#### Creating the Order Model Class
Background context: The `Order` model is essential for capturing user shipping information. It includes validation attributes to ensure that required fields are filled out.

:p What are the key properties of the `Order` class?
??x
The `Order` class contains several properties including:
- `OrderID`: An integer to uniquely identify an order (set with `[BindNever]` to prevent user input).
- `Lines`: A collection of `CartLine` objects representing items in the order.
- `Name`, `Line1`, `Line2`, `Line3`, `City`, `State`, `Zip`, and `Country`: These are string fields required for shipping details, with validation attributes applied to ensure they are not empty.
- `GiftWrap`: A boolean field indicating whether gift wrapping is selected.

```csharp
// Order.cs (SportsStore/Models)
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

        // Other required fields...
    }
}
```
x??

---
#### Creating the OrderController
Background context: The `OrderController` is responsible for processing order checkout requests. It initializes an instance of the `Order` model and passes it to the view for user input.

:p What does the `Checkout` method in `OrderController` do?
??x
The `Checkout` method returns a view with an `Order` object pre-populated, allowing users to enter their shipping details.

```csharp
// OrderController.cs (SportsStore/Controllers)
using Microsoft.AspNetCore.Mvc;
using SportsStore.Models;

namespace SportsStore.Controllers {
    public class OrderController : Controller {
        public ViewResult Checkout() => View(new Order());
    }
}
```
x??

---
#### Creating the Checkout View
Background context: The `Checkout` view is where users will input their shipping details. It uses Razor syntax and Bootstrap classes to style and organize the form fields.

:p What does the `Checkout.cshtml` view look like?
??x
The `Checkout.cshtml` view contains a form with input fields for user data, including name, address lines, city, state, zip code, country, and gift wrapping options. The form uses tag helpers to bind inputs directly to properties in the `Order` model.

```csharp
// Checkout.cshtml (SportsStore/Views/Order)
@model Order

<h2>Check out now</h2>
<p>Please enter your details, and we'll ship your goods right away.</p>

<form asp-action="Checkout" method="post">
    <h3>Ship to</h3>
    <div class="form-group">
        <label>Name:</label>
        <input asp-for="Name" class="form-control" />
    </div>

    // Other address fields...

    <h3>Options</h3>
    <div class="checkbox">
        <label>
            <input asp-for="GiftWrap" /> Gift wrap these items
        </label>
    </div>

    <div class="text-center">
        <input class="btn btn-primary" type="submit"
               value="Complete Order" />
    </div>
</form>
```
x??

---

#### Adding a Property to the Database Context
Background context: The initial setup in Chapter 7 provided a solid foundation for adding properties to the database context. This allows Entity Framework Core (EF Core) to manage data access and storage.

:p How do you add a property to the `StoreDbContext` class?
??x
To add a new property, you need to modify the `StoreDbContext` class by adding a new `DbSet`. Here's how it can be done:
```csharp
public DbSet<Order> Orders => Set<Order>();
```
This line adds an `Orders` collection to the database context. EF Core will now recognize and manage `Order` objects.
x??

---

#### Creating a Database Migration
Background context: A migration is a way to update your database schema without losing existing data. This process ensures that your application's model stays in sync with the actual database.

:p How do you create a migration for storing Orders?
??x
To create a migration, use the following command from the root of the project:
```sh
dotnet ef migrations add Orders
```
This command generates a new migration file named `Orders`. The migration will automatically be applied when the application starts because it calls the `Migrate` method.
x??

---

#### Deleting and Resetting the Database
Background context: When developing, you may need to reset the database frequently to test changes or to start from a clean state. However, this should only be done during development since data loss is likely.

:p How do you delete the database in a project?
??x
To delete the database, use the following command:
```sh
dotnet ef database drop --force --context StoreDbContext
```
This command forcefully drops the database associated with `StoreDbContext`. Be cautious as this will remove all data stored within it.
x??

---

#### Creating an Order Repository Interface
Background context: A repository pattern is a design pattern that abstracts the storage mechanism of entities. It helps in separating concerns and making the code more maintainable.

:p What is the interface for the order repository?
??x
The interface `IOrderRepository` defines methods to interact with orders:
```csharp
namespace SportsStore.Models {
    public interface IOrderRepository {
        IQueryable<Order> Orders { get; }
        void SaveOrder(Order order);
    }
}
```
This interface provides a contract that must be implemented by any class handling order operations.
x??

---

#### Implementing the Order Repository with EF Core
Background context: The `EFOrderRepository` implements the repository pattern using Entity Framework Core. It handles reading and saving orders, ensuring related data is properly loaded.

:p How does the `EFOrderRepository` implement the `SaveOrder` method?
??x
The `SaveOrder` method in `EFOrderRepository` ensures that related entities are correctly handled:
```csharp
public void SaveOrder(Order order) {
    context.AttachRange(order.Lines.Select(l => l.Product));
    if (order.OrderID == 0) {
        context.Orders.Add(order);
    }
    context.SaveChanges();
}
```
- `context.AttachRange(order.Lines.Select(l => l.Product))`: Ensures that related product entities are tracked by EF Core.
- If the order ID is zero, it means this is a new order; otherwise, it updates an existing one.
x??

---

#### Loading Related Data with Include and ThenInclude
Background context: Entity Framework Core requires explicit instructions to load related data from multiple tables. This ensures efficient data retrieval without separate queries.

:p How does the `Orders` property use `Include` and `ThenInclude`?
??x
The `Orders` property uses these methods to ensure that all necessary data is loaded:
```csharp
public IQueryable<Order> Orders => context.Orders
    .Include(o => o.Lines)
    .ThenInclude(l => l.Product);
```
This line ensures that when an order is retrieved, the associated lines and their products are also fetched.
x??

---

#### Registering Services in Program.cs
Background context: The `Program.cs` file configures services used by the application. This includes setting up repositories for data access.

:p How do you register the order repository as a service?
??x
In `Program.cs`, you register the `EFOrderRepository` as a service:
```csharp
using SportsStore.Models;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddControllersWithViews();
// Registering the order repository
builder.Services.AddScoped<IOrderRepository, EFOrderRepository>();
```
This line ensures that the `IOrderRepository` is available throughout the application.
x??

---

#### Adding Services for DbContext and Repositories

Background context: To complete the cart functionality, we need to ensure that our application is properly configured with the required services. This includes setting up a database context (`StoreDbContext`) and repositories (`IOrderRepository` and `IStoreRepository`). These components are essential for handling orders and storing product data.

:p What are the key services added in this configuration?

??x
The key services added include configuring the database context to use SQL Server, setting up scoped repositories for both store operations and order processing, and adding necessary middleware for static files, sessions, and routing. Here’s a breakdown of each:

```csharp
builder.Services.AddDbContext<StoreDbContext>(opts => {
    opts.UseSqlServer(builder.Configuration["ConnectionStrings:SportsStoreConnection"]);
});
```

- This line configures the `StoreDbContext` to use SQL Server with the specified connection string.

```csharp
builder.Services.AddScoped<IOrderRepository, EFOrderRepository>();
builder.Services.AddScoped<IStoreRepository, EFStoreRepository>();
```

- These lines register scoped services for repositories. The `IOrderRepository` and `IStoreRepository` interfaces are implemented by `EFOrderRepository` and `EFStoreRepository`, respectively.

```csharp
builder.Services.AddRazorPages();
builder.Services.AddDistributedMemoryCache();
builder.Services.AddSession();
builder.Services.AddScoped<Cart>(sp => SessionCart.GetCart(sp));
```

- These lines add services for Razor Pages, distributed memory cache (for session management), and scoped `Cart` services. The last line uses a custom `SessionCart` to manage the cart data in the user's session.

```csharp
builder.Services.AddSingleton<IHttpContextAccessor, HttpContextAccessor>();
```

- This registers an HTTP context accessor as a singleton service, which can be used throughout the application to access the current HTTP request context.
x??

---

#### Completing the OrderController Constructor

Background context: The `OrderController` is responsible for processing orders. It needs to receive the necessary services (order repository and cart) through its constructor to function correctly. This ensures that all required components are available when handling order checkout processes.

:p How does the constructor of the `OrderController` ensure it has access to the required services?

??x
The `OrderController` constructor accepts two parameters: an instance of `IOrderRepository` and a `Cart`. These dependencies are provided through the service container, ensuring that the controller can interact with both the order storage and shopping cart functionalities.

```csharp
public class OrderController : Controller {
    private IOrderRepository repository;
    private Cart cart;

    public OrderController(IOrderRepository repoService, Cart cartService) {
        repository = repoService;
        cart = cartService;
    }
}
```

- The constructor initializes `repository` and `cart` with the provided services.
x??

---

#### Handling HTTP POST Requests in Checkout Action

Background context: The `Checkout` action method in `OrderController` handles both GET and POST requests. For POST, it processes order data by validating the cart contents and saving the order if valid. This ensures that only users with a non-empty cart can proceed to checkout.

:p What happens when an HTTP POST request is made to the Checkout action?

??x
When an HTTP POST request is made to the `Checkout` action in `OrderController`, it first checks whether the cart contains any items. If not, it sets up validation errors and returns a view that allows the user to correct their input. If there are items in the cart and shipping details are valid, the order is saved, and the cart is cleared.

```csharp
[HttpPost]
public IActionResult Checkout(Order order) {
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

- The method first checks the count of items in the cart. If it's zero, an error is added to `ModelState`.
- If there are items, the method copies the cart lines into the order and saves the order via the repository.
- After saving, the cart is cleared, and a redirection is performed to a "Completed" page with the order ID.
x??

---

#### Unit Testing OrderController

Background context: To ensure the `OrderController` works correctly, unit tests are necessary. These tests check various scenarios such as empty carts, invalid shipping details, and valid checkout processes. This ensures that only properly filled carts and valid data can proceed to save orders.

:p How do you test the behavior of the POST version of the Checkout method in the OrderController?

??x
To test the POST version of the `Checkout` method, a unit test is created using Moq for mocking repositories and setting up scenarios. The test verifies that an order cannot be processed with an empty cart or invalid shipping details.

```csharp
[Fact]
public void Cannot_Checkout_Empty_Cart() {
    Mock<IOrderRepository> mock = new Mock<IOrderRepository>();
    Cart cart = new Cart();
    Order order = new Order();
    OrderController target = new OrderController(mock.Object, cart);

    ViewResult? result = target.Checkout(order) as ViewResult;
    
    mock.Verify(m => m.SaveOrder(It.IsAny<Order>()), Times.Never);
    Assert.True(string.IsNullOrEmpty(result?.ViewName));
    Assert.False(result?.ViewData.ModelState.IsValid);
}
```

- The test sets up a mock `IOrderRepository`, an empty cart, and a new order.
- It then creates an instance of the `OrderController` with these mocks.
- When the `Checkout` method is called with an empty cart, it ensures that no order is saved (mock verification) and returns the default view.
x??

---

