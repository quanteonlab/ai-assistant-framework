# Flashcards: Pro-ASPNET-Core-7_processed (Part 74)

**Starting Chapter:** 8.2.5 Implementing the cart feature

---

#### HTTP POST Method for Form Submission
Background context: In web applications, forms are often used to send data from a client (browser) to a server. The `method` attribute of the `<form>` element determines how the form data is sent to the server. Typically, developers use the `POST` method when adding items to a shopping cart or performing other operations that should not be idempotent (i.e., each request must have a unique effect).

:p What is the purpose of using the POST method for form submission in a shopping cart application?
??x
The purpose of using the POST method is to ensure that actions like adding an item to a shopping cart are handled as individual, unique requests. This prevents accidental multiple submissions and ensures that each request modifies the state correctly.

```html
<form action="/add-to-cart" method="post">
    <!-- form fields -->
</form>
```
x??

---

#### Enabling Sessions in ASP.NET Core
Background context: Sessions allow storing user-specific data across multiple HTTP requests. In ASP.NET, session state can be stored using various mechanisms such as memory, cookies, or distributed cache. For simplicity and ease of implementation, the `AddDistributedMemoryCache` method stores session data in memory.

:p How does enabling sessions with `AddDistributedMemoryCache` work in an ASP.NET application?
??x
Enabling sessions with `AddDistributedMemoryCache` sets up an in-memory store for session data. This means that when a user makes multiple requests, the application can maintain state information about the user's session.

In the `Program.cs` file:

```csharp
builder.Services.AddDistributedMemoryCache();
```

This line of code adds the distributed memory cache service to the service collection, allowing session management to be handled in-memory. The session system then automatically associates requests with sessions when they arrive from the client by calling `app.UseSession()`.

```csharp
app.UseStaticFiles();
app.UseSession();  // Registers middleware for handling sessions
```

The `UseSession` method registers the necessary middleware so that sessions can be managed during request processing.
x??

---

#### Implementing Cart Features in SportsStore Application
Background context: The `Cart` and `CartLine` classes are used to manage shopping cart functionality. These classes allow adding, removing items from the cart, computing the total value of items, and clearing the entire cart.

:p How does the `Cart` class manage items in a user's shopping cart?
??x
The `Cart` class manages items by using a list of `CartLine` objects to store product information along with their quantities. It provides methods to add new items, remove existing ones, calculate the total value, and clear the entire cart.

```csharp
public class Cart {
    public List<CartLine> Lines { get; set; } = new List<CartLine>();

    public void AddItem(Product product, int quantity) {
        CartLine? line = Lines
            .Where(p => p.Product.ProductID == product.ProductID)
            .FirstOrDefault();
        if (line == null) {
            Lines.Add(new CartLine { 
                Product = product, 
                Quantity = quantity 
            });
        } else {
            line.Quantity += quantity;
        }
    }

    public void RemoveLine(Product product) => 
        Lines.RemoveAll(l => l.Product.ProductID 
            == product.ProductID);

    public decimal ComputeTotalValue() => 
        Lines.Sum(e => e.Product.Price * e.Quantity);

    public void Clear() => 
        Lines.Clear();
}

public class CartLine {
    public int CartLineID { get; set; }
    public Product Product { get; set; } = new();
    public int Quantity { get; set; }
}
```

- `AddItem`: Adds a product to the cart with the specified quantity. If the product already exists in the cart, it increases its quantity.
- `RemoveLine`: Removes all instances of a specific product from the cart.
- `ComputeTotalValue`: Calculates the total value of all items in the cart by summing up the price of each item multiplied by its quantity.
- `Clear`: Empties the entire cart.

```csharp
cart.AddItem(product, 1); // Adds one unit of product to the cart
cart.RemoveLine(product); // Removes all units of product from the cart
decimal totalValue = cart.ComputeTotalValue(); // Computes the total value of items in the cart
cart.Clear(); // Clears the entire cart
```

The `Cart` class is used to manage shopping cart functionality by adding, removing, and computing the values of products in a user's cart.
x??

---
#### Adding New Items to Cart
Background context: The `Cart` class handles adding new items to the shopping cart. When a product is added for the first time, it should create a new `CartLine`. This ensures that each unique product has its own line item.
:p What happens when you add an item to the cart for the first time?
??x
When you add an item to the cart for the first time, a new `CartLine` is created in the `Lines` collection. The `AddItem` method checks if there's already a line with the same product ID; if not, it adds a new one.
```csharp
public class Cart {
    // Assume AddItem method exists here
    public void AddItem(Product product, int quantity) {
        bool existingLine = Lines.Any(l => l.Product.ProductID == product.ProductID);
        
        if (!existingLine) {
            Lines.Add(new CartLine { Product = product, Quantity = quantity });
        } else {
            // Logic to handle existing line
        }
    }
}
```
x??
---
#### Incrementing Existing Items in Cart
Background context: When a customer adds the same product again, the `Cart` should increment the quantity of the existing `CartLine` instead of creating a new one. This ensures that the total count of items is correctly updated.
:p What happens when you add an item to the cart that already exists?
??x
When adding an item to the cart that already exists, the `Cart` increments the quantity of the existing `CartLine`. The `AddItem` method first checks if a line with the same product ID exists. If it does, the quantity is incremented.
```csharp
public class Cart {
    // Assume AddItem method exists here
    public void AddItem(Product product, int quantity) {
        bool existingLine = Lines.Any(l => l.Product.ProductID == product.ProductID);
        
        if (existingLine) {
            var line = Lines.FirstOrDefault(l => l.Product.ProductID == product.ProductID);
            line.Quantity += quantity;
        } else {
            // Logic to handle new line
        }
    }
}
```
x??
---
#### Removing Items from Cart
Background context: The `Cart` class needs to support the removal of items. This is implemented using the `RemoveLine` method, which removes a specific product from the cart.
:p How can you remove an item from the cart?
??x
To remove an item from the cart, the `RemoveLine` method is used. It finds the `CartLine` with the specified product and removes it if found. The `RemoveLine` method checks for the presence of a line in the `Lines` collection.
```csharp
public class Cart {
    // Assume RemoveLine method exists here
    public void RemoveLine(Product product) {
        var line = Lines.FirstOrDefault(l => l.Product.ProductID == product.ProductID);
        
        if (line != null) {
            Lines.Remove(line);
        }
    }
}
```
x??
---

---
#### Testing Cart Total Calculation
Background context: The scenario involves testing a shopping cart system where the total cost of items needs to be calculated correctly. This is part of ensuring the functionality and correctness of the shopping cart feature.

:p How can we test the calculation of the total cost in the shopping cart?
??x
To test the calculation of the total cost, you need to set up specific scenarios with known products and quantities, add these items to the cart, compute the total value, and then assert that the computed total matches the expected result.

```csharp
[Fact]
public void Calculate_Cart_Total()
{
    // Arrange - create some test products
    Product p1 = new Product { ProductID = 1, Name = "P1", Price = 100M };
    Product p2 = new Product { ProductID = 2, Name = "P2", Price = 50M };

    // Arrange - create a new cart
    Cart target = new Cart();

    // Act
    target.AddItem(p1, 1);
    target.AddItem(p2, 1);
    target.AddItem(p1, 3);
    decimal result = target.ComputeTotalValue();

    // Assert
    Assert.Equal(450M, result);
}
```
x??

---
#### Clearing Cart Contents
Background context: The requirement is to ensure that the cart contents are properly cleared when a reset operation is performed. This involves adding items to the cart and then verifying that all items are removed upon calling the `Clear()` method.

:p How can we test if the cart's contents are cleared correctly after resetting?
??x
To test this, you need to set up initial conditions with some products in the cart, reset the cart, and then verify that there are no items left in the cart.

```csharp
[Fact]
public void Can_Clear_Contents()
{
    // Arrange - create some test products
    Product p1 = new Product { ProductID = 1, Name = "P1", Price = 100M };
    Product p2 = new Product { ProductID = 2, Name = "P2", Price = 50M };

    // Arrange - create a new cart
    Cart target = new Cart();

    // Arrange - add some items
    target.AddItem(p1, 1);
    target.AddItem(p2, 1);

    // Act - reset the cart
    target.Clear();

    // Assert
    Assert.Empty(target.Lines);
}
```
x??

---
#### Extending ISession for Cart Storage
Background context: The `Cart` object needs to be stored in session state. However, since ASP.NET Core session state only supports int, string, and byte[] values, we need to extend the `ISession` interface with methods that can serialize and deserialize `Cart` objects.

:p How do you define extension methods for serializing and deserializing Cart objects in ISession?
??x
To define these methods, you create an extension class called `SessionExtensions`. Here are the two main methods:

```csharp
using System.Text.Json;

namespace SportsStore.Infrastructure {
    public static class SessionExtensions {
        // Method to set a Cart object as JSON in session state
        public static void SetJson(this ISession session,
                                   string key, 
                                   object value) {
            session.SetString(key, JsonSerializer.Serialize(value));
        }

        // Method to get a Cart object from session state as JSON
        public static T? GetJson<T>(this ISession session, string key) {
            var sessionData = session.GetString(key);
            return sessionData == null ? default(T) : JsonSerializer.Deserialize<T>(sessionData);
        }
    }
}
```

These methods use `JsonSerializer` to convert the `Cart` object into a JSON string and back. This allows you to store complex objects in the session state.
x??

---

#### Razor Pages and Cart Implementation
Razor Pages is a feature of ASP.NET Core that allows you to combine HTML content, Razor expressions, and server-side logic into a single file. This enables the creation of interactive web pages with minimal setup.

:p What is the purpose of using Razor Pages in web development?
??x
The purpose of using Razor Pages is to simplify web application development by allowing developers to write both HTML markup and C# code within a single file, making it easier to handle server-side logic and user interactions.
x??

---

#### CartModel Class for Handling Requests
In the provided text, `Cart.cshtml.cs` defines a page model class named `CartModel`. This class handles requests related to the shopping cart functionality. It uses dependency injection to get an instance of `IStoreRepository`, which provides data access and business logic.

:p What is the purpose of the `CartModel` class in the provided context?
??x
The purpose of the `CartModel` class is to handle HTTP GET and POST requests related to the shopping cart. It manages operations such as displaying the current cart contents, adding items to the cart, and updating the session with cart data.
x??

---

#### OnGet Method for Handling HTTP GET Requests
The `OnGet` method in the `CartModel` class is responsible for handling HTTP GET requests. This method sets up the initial state of the cart before rendering the cart view.

:p What does the `OnGet` method do when invoked?
??x
When the `OnGet` method is invoked, it initializes or retrieves the current cart from session storage and sets the `ReturnUrl` property to the provided return URL. If no return URL is provided, it defaults to "/".

```csharp
public void OnGet(string returnUrl)
{
    ReturnUrl = returnUrl ?? "/";
    Cart = HttpContext.Session.GetJson<Cart>("cart") 
          ?? new Cart();
}
```
x??

---

#### OnPost Method for Handling HTTP POST Requests
The `OnPost` method in the `CartModel` class handles HTTP POST requests, specifically adding products to the cart. It checks if the product exists and then updates the cart accordingly.

:p What does the `OnPost` method do when invoked?
??x
When the `OnPost` method is invoked, it first retrieves the product with the given `productId`. If the product exists, it adds an item to the cart with a quantity of 1. It then saves the updated cart back into session storage and redirects the user back to the specified return URL.

```csharp
public IActionResult OnPost(long productId, string returnUrl)
{
    Product? product = repository.Products 
                      .FirstOrDefault(p => p.ProductID == productId);
    if (product != null) {
        Cart = HttpContext.Session.GetJson<Cart>("cart") 
              ?? new Cart();
        Cart.AddItem(product, 1);
        HttpContext.Session.SetJson("cart", Cart);
    }
    return RedirectToPage(new { returnUrl = returnUrl });
}
```
x??

---

#### Displaying Cart Contents in Cart.cshtml
The `Cart.cshtml` Razor Page displays the current contents of the cart. It iterates through each line item, showing the quantity, product name, price, and subtotal.

:p How does the `Cart.cshtml` page display the items in the cart?
??x
The `Cart.cshtml` page uses a Razor loop to iterate over each item in the cart's lines. For each item, it displays the quantity, product name, and calculates and shows the total price for that line.

```razor
@foreach (var line in Model.Cart?.Lines ?? Enumerable.Empty<CartLine>()) {
    <tr>
        <td class="text-center">@line.Quantity</td>
        <td class="text-left">@line.Product.Name</td>
        <td class="text-right">@line.Product.Price.ToString("c")</td>
        <td class="text-right">@((line.Quantity * line.Product.Price).ToString("c"))</td>
    </tr>
}
```
x??

---

#### Computing the Total Value of the Cart
The `Cart` property in the `CartModel` class provides a method to compute the total value of the cart. This is used to display the grand total at the bottom of the cart view.

:p How does the `Cart` model calculate the total value of the cart?
??x
The `Cart` model calculates the total value by calling the `ComputeTotalValue()` method, which sums up the subtotals for all items in the cart. The result is formatted as a currency string.

```csharp
public decimal ComputeTotalValue()
{
    return Lines.Sum(l => l.Quantity * l.Product.Price);
}
```
x??

---

#### Razor Pages Overview
Razor Pages are a feature of ASP.NET Core that allow for more streamlined and self-contained web pages. They complement MVC Frameworks by providing an easier way to handle simpler tasks without the overhead of full controllers and views. 
:p What is a key benefit of using Razor Pages over traditional MVC in SportsStore?
??x
Razor Pages offer simplicity and ease-of-use, making them ideal for handling single-page workflows such as managing shopping carts or product listings where complex interactions are not required.
x??

---
#### Cart Model Handling
The `CartModel` class is responsible for managing the cart state within a Razor Page. It retrieves products from a repository, updates the cart's content, and persists changes to the session data.
:p How does the `CartModel` manage the user's cart?
??x
The `CartModel` manages the user's cart by:
1. Retrieving a product from the database using the repository.
2. Fetching the user’s cart from the session data.
3. Updating the cart content with the retrieved product.
4. Storing the modified cart and redirecting to the same Razor Page via GET request.
5. Using model binding to handle form posts without needing explicit processing.
```csharp
public class CartModel : PageModel {
    public IStoreRepository Repository { get; set; }
    public Cart Cart { get; private set; }
    public string ReturnUrl { get; set; }

    public CartModel(IStoreRepository repo) {
        Repository = repo;
        Cart = new Cart();
    }

    public void OnGet(string returnUrl = null) {
        // Handle GET request
    }

    public IActionResult OnPost(int id, int quantity) {
        var selectedProduct = Repository.Products.FirstOrDefault(p => p.ProductID == id);
        if (selectedProduct != null) {
            Cart.AddItem(selectedProduct, quantity);
            return RedirectToPage(new { Cart = Cart });
        }
        return Page();
    }
}
```
x??

---
#### Unit Testing Razor Pages
Unit testing for Razor Pages often requires mocking of interfaces like `ISession` and `IStoreRepository`. These mocks help simulate the environment in which the page model operates.
:p How is the `CartModel` tested for loading a cart?
??x
The test for loading a cart involves:
1. Creating mock repository and product instances.
2. Initializing a cart with added items.
3. Setting up session data to mimic cart retrieval.
4. Configuring context objects for the page model.
5. Calling `OnGet()` to simulate a GET request.
6. Asserting the expected state of the cart.
```csharp
[Fact]
public void Can_Load_Cart() {
    // Arrange
    Product p1 = new Product { ProductID = 1, Name = "P1" };
    Product p2 = new Product { ProductID = 2, Name = "P2" };
    Mock<IStoreRepository> mockRepo = new Mock<IStoreRepository>();
    mockRepo.Setup(m => m.Products).Returns((new Product[] {
        p1, p2
    }).AsQueryable<Product>());
    Cart testCart = new Cart();
    testCart.AddItem(p1, 2);
    testCart.AddItem(p2, 1);
    Mock<ISession> mockSession = new Mock<ISession>();
    byte[] data = Encoding.UTF8.GetBytes(
        JsonSerializer.Serialize(testCart));
    mockSession.Setup(c => c.TryGetValue(It.IsAny<string>(), out data.));
    Mock<HttpContext> mockContext = new Mock<HttpContext>();
    mockContext.SetupGet(c => c.Session).Returns(mockSession.Object);
    // Action
    CartModel cartModel = new CartModel(mockRepo.Object) {
        PageContext = new PageContext(new ActionContext {
            HttpContext = mockContext.Object,
            RouteData = new RouteData(),
            ActionDescriptor = new PageActionDescriptor()
        })
    };
    cartModel.OnGet("myUrl");
    // Assert
    Assert.Equal(2, cartModel.Cart?.Lines.Count());
    Assert.Equal("myUrl", cartModel.ReturnUrl);
}
```
x??

---
#### Updating the Cart with Unit Tests
Unit testing for updating the cart involves mocking `IStoreRepository` and session to simulate the changes in cart data. The test captures the byte array passed to the session and deserializes it to verify its contents.
:p How is the `CartModel` tested for updating a cart?
??x
The test for updating a cart involves:
1. Setting up mocks for repository and session.
2. Configuring the mock session to set cart data upon update.
3. Creating an instance of `CartModel`.
4. Calling `OnPost()` with product ID and quantity.
5. Verifying the updated cart content.
```csharp
[Fact]
public void Can_Update_Cart() {
    // Arrange
    Mock<IStoreRepository> mockRepo = new Mock<IStoreRepository>();
    mockRepo.Setup(m => m.Products).Returns((new Product[] {
        new Product { ProductID = 1, Name = "P1" }
    }).AsQueryable<Product>());
    Cart? testCart = new Cart();
    Mock<ISession> mockSession = new Mock<ISession>();
    mockSession.Setup(s => s.Set(It.IsAny<string>(), It.IsAny<byte[]>()))
               .Callback<string, byte[]>((key, val) => {
                   testCart = JsonSerializer.Deserialize<Cart>(Encoding.UTF8.GetString(val));
               });
    Mock<HttpContext> mockContext = new Mock<HttpContext>();
    mockContext.SetupGet(c => c.Session).Returns(mockSession.Object);
    // Action
    CartModel cartModel = new CartModel(mockRepo.Object) {
        PageContext = new PageContext(new ActionContext {
            HttpContext = mockContext.Object,
            RouteData = new RouteData(),
            ActionDescriptor = new PageActionDescriptor()
        })
    };
    cartModel.OnPost(1, 3); // Example post
    // Assert
    Assert.Equal(3, testCart.Lines.First(l => l.ProductID == 1).Quantity);
}
```
x??

---
#### Redirection and Navigation in Razor Pages
The `OnGet` method handles GET requests to display the cart contents. It sets properties like `ReturnUrl` and `Cart`, rendering the Razor content section with these values.
:p How does the `OnGet` method handle displaying cart contents?
??x
The `OnGet` method displays cart contents by:
1. Initializing the `Cart` object and setting `ReturnUrl`.
2. Retrieving the product list from the repository.
3. Rendering the Razor content using the `CartModel` as a view model.
4. Passing properties like `ReturnUrl` to the Razor template for display.
```csharp
public void OnGet(string returnUrl = null) {
    ReturnUrl = returnUrl;
    if (Repository != null) {
        Cart = new Cart();
        foreach (Product p in Repository.Products) {
            Cart.AddItem(p, 1);
        }
    }
}
```
x??

---

#### Navigation Controls and URL Structure
Background context explaining how navigation controls work, especially when combined with page numbers and category selection. This is crucial for understanding how data is passed through URLs during database queries.

:p How do navigation controls pass selected categories in the request URL?
??x
Navigation controls typically include the selected category in the request URL as a query parameter or part of the path, which helps combine it with the page number when querying the database. For instance, if a user navigates to the "Electronics" category on page 2, the URL might look like `http://example.com/Electronics?page=2`.

```csharp
// Example of how a navigation control might be set up in Razor Pages
PageActionDescriptor actionDescriptor = new PageActionDescriptor()
{
    RouteValues = new { CategoryId = "Electronics", PageNumber = 2 }
};
```
x??

---

#### View Bag and Passing Data to Views
Explanation on using the `ViewBag` object to pass additional data, alongside the model, directly to a view in ASP.NET Core. This is useful for dynamic content rendering.

:p How does `ViewBag` help in passing data to views?
??x
The `ViewBag` is a dynamic property that allows developers to easily pass data from controllers to views without having to create a new view model class. It can be used to pass additional information or configuration settings directly to the view.

```csharp
// Example of setting ViewBag properties in a controller action
public IActionResult Index()
{
    var category = "Electronics";
    ViewBag.CategoryName = category;
    return Page();
}
```

In this example, `ViewBag.CategoryName` can be accessed within the Razor view using `@ViewBag.CategoryName`.

x??

---

#### Razor Pages for Simple Features
Explanation on how Razor Pages are suitable for simple self-contained features such as displaying a shopping cart. They offer a streamlined approach to building web pages without needing to write a lot of boilerplate code.

:p Why are Razor Pages well-suited for simple, self-contained features?
??x
Razor Pages are particularly well-suited for simple, self-contained features because they combine the view and logic in a single file, reducing the complexity of separating views from controllers. This makes them ideal for scenarios like displaying the contents of a shopping cart where the logic is straightforward and doesn't require complex interactions.

```csharp
// Example of a Razor Page for displaying a shopping cart
@page "/cart"
@model ShoppingCartModel

<h1>Shopping Cart</h1>
<ul>
    @foreach (var item in Model.Cart.Lines)
    {
        <li>@item.Product.Name x @item.Quantity</li>
    }
</ul>
```

In this example, the `ShoppingCartModel` class contains logic to manage the cart, and the Razor page directly references it.

x??

---

#### Sessions for Data Association
Explanation on using sessions in ASP.NET Core to associate data with a series of related requests. This is useful for maintaining state across multiple requests from the same user.

:p How do sessions help maintain state in ASP.NET Core?
??x
Sessions allow you to store data that can be accessed across multiple HTTP requests during a user's interaction with your application. This is particularly useful for maintaining state, such as shopping cart items or user preferences, over multiple pages and requests.

```csharp
// Example of setting session values in a controller action
public IActionResult AddToCart(int productId)
{
    var cart = HttpContext.Session.GetJson<ShoppingCart>("cart");
    if (cart == null)
    {
        cart = new ShoppingCart();
        HttpContext.Session.SetJson("cart", cart);
    }

    var product = _context.Products.Find(productId);
    cart.AddItem(product, 1);

    return RedirectToPage("/Cart");
}
```

In this example, the session is used to store and manipulate a shopping cart across multiple requests.

x??

---

---
#### Refining the Cart Model with a Service
In this context, we are working on improving the shopping cart functionality of an e-commerce application called SportsStore. The initial implementation stored the cart data directly within Razor Pages and controllers, leading to duplicated code and making maintenance challenging.

The objective is to refactor this by introducing a `Cart` model class that can be managed through services. This will allow other components such as Razor Pages or controllers to interact with the cart without needing to deal with session management details.
:p What is the main issue with storing the cart data directly in Razor Pages and controllers?
??x
Storing the cart data directly within individual pages or controllers results in duplicated code, making it difficult to maintain and update. Each page or controller would need to handle getting and setting the cart object in the session state, leading to redundancy.
```csharp
public class CartPage : PageModel {
    public void OnGet() {
        // Code to get and set cart from session here
    }
}
```
x??
---

#### Creating a Storage-Aware Cart Class
To address the issue of duplicated code, we create a subclass of `Cart` called `SessionCart`. This new class is responsible for managing the cart's storage using session state. It overrides key methods to ensure that updates are saved back to the session.

:p What steps were taken to prepare for creating the `SessionCart` class?
??x
The first step was to make the base `Cart` class virtual by adding the `virtual` keyword, which allows us to override its methods in derived classes. This is necessary because we want to modify how certain operations are performed without changing the original implementation.
```csharp
public class Cart {
    public List<CartLine> Lines { get; set; } = new List<CartLine>();
    // Other methods...
}
```
x??
---

#### Defining the `SessionCart` Class
The `SessionCart` class is designed to manage a cart's storage in session state. It includes a static method called `GetCart` which retrieves or creates an instance of `SessionCart` and injects it with the current session.

:p How does the `GetCart` method work?
??x
The `GetCart` method uses dependency injection to obtain an `IHttpContextAccessor`, which provides access to the HTTP context. From there, it gets a reference to the session object. If a cart already exists in the session, it deserializes and returns that instance; otherwise, it creates a new one.
```csharp
public class SessionCart : Cart {
    public static Cart GetCart(IServiceProvider services) {
        ISession? session = 
            services.GetRequiredService<IHttpContextAccessor>()
                    .HttpContext?.Session;
        SessionCart cart = session?.GetJson<SessionCart>("Cart") ?? new SessionCart();
        cart.Session = session;
        return cart;
    }
}
```
x??
---

#### Overriding Cart Methods in `SessionCart`
The `SessionCart` class overrides three methods from the base `Cart` class: `AddItem`, `RemoveLine`, and `Clear`. These methods call their respective base implementations before updating the state of the cart. The updated state is then serialized and stored back into the session.

:p What are the key responsibilities of the overridden methods in `SessionCart`?
??x
The key responsibility of each method is to first perform the necessary updates on the `CartLine` items, such as adding a new item or removing an existing one. After making these changes, they serialize the updated cart and save it back into the session using the `ISession` interface.
```csharp
public class SessionCart : Cart {
    public override void AddItem(Product product, int quantity) {
        base.AddItem(product, quantity);
        Session?.SetJson("Cart", this);
    }

    public override void RemoveLine(Product product) {
        base.RemoveLine(product);
        Session?.SetJson("Cart", this);
    }

    public override void Clear() {
        base.Clear();
        Session?.Remove("Cart");
    }
}
```
x??
---

