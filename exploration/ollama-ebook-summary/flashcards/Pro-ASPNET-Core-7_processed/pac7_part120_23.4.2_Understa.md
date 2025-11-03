# Flashcards: Pro-ASPNET-Core-7_processed (Part 120)

**Starting Chapter:** 23.4.2 Understanding action results in Razor Pages

---

#### Razor Pages Namespace Configuration
Background context: In Razor Pages, when the view and its page model class are in the same namespace, you can use the `@model` directive without fully qualifying the type. This simplifies the development process by allowing direct referencing of the model class.

:p How does setting up the view and page model class in the same namespace affect the `@model` directive usage?
??x
When the view and its page model are in the same namespace, you can use `@model ClassName` without fully qualifying the type. This is because Razor Pages automatically includes the necessary namespaces to resolve the types.

```csharp
// Example of Index.cshtml file
@page "{id:long?}"
@model IndexModel

<!DOCTYPE html>
<html>
<head>
    <link href="/lib/bootstrap/css/bootstrap.min.css" rel="stylesheet" />
</head>
<body>
    <div class="bg-primary text-white text-center m-2 p-2">
        @Model.Product?.Name
    </div>
</body>
</html>
```

x??

---

#### Razor Page Handler Methods and IActionResult
Background context: Razor Page handler methods use the `IActionResult` interface to control responses. By default, these methods return a view result unless specified otherwise.

:p How do handler methods in Razor Pages handle responses?
??x
Handler methods in Razor Pages use the `IActionResult` interface. By default, they return a view part of the page, but you can explicitly specify different actions using various `PageModel` action result methods such as `Page()`, `NotFound()`, etc.

```csharp
// Example of OnGetAsync method with explicit Page result
public async Task<IActionResult> OnGetAsync(long id = 1)
{
    Product = await context.Products.FindAsync(id);
    return Page(); // Returns a view part of the page
}
```

x??

---

#### PageModel Action Result Methods
Background context: The `PageModel` class provides several methods to create different action results, allowing for more control over how pages respond.

:p What are some common `PageModel` action result methods used in Razor Pages?
??x
Common `PageModel` action result methods include:

- `Page()`: Produces a 200 OK status code and renders the view part of the page.
- `NotFound()`: Produces a 404 NOT FOUND status code.
- `BadRequest(state)`: Produces a 400 BAD REQUEST status code, optionally with a model state object.
- `File(name, type)`: Produces a 200 OK response, sets Content-Type header to the specified type, and sends the file to the client.
- `Redirect(path)`, `RedirectPermanent(path)`: Produce 302 FOUND and 301 MOVED PERMANENTLY responses, respectively.
- `RedirectToAction(name)`, `RedirectToActionPermanent(name)`: Produce 302 FOUND and 301 MOVED PERMANENTLY responses for redirecting to actions in controllers.
- `RedirectToPage(name)`, `RedirectToPagePermanent(name)`: Produce 302 FOUND and 301 MOVED PERMANENTLY responses for redirecting to other Razor Pages.

```csharp
public IActionResult OnGetAsync(long id = 1)
{
    Product = await context.Products.FindAsync(id);
    if (Product == null)
    {
        return NotFound(); // Produces a 404 NOT FOUND status code
    }
    return Page(); // Default action: renders the view part of the page
}
```

x??

---

#### Handling Not Found Requests in Razor Pages
Background context: To handle not found requests, you can create a dedicated Razor Page and use `RedirectToPage` or similar methods to redirect users.

:p How do you handle not found requests in Razor Pages?
??x
You can handle not found requests by creating a specific Razor Page for this purpose. For example:

```csharp
// NotFound.cshtml file content
@page "/noid"
@model NotFoundModel

<!DOCTYPE html>
<html>
<head>
    <link href="/lib/bootstrap/css/bootstrap.min.css" rel="stylesheet" />
    <title>Not Found</title>
</head>
<body>
    <div class="bg-primary text-white text-center m-2 p-2">
        No Matching ID
    </div>
    <ul class="list-group m-2">
        @foreach (Product p in Model.Products)
        {
            <li class="list-group-item">@p.Name (ID: @p.ProductId)</li>
        }
    </ul>
</body>
</html>

// Corresponding PageModel
public class NotFoundModel : PageModel
{
    private DataContext context;
    public IEnumerable<Product> Products { get; set; } = Enumerable.Empty<Product>();
    public NotFoundModel(DataContext ctx)
    {
        context = ctx;
    }

    public void OnGetAsync(long id = 1)
    {
        Products = context.Products;
    }
}
```

In the handler method of the main page, you can use `RedirectToPage` to redirect users to this dedicated not found page:

```csharp
public async Task<IActionResult> OnGetAsync(long id = 1)
{
    Product = await context.Products.FindAsync(id);
    if (Product == null)
    {
        return RedirectToPage("NotFound");
    }
    return Page();
}
```

x??

---

These flashcards cover the key concepts of Razor Pages, including namespace configuration, handling responses using `IActionResult`, and managing not found requests.

#### Routing and Redirects in Razor Pages
Background context: In ASP.NET Core, the routing system is used to manage URLs and redirect clients to appropriate pages based on defined patterns. The `@page` directive specifies a URL pattern for a Razor Page.

:p How does the `RedirectToPage` method work with the `@page` directive?
??x
The `RedirectToPage` method takes an argument, which is translated into a redirection according to the routing pattern specified in the `@page` directive. For example, if the `@page` directive uses the argument "NotFound," it redirects to the `/noid` path.

```csharp
public IActionResult OnGetAsync(long id) {
    Product = await context.Products.FindAsync(id);
    return RedirectToPage("/NotFound");
}
```
x??

---
#### Handling Multiple HTTP Methods in Razor Pages
Background context: Razor Pages support different HTTP methods like GET and POST to handle various operations, such as viewing data (GET) and editing or saving data (POST). This example demonstrates handling both methods with a simple product editor page.

:p What are the two common HTTP methods supported by Razor Pages for typical CRUD operations?
??x
The two common HTTP methods supported by Razor Pages for typical CRUD operations are GET and POST. GET is used to fetch and display data, while POST is used to submit changes or save data.

```csharp
public class EditorModel : PageModel {
    public async Task<IActionResult> OnGetAsync(long id) {
        Product = await context.Products.FindAsync(id);
        return Page();
    }

    public async Task<IActionResult> OnPostAsync(long id, decimal price) {
        Product? p = await context.Products.FindAsync(id);
        if (p != null) {
            p.Price = price;
        }
        await context.SaveChangesAsync();
        return RedirectToPage();
    }
}
```
x??

---
#### Handling GET and POST Methods in Editor.cshtml
Background context: The `Editor.cshtml` Razor Page handles both GET and POST methods. It fetches product details via the GET method and updates them via the POST method.

:p How does the `Editor.cshtml` file handle data retrieval and submission?
??x
The `Editor.cshtml` file uses a combination of server-side logic and client-side HTML to handle both GET and POST methods:
- On GET: It fetches product details from the database using the provided ID.
- On POST: It updates the product price based on user input and saves the changes.

```csharp
@page "{id:long}"
@model EditorModel

<!DOCTYPE html>
<html>
<head>
    <link href="/lib/bootstrap/css/bootstrap.min.css" rel="stylesheet" />
</head>
<body>
    <div class="bg-primary text-white text-center m-2 p-2">Editor</div>
    <div class="m-2">
        <table class="table table-sm table-striped table-bordered">
            <tbody>
                <tr><th>Name</th><td>@Model.Product?.Name</td></tr>
                <tr><th>Price</th><td>@Model.Product?.Price</td></tr>
            </tbody>
        </table>
        <form method="post">
            @Html.AntiForgeryToken()
            <div class="form-group">
                <label>Price</label>
                <input name="price" class="form-control" value="@Model.Product?.Price" />
            </div>
            <button class="btn btn-primary mt-2" type="submit">Submit</button>
        </form>
    </div>
</body>
</html>

namespace WebApp.Pages {
    public class EditorModel : PageModel {
        private DataContext context;
        public Product? Product { get; set; }
        public EditorModel(DataContext ctx) {
            context = ctx;
        }
        public async Task<IActionResult> OnGetAsync(long id) {
            Product = await context.Products.FindAsync(id);
            return Page();
        }
        public async Task<IActionResult> OnPostAsync(long id, decimal price) {
            Product? p = await context.Products.FindAsync(id);
            if (p != null) {
                p.Price = price;
            }
            await context.SaveChangesAsync();
            return RedirectToPage();
        }
    }
}
```
x??

---
#### CSRF Protection in Razor Pages
Background context: Cross-Site Request Forgery (CSRF) is a security vulnerability where an attacker tricks a user into submitting a malicious request. ASP.NET Core uses anti-forgery tokens to protect against such attacks.

:p What is the purpose of `@Html.AntiForgeryToken()` in the `Editor.cshtml` file?
??x
The purpose of `@Html.AntiForgeryToken()` is to add a hidden form field that contains an anti-forgery token. This helps prevent CSRF attacks by ensuring that only legitimate requests with valid tokens are processed.

```csharp
@Html.AntiForgeryToken()
```
x??

---

#### Handling HTTP Methods in Razor Pages

Razor Pages handle different HTTP methods like GET and POST. The `OnPostAsync` method is used to process POST requests, which are typically sent when a form is submitted by the browser.

:p How does the OnPostAsync method handle POST requests?
??x
The `OnPostAsync` method handles POST requests by extracting parameters from the request, such as the id value from the URL route and the price value from the form. After processing these values (e.g., updating the database), it redirects the user to a GET URL using `RedirectToPage()`, preventing resubmission of the POST request if the page is refreshed.

```csharp
public class ProductEditorModel : PageModel
{
    public async Task<IActionResult> OnPostAsync(long id = 1, string price = "0")
    {
        var product = await _context.Products.FindAsync(id);
        if (product != null)
        {
            product.Price = decimal.Parse(price);
            await _context.SaveChangesAsync();
        }
        
        return RedirectToPage();
    }
}
```
x??

---

#### Redirecting After POST

After processing a POST request, it's common to redirect the user back to a GET URL using `RedirectToPage()`. This prevents accidental resubmission of the POST request if the page is refreshed.

:p How does the `RedirectToPage()` method work?
??x
The `RedirectToPage()` method redirects the client to the URL for the Razor Page, effectively telling the browser to send a GET request. This prevents the user from accidentally re-sending the POST request if they reload the page, thus avoiding potential issues like double submissions.

```csharp
public IActionResult OnPostAsync()
{
    // Process data...
    
    return RedirectToPage();
}
```
x??

---

#### Selecting Handler Methods

Razor Pages can define multiple handler methods (e.g., `OnGetAsync`, `OnGetRelatedAsync`) to handle different types of requests. The request can select the appropriate method using a query string parameter or routing segment variable.

:p How do you differentiate between handler methods in Razor Pages?
??x
In Razor Pages, you can use a query string parameter or routing segment to specify which handler method to execute. For example, by default, `OnGetAsync` is used for GET requests unless specified otherwise. You can name the handler method without the "On" and "Async" prefixes to select it using a specific value.

```csharp
public class HandlerSelectorModel : PageModel
{
    public async Task OnGetAsync(long id = 1)
    {
        // Handle GET request...
    }

    public async Task OnGetRelatedAsync(long id = 1)
    {
        // Handle related data for the GET request...
    }
}
```
x??

---

#### Using Rate Limiting and Output Caching

Rate limiting and output caching are features in ASP.NET Core that can be applied to both controllers and page model classes. These features help manage performance, especially when dealing with high traffic.

:p How do you apply rate limiting and output caching to Razor Pages?
??x
You can use the `[RateLimitAttribute]` and `[OutputCacheAttribute]` attributes directly on a Razor Page class or its methods to enable rate limiting and output caching. These features help control how many requests are processed within a given time frame and reduce database load by serving cached responses.

```csharp
[RateLimit(10, TimeWindow = TimeSpan.FromSeconds(60))]
public class ProductSelectorModel : PageModel
{
    [OutputCache(Duration = 300)]
    public async Task<IActionResult> OnGetAsync(long id)
    {
        // Handle GET request...
    }
}
```
x??

---

