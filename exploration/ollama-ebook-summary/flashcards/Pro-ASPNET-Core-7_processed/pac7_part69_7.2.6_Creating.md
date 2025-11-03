# Flashcards: Pro-ASPNET-Core-7_processed (Part 69)

**Starting Chapter:** 7.2.6 Creating the database migration. 7.2.7 Creating seed data

---

#### Entity Framework Core Migrations
Entity Framework Core can generate database schema from data model classes through a feature called migrations. When you prepare a migration, EF Core creates a C# class containing SQL commands required to prepare the database.

:p What is a migration in Entity Framework Core?
??x
A migration in Entity Framework Core is a mechanism that allows developers to update the database schema based on changes made to the model classes. It generates a set of SQL commands as a C# class, which can be used to apply or roll back these changes automatically.
??x

---

#### Creating the Initial Migration Class
To create an initial migration, you use the `dotnet ef migrations add` command in PowerShell from the project folder.

:p How do you create an initial migration using Entity Framework Core?
??x
You run the following command in a PowerShell window:
```shell
dotnet ef migrations add Initial
```
This command generates a C# class under the Migrations folder, which contains SQL commands necessary to set up the database schema for the first time. The class name typically includes a timestamp and a suffix like `_Initial.cs`.
??x

---

#### Entity Framework Core Seed Data
Seed data is used to populate the database with initial or sample data when the application starts.

:p What is seed data in the context of Entity Framework Core?
??x
Seed data refers to predefined data that is loaded into the database during application startup. This can be useful for providing initial values or examples, especially for testing and development purposes.
??x

---

#### SeedData.cs Class Details
The `SeedData` class contains a static method named `EnsurePopulated` which checks if there are any products in the database and adds some sample data if not.

:p What is the purpose of the `EnsurePopulated` method in the `SeedData` class?
??x
The `EnsurePopulated` method ensures that the database has initial product records. It first checks whether there are pending migrations using `context.Database.GetPendingMigrations().Any()`. If there are, it applies these migrations with `context.Database.Migrate()`. Then, it checks if any products exist in the database and adds sample data if none are found.
??x

---

#### Calling Seed Data during Application Startup
You need to call the `EnsurePopulated` method from the `Program.cs` file to seed the database when the application starts.

:p How do you ensure that seed data is applied when the application starts?
??x
In the `Program.cs` file, you add a call to the `EnsurePopulated` method of the `SeedData` class inside the `app` object. This ensures that the database gets seeded with initial data whenever the application runs.
```csharp
using SportsStore.Models;

var builder = WebApplication.CreateBuilder(args);
// other configurations

builder.Services.AddControllersWithViews();
builder.Services.AddDbContext<StoreDbContext>(opts => {
    opts.UseSqlServer(builder.Configuration["ConnectionStrings:SportsStoreConnection"]);
});
builder.Services.AddScoped<IStoreRepository, EFStoreRepository>();

var app = builder.Build();

app.UseStaticFiles();
app.MapDefaultControllerRoute();
SeedData.EnsurePopulated(app);  // Seed data here

app.Run();
```
??x

---

#### Dropping and Re-creating the Database
You can reset the database by dropping it and recreating it with initial seed data.

:p How do you reset the database using Entity Framework Core?
??x
To reset the database, you run the following command in a PowerShell window:
```shell
dotnet ef database drop --force --context StoreDbContext
```
This drops the existing database. After running this, you can start the application again to re-create and seed the database.
??x

#### Preparation Work for an ASP.NET Core Project
Background context: The initial setup of an ASP.NET Core project can be time-consuming, but once the foundational work is done, adding features becomes faster. This section focuses on creating a controller to display product details from the repository.

:p What is the main goal of this section in the text?
??x
The main goal is to create a controller and action method that can display a list of products stored in the repository.
x??

---

#### Using Visual Studio Scaffolding (Optional)
Background context: Visual Studio supports scaffolding to automatically generate code, but the author recommends not using it for this book. The generated code is generic and doesn't address common development problems effectively.

:p Why does the author recommend against using scaffolding in this book?
??x
The author avoids using scaffolding because the generated code is too generic and doesn’t address common development issues, making it harder to understand how everything works behind the scenes.
x??

---

#### Preparing the Controller for Displaying Products
Background context: The `HomeController` is prepared with a constructor that takes an `IStoreRepository`. This allows dependency injection, which enables the controller to access repository methods without knowing the exact implementation.

:p What does the `HomeController` constructor do in this example?
??x
The `HomeController` constructor accepts an `IStoreRepository` as a parameter. It sets up the `repository` field and uses it to fetch products for displaying on the view.
x??

---

#### Dependency Injection in ASP.NET Core
Background context: Dependency injection is used here to manage dependencies between components. The `Program.cs` file configures which implementation class should be used, allowing for easy reconfiguration if needed.

:p How does dependency injection work with `HomeController`?
??x
Dependency injection works by having the constructor of `HomeController` accept an instance of `IStoreRepository`. ASP.NET Core creates a new instance of the configured repository and passes it to the controller's constructor. This allows the controller to use the repository without knowing its implementation.
x??

---

#### Unit Testing Controller Access to Repository
Background context: A unit test is created to verify that the controller correctly accesses the repository. A mock repository is used for testing purposes.

:p How can you unit test the `HomeController`'s access to the repository?
??x
You can create a mock repository, inject it into the `HomeController` constructor, call the `Index` method, and then compare the product objects returned with expected values from the mock implementation.
x??

---

#### Code Example for HomeController Tests
Background context: The provided code sets up a unit test to ensure that the controller is accessing the repository correctly. A mock repository is used to simulate product data.

:p What does this unit test check?
??x
This unit test checks whether the `HomeController`'s `Index` method correctly fetches and displays products by using a mock repository.
x??

---
Note: The code examples in the text are primarily about setup and not fully complete. They serve as prompts for understanding the process rather than being fully functional on their own.

#### Retrieving Data from Action Method Result
Background context: In this scenario, we are working with an action method that returns a `ViewResult` object. The method's result needs to be cast and checked for null values before being processed further.

:p How do you retrieve and process the data returned by an action method in a unit test?
??x
To retrieve and process the data returned by an action method, first cast it to the expected type (in this case, `ViewResult`). Then, extract the model from the view data. If the result is null or not of the correct type, handle these cases appropriately.

Example code:
```csharp
IEnumerable<Product>? result = (controller.Index() as ViewResult)?.ViewData.Model as IEnumerable<Product>;
```
??x
The answer with detailed explanations.
To retrieve and process the data returned by an action method in a unit test, you need to ensure that you correctly handle the possible null or unexpected types. The code snippet first casts the result of `controller.Index()` to a `ViewResult` if it is not null. Then, it extracts the model from the `ViewData` property and casts it to `IEnumerable<Product>`. If any part of this process results in a null value, the result will be null, which you can handle with a null-coalescing operator or similar mechanism.

```csharp
// Act
IEnumerable<Product>? result = (controller.Index() as ViewResult)?.ViewData.Model as IEnumerable<Product>;

// Assert
Product[] prodArray = result?.ToArray() ?? Array.Empty<Product>();
Assert.True(prodArray.Length == 2);
Assert.Equal("P1", prodArray[0].Name);
Assert.Equal("P2", prodArray[1].Name);
```
x??

---

#### Updating the View to Use Product Data
Background context: The `Index` action method passes a collection of `Product` objects from the repository to the view, which uses Razor syntax to generate HTML content.

:p How do you modify an existing Razor view to use product data?
??x
To modify an existing Razor view to use product data, you need to specify that the model type is `IQueryable<Product>`. Then, use a loop to iterate through these products and generate appropriate HTML elements for each one.

Example code:
```html
@model IQueryable<Product>

@foreach (var p in Model ?? Enumerable.Empty<Product>()) {
    <div>
        <h3>@p.Name</h3>
        @p.Description
        <h4>@p.Price.ToString("c")</h4>
    </div>
}
```
??x
The answer with detailed explanations.
To modify an existing Razor view to use product data, you first need to declare the model type at the top of the file. In this case, it is `IQueryable<Product>`, which means that the view expects a collection of products as its model.

Using the `@model` directive ensures that the view knows about the expected input from the action method. The `@foreach` loop iterates over each product in the model. If the model is null or empty, it uses an empty enumerable to prevent errors.

Example code:
```html
@model IQueryable<Product>

@foreach (var p in Model ?? Enumerable.Empty<Product>()) {
    <div>
        <h3>@p.Name</h3>
        @p.Description
        <h4>@p.Price.ToString("c")</h4>
    </div>
}
```
??x
The answer with detailed explanations.
To modify the view to use product data:
1. Use `@model IQueryable<Product>` at the top of the file to specify that the model type is a collection of products.
2. Use an `@foreach` loop to iterate through each product in the model.
3. For null or empty models, use the null-coalescing operator (`??`) with an empty enumerable.

```html
@model IQueryable<Product>

@foreach (var p in Model ?? Enumerable.Empty<Product>()) {
    <div>
        <h3>@p.Name</h3>
        @p.Description
        <h4>@p.Price.ToString("c")</h4>
    </div>
}
```
x??

---
#### Understanding Nullable Models in Razor Views
Background context: In Razor views, the model data is always nullable even if the specified type is not. This behavior can lead to issues unless handled properly.

:p Why do you need to handle null values in a Razor view?
??x
In Razor views, the model data is always considered nullable, even if the specified type (like `Product`) is not. Handling null values ensures that your application does not crash or produce unexpected results when no data is passed from the action method.

Example code:
```html
@model IQueryable<Product>

@foreach (var p in Model ?? Enumerable.Empty<Product>()) {
    <div>
        <h3>@p.Name</h3>
        @p.Description
        <h4>@p.Price.ToString("c")</h4>
    </div>
}
```
??x
The answer with detailed explanations.
In Razor views, the model data is always nullable even if you specify a non-nullable type like `Product`. This means that if no data is passed from the action method, the model will be null. Handling this situation prevents runtime errors and ensures that your view can gracefully handle scenarios where there might not be any products to display.

Example code:
```html
@model IQueryable<Product>

@foreach (var p in Model ?? Enumerable.Empty<Product>()) {
    <div>
        <h3>@p.Name</h3>
        @p.Description
        <h4>@p.Price.ToString("c")</h4>
    </div>
}
```
The null-coalescing operator (`??`) is used here to provide a fallback when `Model` is null. If it is null, an empty enumerable of products is provided instead.

```html
@foreach (var p in Model ?? Enumerable.Empty<Product>()) {
    // Code inside the loop
}
```
x??

---

