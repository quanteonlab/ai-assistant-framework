# Flashcards: Pro-ASPNET-Core-7_processed (Part 21)

**Starting Chapter:** 7.2.6 Creating the database migration. 7.2.7 Creating seed data

---

#### Creating a Database Migration Using Entity Framework Core
Background context: Entity Framework Core can generate database schema from model classes using migrations. Migrations are used to handle changes in your data models without manually writing SQL commands. This is particularly useful for maintaining a consistent and up-to-date database schema.

:p What command creates a migration class that prepares the database for its first use?
??x
The `dotnet ef migrations add Initial` command generates a C# class containing the necessary SQL commands to prepare the database.
```sh
dotnet ef migrations add Initial
```
x??

---

#### Understanding Seed Data and Migrations
Background context: To populate the database with sample data, seed data is created using classes. This helps in providing initial data for testing and demonstration purposes. The `EnsurePopulated` method ensures that the database has the initial data.

:p What does the `EnsurePopulated` method do?
??x
The `EnsurePopulated` method checks if there are any pending migrations to create or update the database schema, then seeds it with sample products.
```csharp
public static void EnsurePopulated(IApplicationBuilder app)
{
    // Obtain a StoreDbContext object through IApplicationBuilder interface and call Database.Migrate() 
    StoreDbContext context = app.ApplicationServices
        .CreateScope().ServiceProvider
        .GetRequiredService<StoreDbContext>();
    
    if (context.Database.GetPendingMigrations().Any())
    {
        context.Database.Migrate();
    }
    
    // Check if any products exist, if not add a collection of sample products
    if (!context.Products.Any())
    {
        context.Products.AddRange(
            new Product { Name = "Kayak", Description = "A boat for one person", Category = "Watersports", Price = 275m },
            // more products...
        );
        context.SaveChanges();
    }
}
```
x??

---

#### Running the Application to Seed Data
Background context: To seed data when the application starts, a call is added in `Program.cs` to ensure that the database has initial data. This involves creating an instance of `WebApplication` and calling the `EnsurePopulated` method.

:p How do you seed the database when the application starts?
??x
You add a call to the `EnsurePopulated` method from within the `Program.cs` file:
```csharp
var app = builder.Build();
// other middleware configurations...

SeedData.EnsurePopulated(app);

app.Run();
```
x??

---

#### Resetting the Database Using Migrations
Background context: If you need to reset the database, you can use the `dotnet ef database drop` command. This drops the existing database and recreates it based on migrations.

:p How do you reset the database using Entity Framework Core?
??x
You run the following command in the SportsStore folder:
```sh
dotnet ef database drop --force --context StoreDbContext
```
This command drops the existing database, forces a re-creation of the schema, and seeds it with initial data.
x??

---

#### Initial Setup of ASP.NET Core Project
Background context: The initial setup for an ASP.NET Core project can be time-consuming, involving various configurations and preparations. However, once these foundational steps are completed, development becomes more efficient.

:p What is the first step to add a new item to an ASP.NET Core project using Visual Studio?
??x
To add a new item to your project in Visual Studio, you need to right-click on a folder within Solution Explorer, select "Add" > "New Item," and then choose an appropriate template from the Add New Item window. This action initiates the process of adding a new file or class to your project.
x??

---

#### Using Scaffolding vs Manual Coding
Background context: While Visual Studio supports scaffolding for generating items quickly, this book opts for manual coding. Scaffolding generates generic code that is not always useful and does not cover common development scenarios. The goal here is to understand how things work under the hood rather than relying on predefined templates.

:p Why might one prefer using scaffolding over manually coding a solution?
??x
Using scaffolding can save time by automatically generating boilerplate code, making it faster to prototype solutions. However, in this book, manual coding is preferred because it ensures a deeper understanding of how components interact and are configured within the application. This approach allows for more flexibility and customization.
x??

---

#### Preparing the HomeController
Background context: The `HomeController` class is responsible for handling HTTP requests and displaying views related to products. To ensure that the controller can access the repository, dependency injection is used.

:p How does ASP.NET Core handle the creation of a new instance of the `HomeController`?
??x
ASP.NET Core inspects the constructor of the `HomeController` class to determine its dependencies. In this case, it requires an implementation of `IStoreRepository`. It uses the configuration in the `Program.cs` file to create an appropriate repository (e.g., `EFStoreRepository`) and pass it into the controller's constructor. This process is known as dependency injection.

Here’s a simplified version of how this works:
```csharp
public class HomeController : Controller {
    private IStoreRepository repository;

    public HomeController(IStoreRepository repo) {
        repository = repo;
    }
}
```
x??

---

#### Dependency Injection in Action
Background context: Dependency injection allows the `HomeController` to access the application’s data through an interface (`IStoreRepository`) without knowing which specific implementation is used. This makes it easier to switch implementations (e.g., from Entity Framework to another storage solution) without modifying the controller.

:p What is dependency injection, and why is it useful in this context?
??x
Dependency injection is a design pattern where objects are provided with their dependencies instead of creating those dependencies themselves. In the context of `HomeController`, it means that the controller does not directly instantiate its repository but receives one via its constructor. This decouples the controller from any specific implementation, making the application more flexible and easier to test.

For example:
```csharp
public class HomeController : Controller {
    private IStoreRepository repository;

    public HomeController(IStoreRepository repo) {
        repository = repo;
    }

    public IActionResult Index() => View(repository.Products);
}
```
x??

---

#### Unit Testing with Mock Objects
Background context: To verify that the `HomeController` is correctly accessing the repository, a unit test can be written using mock objects. A mock repository is created to simulate data and behavior.

:p How can you set up a unit test for the `Index()` action in `HomeController`?
??x
You can create a mock repository that mimics the actual repository's behavior but is controlled by your tests. Here’s an example of how to set it up:

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.AspNetCore.Mvc;
using Moq;
using SportsStore.Controllers;
using SportsStore.Models;
using Xunit;

namespace SportsStore.Tests {
    public class HomeControllerTests {

        [Fact]
        public void Can_Use_Repository() {
            // Arrange
            Mock<IStoreRepository> mock = new Mock<IStoreRepository>();
            mock.Setup(m => m.Products).Returns((new Product[] {
                new Product { ProductID = 1, Name = "P1" },
                new Product { ProductID = 2, Name = "P2" }
            }).AsQueryable<Product>());

            HomeController controller = new HomeController(mock.Object);

            // Act
            var result = controller.Index() as ViewResult;

            // Assert
            Assert.IsAssignableFrom<ViewResult>(result);
            Assert.Contains(result.Model as List<Product>, p => p.Name == "P1");
            Assert.Contains(result.Model as List<Product>, p => p.Name == "P2");
        }
    }
}
```

This test checks that the `Index` action returns a view with products and ensures they match the expected data.
x??

---

---
#### Getting Data from an Action Method
Background context explaining how action methods can return different types of results, and how to handle them. The example uses `ViewResult` which is a common result type returned by ASP.NET Core MVC actions.

:p How do you retrieve data from an action method that returns a `ViewResult` in ASP.NET Core MVC?
??x
To retrieve data from an action method that returns a `ViewResult`, you need to cast the `ViewData.Model` property of the `ViewResult`. The following code snippet demonstrates how to achieve this:

```csharp
// Act
IEnumerable<Product>? result = 
    (controller.Index() as ViewResult)?.ViewData.Model 
    as IEnumerable<Product>;

// Assert
Product[] prodArray = result?.ToArray() ?? Array.Empty<Product>();
Assert.True(prodArray.Length == 2);
Assert.Equal("P1", prodArray[0].Name);
Assert.Equal("P2", prodArray[1].Name);
```

In this example, the `controller.Index()` method is expected to return a `ViewResult` which contains a model of type `IEnumerable<Product>`. The cast and null-check are necessary because the result can potentially be null.
x??
---
#### Using Product Data in the View
Background context explaining how action methods pass data (ViewModel) to views, specifically using Razor syntax. The example demonstrates retrieving an `IQueryable<Product>` from the repository.

:p How do you use product data within a Razor view file?
??x
To use product data within a Razor view file, you specify the model type at the top of your `.cshtml` file and then iterate over the collection using an `@foreach` loop. The following code snippet shows how to implement this:

```csharp
@model IQueryable<Product>

@foreach (var p in Model ?? Enumerable.Empty<Product>()) {
    <div>
        <h3>@p.Name</h3>
        @p.Description
        <h4>@p.Price.ToString("c")</h4>
    </div>
}
```

In this example, the `@model` directive specifies that the view expects an `IQueryable<Product>` as its model. The `@foreach` loop iterates over each product and generates HTML content for display. The null-coalescing operator (`??`) ensures that if the model is empty or null, an empty enumerable is used to avoid runtime errors.
x??
---
#### Handling Nullable Model Data in Razor Views
Background context explaining the behavior of `@model` in Razor views, which always treats model data as nullable even when a non-nullable type is specified.

:p Why does the @model expression treat its value as nullable in Razor views?
??x
The `@model` directive in Razor views can return null, regardless of the actual type. This behavior is necessary because there are scenarios where a view might not receive any data (e.g., when a route constraint fails). The following example demonstrates how to handle this situation:

```csharp
@model IQueryable<Product>

@foreach (var p in Model ?? Enumerable.Empty<Product>()) {
    <div>
        <h3>@p.Name</h3>
        @p.Description
        <h4>@p.Price.ToString("c")</h4>
    </div>
}
```

Here, the null-coalescing operator (`??`) is used to ensure that if `Model` is null or empty, a fallback enumerable (in this case, an empty product collection) is used. This prevents runtime errors and ensures consistent behavior.
x??
---

