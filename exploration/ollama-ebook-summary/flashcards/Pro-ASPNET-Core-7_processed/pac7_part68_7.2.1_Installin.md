# Flashcards: Pro-ASPNET-Core-7_processed (Part 68)

**Starting Chapter:** 7.2.1 Installing the Entity Framework Core packages

---

---
#### Data Model Definition for Product
Background context: The provided C# code defines a data model for storing product information in an application, specifically within the `SportsStore` namespace. This model includes properties such as `ProductID`, `Name`, `Description`, `Price`, and `Category`.

:p What is the purpose of defining the `ProductID` property with `long?` type?
??x
The `ProductID` property uses `long?`, which means it can be a nullable long integer. This allows the ID to be optional in some scenarios, such as when creating or updating products where the ID might not yet be assigned.

```csharp
public class Product {
    public long? ProductID { get; set; } // Nullable long for product ID
    public string Name { get; set; } = String.Empty; // Default empty string for name
}
```
x??

---
#### Properties Definition and Column Attribute
Background context: The `Product` model includes several properties with default values. Additionally, the `Price` property is decorated with a `Column` attribute to specify its SQL data type.

:p What does the `Column` attribute do in this scenario?
??x
The `Column` attribute specifies that the `Price` property should be stored as an `decimal(8, 2)` column in the database. This ensures that the price values are stored with a precision of up to 8 digits and 2 decimal places.

```csharp
[Column(TypeName = "decimal(8, 2)")] // Specifies SQL data type for Price property
public decimal Price { get; set; }
```
x??

---
#### Running the Application
Background context: The application needs to be tested before adding any data. The provided command can be used to run the application and check its response.

:p What command is used to run the example application?
??x
The command `dotnet run` is used to execute the SportsStore application in the current folder.

```sh
dotnet run
```
x??

---
#### Adding Data to Application
Background context: Once the application has been confirmed to build and run, it's time to add data. The application uses Entity Framework Core to store its data in a SQL Server LocalDB database.

:p What is the significance of adding data to an ASP.NET Core project?
??x
Adding data to an ASP.NET Core project allows the application to provide more meaningful responses beyond just basic setup. In this case, populating the `SportsStore` with product data will enable users to view and interact with real items instead of placeholder data.

```csharp
// Example of adding a new Product record
public class ProductService {
    private readonly ApplicationDbContext _context;
    
    public ProductService(ApplicationDbContext context) {
        _context = context;
    }

    public void AddProduct(Product product) {
        _context.Products.Add(product);
        _context.SaveChanges();
    }
}
```
x??

---
#### Entity Framework Core and SQL Server LocalDB
Background context: The `SportsStore` application uses Entity Framework Core for database access, storing data in a SQL Server LocalDB. This approach simplifies the development process by abstracting many of the complexities associated with direct SQL interactions.

:p Why is Entity Framework Core important for ASP.NET Core projects?
??x
Entity Framework Core (EF Core) is crucial because it acts as an ORM (Object-Relational Mapping) framework, allowing developers to work with data using C# objects rather than raw SQL commands. This makes the code more maintainable and less prone to SQL injection attacks.

```csharp
// Example of Entity Framework Core context setup
public class ApplicationDbContext : DbContext {
    public DbSet<Product> Products { get; set; }
    
    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder) {
        optionsBuilder.UseSqlServer("Server=(localdb)\\mssqllocaldb;Database=SportsStoreDb;");
    }
}
```
x??

---
#### Database Setup
Background context: For the `SportsStore` application to work, a SQL Server LocalDB database must be set up. This is a separate step from building and running the application.

:p Why is it necessary to install SQL Server LocalDB if you did not do so in chapter 2?
??x
It is essential to install SQL Server LocalDB because the `SportsStore` application relies on it for storing data. Without this database, the application cannot persist product information or perform other database operations that are critical for its functionality.

```sh
// Command to install SQL Server LocalDB (example)
dotnet sqllocaldb create "MSSQLLocalDB"
```
x??

---

---
#### Adding Entity Framework Core Packages
Background context: The first step to integrate Entity Framework Core (EF Core) with the SportsStore project involves adding specific NuGet packages that provide the necessary functionality. EF Core allows for data manipulation and storage through code-first migrations, while SQL Server support enables interactions with a relational database.

:p What are the commands used to add Entity Framework Core packages to the SportsStore project?
??x
The `dotnet add package` command is used twice in Listing 7.13:
```bash
dotnet add package Microsoft.EntityFrameworkCore.Design --version 7.0.0
dotnet add package Microsoft.EntityFrameworkCore.SqlServer --version 7.0.0
```
These commands install the required packages for EF Core and SQL Server support.

To ensure compatibility with ASP.NET Core, a tools package is also installed:
```bash
# Uninstall any existing version of the tool package
dotnet tool uninstall --global dotnet-ef

# Install the specified version of the tool package
dotnet tool install --global dotnet-ef --version 7.0.0
```
x??

---
#### Configuration Settings for Database Connection
Background context: Configuration settings like database connection strings are stored in JSON configuration files to ensure flexibility and ease of management across different environments (e.g., development, production).

:p How do you add a database connection string to the `appsettings.json` file?
??x
Add or modify the following entry in the `appsettings.json` file located in the SportsStore folder:
```json
{
  "ConnectionStrings": {
    "SportsStoreConnection": "Server=(localdb)\\MSSQLLocalDB;Database=SportsStore;MultipleActiveResultSets=true"
  }
}
```
This connection string specifies a LocalDB database named `SportsStore` and enables the multiple active result set (MARS) feature, which is essential for certain EF Core queries.

Ensure that JSON data structure adheres strictly to the format provided. Incorrect quoting or formatting can lead to configuration errors.
??x

---
#### Creating the Database Context Class
Background context: A database context class in Entity Framework Core acts as an entry point for interacting with the database. It encapsulates all interactions and provides a way to manage entities (e.g., `Product`).

:p How do you define a database context class for managing products?
??x
Create a new file named `StoreDbContext.cs` in the `Models` folder and add the following code:
```csharp
using Microsoft.EntityFrameworkCore;

namespace SportsStore.Models {
    public class StoreDbContext : DbContext {
        public StoreDbContext(DbContextOptions<StoreDbContext> options) 
            : base(options) { }

        public DbSet<Product> Products => Set<Product>();
    }
}
```
Explanation: The `StoreDbContext` class derives from `DbContext`, which is part of EF Core. It takes a `DbContextOptions<T>` object as its constructor parameter, allowing for configuration such as connection strings.

The `Products` property uses the `DbSet<T>` collection to manage entities of type `Product`.
??x

---

#### Configuring Entity Framework Core
Entity Framework Core must be configured to know which database type it will connect to, the connection string describing that connection, and the context class presenting data. This setup is done via the `Program.cs` file using ASP.NET Core's configuration system.

:p How do you configure Entity Framework Core in the `Program.cs` file for connecting to a SQL Server database?
??x
To configure Entity Framework Core, you use the `AddDbContext` method within the builder services. The connection string is retrieved from the configuration settings via `builder.Configuration`.

```csharp
builder.Services.AddDbContext<StoreDbContext>(opts =>
    opts.UseSqlServer(
        builder.Configuration["ConnectionStrings:SportsStoreConnection"]));
```

This registers the database context class and sets up the relationship with the SQL Server database.
x??

---

#### Creating a Repository Interface
A repository pattern is used to provide a consistent way to access features presented by the database context class. The `IStoreRepository` interface uses `IQueryable<Product>` to allow callers to obtain sequences of `Product` objects.

:p What is the purpose of defining an `IStoreRepository` interface?
??x
The purpose of defining an `IStoreRepository` interface is to provide a consistent way to access database features. It allows callers to interact with data without knowing the details of how it's stored or delivered by the implementation class.

```csharp
namespace SportsStore.Models {
    public interface IStoreRepository {
        IQueryable<Product> Products { get; }
    }
}
```

The `IQueryable<T>` interface is used here, which is derived from `IEnumerable<T>` and represents a collection of objects that can be queried efficiently.
x??

---

#### Implementing the Repository Interface
An implementation class for the repository interface, such as `EFStoreRepository`, maps properties defined in the interface to corresponding properties in the context class. The `Products` property returns a `DbSet<Product>` object, which implements `IQueryable<T>`.

:p How does the `EFStoreRepository` implement the `IStoreRepository` interface?
??x
The `EFStoreRepository` implementation maps the `Products` property defined by the `IStoreRepository` interface onto the `Products` property in the `StoreDbContext` class. This makes it easy to use Entity Framework Core.

```csharp
namespace SportsStore.Models {
    public class EFStoreRepository : IStoreRepository {
        private StoreDbContext context;

        public EFStoreRepository(StoreDbContext ctx) {
            context = ctx;
        }

        public IQueryable<Product> Products => context.Products;
    }
}
```

This setup ensures that the application components can access objects implementing `IStoreRepository` without knowing which implementation class is used.
x??

---

#### Adding a Service for the Repository
A service for the `IStoreRepository` interface using `EFStoreRepository` as the implementation is added to ensure each HTTP request gets its own repository object. This is typical of Entity Framework Core usage.

:p How do you add a service for the `IStoreRepository` in the `Program.cs` file?
??x
To add a service for the `IStoreRepository` interface, use the `AddScoped` method to create a service where each HTTP request gets its own repository object. This is typical of Entity Framework Core usage.

```csharp
builder.Services.AddScoped<IStoreRepository, EFStoreRepository>();
```

This line registers the `EFStoreRepository` as an implementation for the `IStoreRepository` interface.
x??

---

