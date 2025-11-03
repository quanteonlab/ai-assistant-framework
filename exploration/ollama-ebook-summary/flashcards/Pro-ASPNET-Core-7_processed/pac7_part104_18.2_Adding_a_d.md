# Flashcards: Pro-ASPNET-Core-7_processed (Part 104)

**Starting Chapter:** 18.2 Adding a data model

---

#### Creating an ASP.NET Core Project
Background context: This section explains how to create a basic ASP.NET Core project using .NET CLI commands. It covers setting up a solution and adding a web application to it, which will serve as the foundation for the examples used throughout this part of the book.

:p How do you start creating a new ASP.NET Core project?
??x
To start creating a new ASP.NET Core project, you first need to ensure that .NET SDK is installed on your system. Then, open a PowerShell command prompt and run the following commands as described in Listing 18.1:

```powershell
dotnet new globaljson --sdk-version 7.0.100 --output WebApp
dotnet new web --no-https --output WebApp --framework net7.0
dotnet new sln -o WebApp
dotnet sln WebApp add WebApp
```
x??

---

#### Adding a Solution and Project to it
Background context: After creating the project, you create a solution that includes your web application.

:p What command do you use to add a project to an existing solution?
??x
To add a project to an existing solution, you can use the `dotnet sln` command followed by the path or name of the solution and then the path or name of the project you want to add. For example:

```powershell
dotnet sln WebApp.sln add WebApp
```
x??

---

#### Configuring Launch Settings
Background context: This section explains how to configure launch settings such as application URLs, environment variables, and browser launching in `launchSettings.json`.

:p How do you set the port in `launchSettings.json`?
??x
To set the HTTP port in `launchSettings.json`, you need to modify the JSON file located at `WebApp/Properties`. Specifically, you change the value of `"applicationUrl"` under the profile section. For instance:

```json
{
  "iisSettings": {
    "windowsAuthentication": false,
    "anonymousAuthentication": true,
    "iisExpress": {
      "applicationUrl": "http://localhost:5000",
      "sslPort": 0
    }
  },
  "profiles": {
    "WebApp": {
      "commandName": "Project",
      "dotnetRunMessages": true,
      "launchBrowser": false,
      "applicationUrl": "http://localhost:5010", // Changed port here
      "environmentVariables": {
        "ASPNETCORE_ENVIRONMENT": "Development"
      }
    }
  }
}
```
x??

---

#### Disabling Automatic Browser Launching
Background context: This part of the text explains how to disable automatic browser launching during development.

:p How do you disable automatic browser launching in `launchSettings.json`?
??x
To disable automatic browser launching, modify the `"launchBrowser"` setting under the profile section in `launchSettings.json`. Set it to `false` as shown below:

```json
{
  "profiles": {
    "WebApp": {
      "commandName": "Project",
      "dotnetRunMessages": true,
      "launchBrowser": false, // Changed this value from true to false
      "applicationUrl": "http://localhost:5010"
    }
  }
}
```
x??

---

#### Adding NuGet Packages to the Project
Background context: To use Entity Framework Core for database operations, we need to add specific NuGet packages. This step is crucial as it allows us to interact with SQL Server databases using LINQ and other features provided by EF Core.

:p How do you add necessary NuGet packages using dotnet CLI?
??x
You can add the required NuGet packages using the following commands in a PowerShell command prompt:
```shell
dotnet add package Microsoft.EntityFrameworkCore.Design --version 7.0.0  
dotnet add package Microsoft.EntityFrameworkCore.SqlServer --version 7.0.0
```
These commands ensure that you have the correct versions of Entity Framework Core and its SQL Server provider installed for your project.
x??

---

#### Using Visual Studio to Add Packages
Background context: Alternatively, if you are using Visual Studio, adding NuGet packages can be done through the IDE's built-in interface. This method is useful when you prefer a graphical approach.

:p How do you add NuGet packages via Visual Studio?
??x
In Visual Studio, you can manage NuGet packages by selecting `Project > Manage NuGet Packages`. From here, you can search for and install the necessary packages.
x??

---

#### Installing Global Tool Package
Background context: To work with Entity Framework Core migrations, it is essential to have a global tool package installed. This step ensures that you can use commands like `dotnet ef` to manage migrations.

:p How do you install the global tool package for Entity Framework Core?
??x
To install the required global tool package, run these commands in a PowerShell window:
```shell
dotnet tool uninstall --global dotnet-ef
dotnet tool install --global dotnet-ef --version 7.0.0
```
These commands first remove any existing version of `dotnet-ef` and then reinstall the correct version to ensure compatibility with your project.
x??

---

#### Creating Data Model Classes
Background context: The data model classes are crucial for defining how entities will be stored and retrieved from the database. These classes use properties that correspond to columns in a SQL Server table, along with navigation properties for relationships between different entities.

:p How do you create data model classes for the project?
??x
Create three related classes named `Category`, `Supplier`, and `Product` within a folder called `Models`. Here is an example of how to define the `Category.cs` class:
```csharp
namespace WebApp.Models {
    public class Category {
        public long CategoryId { get; set; }
        public required string Name { get; set; }
        public IEnumerable<Product>? Products { get; set; }
    }
}
```
Each class has properties that map to database columns, with `Category`, `Supplier`, and `Product` defining the relationships between them.
x??

---

#### Defining Product Class
Background context: The `Product` class is central in our data model as it defines a product entity. It includes key properties like `ProductId`, `Name`, and `Price`, along with navigation properties to link products with categories and suppliers.

:p How do you define the `Product` class?
??x
Here is how to define the `Product.cs` class:
```csharp
using System.ComponentModel.DataAnnotations.Schema;

namespace WebApp.Models {
    public class Product {
        public long ProductId { get; set; }
        public required string Name { get; set; }
        [Column(TypeName = "decimal(8, 2)")]
        public decimal Price { get; set; }
        public long CategoryId { get; set; }
        public Category? Category { get; set; }
        public long SupplierId { get; set; }
        public Supplier? Supplier { get; set; }
    }
}
```
This class includes properties that correspond to database columns, ensuring proper mapping and validation.
x??

---

#### Adding Navigation Properties
Background context: Navigation properties in the data model allow for querying related entities. For example, using `Category` and `Supplier` navigation properties on the `Product` class enables fetching associated category and supplier information.

:p What is the purpose of adding navigation properties to the `Product` class?
??x
Navigation properties like `Category` and `Supplier` in the `Product` class enable you to query related entities. For example, if you want to find all products within a specific category or from a particular supplier, these navigation properties make it straightforward.
x??

---

---
#### Entity Framework Core Context Class
Background context: The Entity Framework Core context class is a crucial part of managing interactions with the database. It provides the main entry point for accessing and manipulating data through object-oriented models.

:p What is the purpose of the `DataContext` class in this example?
??x
The `DataContext` class serves as an Entity Framework Core DbContext, which manages communication between the application code and the database. It defines properties that represent collections of entities (like `Products`, `Categories`, and `Suppliers`) and provides methods for interacting with these entities.

```csharp
using Microsoft.EntityFrameworkCore;

namespace WebApp.Models {
    public class DataContext : DbContext {
        public DataContext(DbContextOptions<DataContext> opts) : base(opts) { }
        public DbSet<Product> Products => Set<Product>();
        public DbSet<Category> Categories => Set<Category>();
        public DbSet<Supplier> Suppliers => Set<Supplier>();
    }
}
```

x??
---
#### Seed Data
Background context: Seed data is used to populate the database with initial or test data. This is particularly useful during development and testing phases.

:p How does seed data help in populating the database?
??x
Seed data helps initialize the database with sample records, ensuring that there is enough data available for testing and demonstration purposes without manually entering each record individually. 

```csharp
using Microsoft.EntityFrameworkCore;

namespace WebApp.Models {
    public static class SeedData {
        public static void SeedDatabase(DataContext context) {
            context.Database.Migrate();
            if (context.Products.Count() == 0 && context.Suppliers.Count() == 0 && context.Categories.Count() == 0) {
                Supplier s1 = new Supplier { Name = "Splash Dudes", City = "San Jose" };
                Supplier s2 = new Supplier { Name = "Soccer Town", City = "Chicago" };
                // More suppliers, categories, and products...
            }
        }
    }
}
```

x??
---
#### Configuring EF Core Services in Program.cs
Background context: The `Program.cs` file is where the application's entry point and configuration are defined. Here, Entity Framework Core services are set up to interact with the database.

:p What changes were made to the `Program.cs` file for configuring Entity Framework Core?
??x
Changes to `Program.cs` included setting up the `DataContext` service and enabling sensitive data logging. The `AddDbContext` method was used to configure the connection string, while `EnableSensitiveDataLogging(true)` ensures that detailed SQL queries are logged.

```csharp
using Microsoft.EntityFrameworkCore;
using WebApp.Models;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddDbContext<DataContext>(opts => {
    opts.UseSqlServer(builder.Configuration["ConnectionStrings:ProductConnection"]);
    opts.EnableSensitiveDataLogging(true);
});
```

x??
---
#### Connection String in appsettings.json
Background context: The connection string defines the database to be used by the application and how it should connect. It is crucial for establishing a data source and ensuring that the application can communicate with the correct database.

:p How does the `appsettings.json` file set up the database connection?
??x
The `appsettings.json` file contains the necessary configuration settings, including the connection string to the SQL Server database.

```json
{
    "ConnectionStrings": {
        "ProductConnection": "Server=(localdb)\\MSSQLLocalDB;Database=Products;MultipleActiveResultSets=True"
    }
}
```

x??
---

