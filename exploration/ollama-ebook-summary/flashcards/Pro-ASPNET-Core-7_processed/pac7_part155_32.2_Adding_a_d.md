# Flashcards: Pro-ASPNET-Core-7_processed (Part 155)

**Starting Chapter:** 32.2 Adding a data model

---

#### Adding a Data Model to an ASP.NET Core Project

In this section, we are adding a data model for managing employees within departments and their locations using Entity Framework Core. The project setup includes configuring `launchSettings.json` for IIS Express and setting up environment variables for development.

:p What is the purpose of updating the `launchSettings.json` file in an ASP.NET Core project?

??x
The purpose of updating the `launchSettings.json` file is to configure how the application starts. It allows you to specify settings such as which web server (IIS Express) and browser should be launched, along with environment-specific configurations like the `ASPNETCORE_ENVIRONMENT`.

Here's an example snippet from the `launchSettings.json`:

```json
{
  "iisSettings": {
    "windowsAuthentication": false,
    "anonymousAuthentication": true,
    "iisExpress": {
      "applicationUrl": "http://localhost:32165",
      "sslPort": 0
    }
  },
  "profiles": {
    "IIS Express": {
      "commandName": "IISExpress",
      "launchBrowser": true,
      "environmentVariables": {
        "ASPNETCORE_ENVIRONMENT": "Development"
      }
    }
  }
}
```
x??

---

#### Adding NuGet Packages for Entity Framework Core

To use Entity Framework Core, we need to add the necessary NuGet packages. These packages enable us to work with databases using an object-relational mapping (ORM) approach.

:p How do you install the required NuGet packages for Entity Framework Core in a .NET project?

??x
You can install the required NuGet packages by running specific `dotnet` commands in a PowerShell terminal. The commands provided are:

```shell
dotnet add package Microsoft.EntityFrameworkCore.Design --version 7.0.0
dotnet add package Microsoft.EntityFrameworkCore.SqlServer --version 7.0.0
```

These commands will install the Entity Framework Core design tools and the SQL Server provider, respectively.

To ensure you have the correct version of the `dotnet-ef` global tool installed, you can also run:

```shell
dotnet tool uninstall --global dotnet-ef
dotnet tool install --global dotnet-ef --version 7.0.0
```

These commands remove any existing versions and then reinstall the required version.
x??

---

#### Defining Data Models with Entity Framework Core

We need to define several data models that represent entities such as people, departments, and locations. Each model has properties representing attributes and relationships between them.

:p What is the structure of a `Person` class in an Entity Framework Core context?

??x
The `Person` class in the Entity Framework Core context defines key properties for identifying each person and storing their personal information. It also includes navigation properties that link to related entities (Department and Location).

Here's the code snippet for the `Person` class:

```csharp
namespace Advanced.Models {
    public class Person {
        public long PersonId { get; set; }
        public string Firstname { get; set; } = String.Empty;
        public string Surname { get; set; } = String.Empty;
        public long DepartmentId { get; set; }
        public long LocationId { get; set; }
        public Department? Department { get; set; }
        public Location? Location { get; set; }
    }
}
```

This class uses `long` for primary keys and strings for basic properties. The navigation properties (`Department` and `Location`) allow Entity Framework Core to manage relationships between entities.
x??

---

#### Defining the Context Class

The context class, `DataContext`, provides an interface to interact with the database using Entity Framework Core.

:p How do you define a context class in Entity Framework Core?

??x
To define a context class in Entity Framework Core, we create a class that inherits from `DbContext`. This class contains properties of type `DbSet<T>` for each entity type defined in the model. These properties represent tables in the database and enable data access.

Here's how you can define the `DataContext`:

```csharp
namespace Advanced.Models {
    using Microsoft.EntityFrameworkCore;

    public class DataContext : DbContext {
        public DataContext(DbContextOptions<DataContext> opts) 
            : base(opts) { }

        public DbSet<Person> People => Set<Person>();
        public DbSet<Department> Departments => Set<Department>();
        public DbSet<Location> Locations => Set<Location>();
    }
}
```

In this example, `DataContext` has three properties of type `DbSet<T>` corresponding to the `Person`, `Department`, and `Location` classes. The constructor accepts an `DbContextOptions<DataContext>` object to configure database connections.
x??

---

#### Adding Classes for Department and Location

We also define additional classes (`Department` and `Location`) that represent department and location entities in our application.

:p What is the structure of a `Department` class in Entity Framework Core?

??x
The `Department` class defines properties for storing department-related information, including navigation properties to link with people. Here's how it looks:

```csharp
namespace Advanced.Models {
    public class Department {
        public long Departmentid { get; set; }
        public string Name { get; set; } = String.Empty;
        public IEnumerable<Person>? People { get; set; }
    }
}
```

The `Department` class has a primary key (`Departmentid`) and a property for storing the department's name. The `People` navigation property allows for accessing people associated with this department.

This class structure is similar to that of the `Person` class, but it focuses on managing departments rather than individual employees.
x??

---

#### Adding Classes for Location

The final step involves creating a `Location` class that represents geographic locations where employees are stationed.

:p What is the structure of a `Location` class in Entity Framework Core?

??x
The `Location` class defines properties to store location-related information, such as city and state. It also includes navigation properties for linking with people who work there. Here's an example:

```csharp
namespace Advanced.Models {
    public class Location {
        public long LocationId { get; set; }
        public string City { get; set; } = String.Empty;
        public string State { get; set; } = String.Empty;
        public IEnumerable<Person>? People { get; set; }
    }
}
```

This `Location` class has a primary key (`LocationId`) and properties for storing the city and state. The `People` navigation property enables linking employees to their respective locations.

This structure mirrors that of the other classes, ensuring consistent data management practices.
x??

---

#### Context Class Definition for Querying Database

Background context: The provided text describes how to set up a context class in C# to define properties that will be used to query a database. This is part of an example project where `Person`, `Department`, and `Location` data need to be queried from the database.

:p What does the context class do, and what entities are involved?
??x
The context class defines properties for querying specific entities such as `Person`, `Department`, and `Location` from a database. This setup is crucial for interacting with the database using Entity Framework Core (EF Core) in an ASP.NET Core application.

```csharp
public class DataContext : DbContext
{
    public DbSet<Person> People { get; set; }
    public DbSet<Department> Departments { get; set; }
    public DbSet<Location> Locations { get; set; }

    // Additional methods and configurations can be added here.
}
```
x??

---

#### Seed Data for Database Population

Background context: The `SeedData` class is used to populate the database with initial data. This ensures that the application has some test or default data available when it starts, which is useful for development and testing purposes.

:p What does the `SeedDatabase` method do?
??x
The `SeedDatabase` method checks if the database is empty and then seeds it with predefined `Department`, `Location`, and `Person` objects. This ensures that the database has initial data to work with, which can be used for testing or demonstrating the application's functionality.

```csharp
public static void SeedDatabase(DataContext context)
{
    context.Database.Migrate(); // Apply any pending migrations

    if (context.People.Count() == 0 && context.Departments.Count() == 0 && context.Locations.Count() == 0)
    {
        Department d1 = new () { Name = "Sales" };
        Department d2 = new () { Name = "Development" };
        // ... other departments
        context.Departments.AddRange(d1, d2, d3, d4);
        context.SaveChanges();

        Location l1 = new () { City = "Oakland", State = "CA" };
        Location l2 = new () { City = "San Jose", State = "CA" };
        // ... other locations
        context.Locations.AddRange(l1, l2, l3);
        context.SaveChanges();

        Person p1 = new Person { Firstname = "Francesca", Surname = "Jacobs", Department = d2, Location = l1 };
        // ... other persons
        context.People.AddRange(p1, p2, p3, p4, p5, p6, p7, p8, p9);
        context.SaveChanges();
    }
}
```
x??

---

#### Configuring Entity Framework Core in Program.cs

Background context: The `Program.cs` file is where the application's entry point is defined. It also contains configurations and services required to run an ASP.NET Core application, including setting up Entity Framework Core (EF Core) for database interactions.

:p What changes were made to the `Program.cs` file?
??x
Changes were made to configure Entity Framework Core in the `Program.cs` file of the Advanced project. Specifically, the code sets up a data context and seeds the database with initial data if it is empty. This ensures that the application has some test or default data available when it starts.

```csharp
using Microsoft.EntityFrameworkCore;
using Advanced.Models;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddDbContext<DataContext>(opts =>
{
    opts.UseSqlServer(builder.Configuration["ConnectionStrings:PeopleConnection"]);
    opts.EnableSensitiveDataLogging(true); // Enable logging for sensitive data
});

var app = builder.Build();
app.MapGet("/", () => "Hello World.");

// Get a scoped service to seed the database
using var context = app.Services.CreateScope().ServiceProvider.GetRequiredService<DataContext>();
SeedData.SeedDatabase(context);

app.Run();
```
x??

---

---
#### Defining a Connection String in appsettings.json
Background context: To set up an application that interacts with a database, you need to define a connection string. This string is used by Entity Framework Core (EF Core) to establish a connection and perform operations on the database.

The connection string provides details such as the server name, database name, and authentication method. In this case, we use a local SQL Server instance for development purposes.

:p How do you define a connection string in `appsettings.json`?
??x
To define a connection string in `appsettings.json`, you need to add an entry under the `"ConnectionStrings"` section. Here's how it looks:

```json
{
  "ConnectionStrings": {
    "PeopleConnection": "Server=(localdb)\\MSSQLLocalDB;Database=People;MultipleActiveResultSets=True"
  }
}
```

This connection string uses a local SQL Server instance and specifies the database name as `People`. The `MultipleActiveResultSets=True` parameter allows multiple active result sets, which can be useful in certain scenarios.

x??
---
#### Creating an Entity Framework Core Migration
Background context: After defining the connection string, you need to create a migration. A migration is a set of instructions that EF Core uses to change your database schema over time. This process involves generating the necessary SQL commands and applying them to the database.

:p How do you create a new migration in an ASP.NET Core project using Entity Framework Core?
??x
To create a new migration, use the following command in the terminal or PowerShell:

```shell
dotnet ef migrations add Initial
```

This command adds a new migration named `Initial` which includes all changes to your models that are not yet applied.

x??
---
#### Applying the Migration to the Database
Background context: Once you have created a migration, you need to apply it to the database. This step ensures that the schema changes specified in the migration are reflected in the actual database.

:p How do you apply an Entity Framework Core migration to the database?
??x
To apply the migration to the database, use the following command:

```shell
dotnet ef database update
```

This command applies all pending migrations and updates the database schema accordingly. The logging messages displayed by the application will show the SQL commands that are sent to the database.

x??
---
#### Installing Bootstrap CSS Framework with Library Manager
Background context: To style the HTML elements in your ASP.NET Core project, you can use the Bootstrap CSS framework. You need to install this package using the Library Manager command-line tool or by adding it through Visual Studio's Solution Explorer.

:p How do you install the Bootstrap CSS framework using the Library Manager?
??x
To install the Bootstrap CSS framework using the Library Manager, run the following commands in your project folder:

```shell
libman init -p cdnjs
libman install bootstrap@5.2.3 -d wwwroot/lib/bootstrap
```

These commands initialize a new `libman.json` configuration file and then install Bootstrap version 5.2.3 under the `wwwroot/lib/bootstrap` directory.

x??
---
#### Configuring Services and Middleware in Program.cs
Background context: In an ASP.NET Core application, you configure services and middleware to define how your application processes incoming requests. This includes setting up controllers, Razor Pages, and database contexts.

:p How do you configure the services and middleware in `Program.cs` for a project that uses both MVC controllers and Razor Pages?
??x
To configure the services and middleware in `Program.cs`, add the necessary statements as shown:

```csharp
builder.Services.AddControllersWithViews();
builder.Services.AddRazorPages();
builder.Services.AddDbContext<DataContext>(opts =>
{
    opts.UseSqlServer(builder.Configuration["ConnectionStrings:PeopleConnection"]);
    opts.EnableSensitiveDataLogging(true);
});
```

These lines configure the application to use MVC controllers, Razor Pages, and a database context that connects using the connection string defined in `appsettings.json`.

x??
---
#### Creating a HomeController and Index View
Background context: To display data from your database using ASP.NET Core MVC, you need to create a controller and views. The controller handles requests and returns views, which are responsible for rendering HTML.

:p How do you create a simple home controller in the `Controllers` folder that fetches people data?
??x
To create a HomeController, add a class file named `HomeController.cs` in the `Controllers` folder with the following content:

```csharp
using Advanced.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace Advanced.Controllers {
    public class HomeController : Controller {
        private DataContext context;

        public HomeController(DataContext dbContext) {
            context = dbContext;
        }

        public IActionResult Index([FromQuery] string selectedCity) {
            return View(new PeopleListViewModel {
                People = context.People
                    .Include(p => p.Department)
                    .Include(p => p.Location),
                Cities = context.Locations.Select(l => l.City).Distinct(),
                SelectedCity = selectedCity
            });
        }
    }

    public class PeopleListViewModel {
        public IEnumerable<Person> People { get; set; } = Enumerable.Empty<Person>();
        public IEnumerable<string> Cities { get; set; } = Enumerable.Empty<string>();
        public string SelectedCity { get; set; } = String.Empty;
        public string GetClass(string? city) => 
            SelectedCity == city ? "bg-info text-white" : "";
    }
}
```

This controller fetches people data from the database and passes it to a view along with other necessary information.

x??
---
#### Creating an Index View for HomeController
Background context: The view is responsible for rendering HTML based on the model data passed by the controller. In this case, we need to create a Razor view that displays a list of people and allows users to select a city.

:p How do you create a simple index view in the `Views/Home` folder using Razor?
??x
To create an Index.cshtml view for HomeController, add the following content:

```html
@model PeopleListViewModel

<h4 class="bg-primary text-white text-center p-2">People</h4>
<table class="table table-sm table-bordered table-striped">
    <thead>
        <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Dept</th>
            <th>Location</th>
        </tr>
    </thead>
    <tbody>
        @foreach (Person p in Model.People) {
            <tr class="@Model.GetClass(p.Location?.City)">
                <td>@p.PersonId</td>
                <td>@p.Surname, @p.Firstname</td>
                <td>@p.Department?.Name</td>
                <td>@p.Location?.City, @p.Location?.State</td>
            </tr>
        }
    </tbody>
</table>

<form asp-action="Index" method="get">
    <div class="form-group">
        <label for="selectedCity">City</label>
        <select name="selectedCity" class="form-control">
            <option disabled selected>Select City</option>
            @foreach (string city in Model.Cities) {
                <option selected="@(city == Model.SelectedCity)">@city</option>
            }
        </select>
    </div>
    <button class="btn btn-primary mt-2" type="submit">Select</button>
</form>
```

This view iterates through the people data and renders a table with their details. It also includes a dropdown for selecting cities.

x??
---

