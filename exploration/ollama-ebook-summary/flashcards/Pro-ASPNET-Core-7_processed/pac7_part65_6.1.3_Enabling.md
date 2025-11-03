# Flashcards: Pro-ASPNET-Core-7_processed (Part 65)

**Starting Chapter:** 6.1.3 Enabling the MVC Framework

---

---
#### Creating a New ASP.NET Core Web Project Using .NET CLI
Background context: This section describes how to create a new web application project using the command-line interface (CLI) of .NET, specifically targeting version 7.0.100 and utilizing the web template for minimal configuration.
:p How do you create a new ASP.NET Core web project using the .NET CLI?
??x
To create a new ASP.NET Core web project, you can use the `dotnet new` command with specific options. Here is the sequence of commands used in the text:

```bash
# Create a global JSON file to specify the SDK version for this solution.
dotnet new globaljson --sdk-version 7.0.100 --output Testing/SimpleApp

# Create a web application project within the specified folder and using .NET 7.0.
dotnet new web --no-https --output Testing/SimpleApp --framework net7.0

# Add a solution file to organize multiple projects under one solution.
dotnet new sln -o Testing

# Add the newly created project to the solution.
dotnet sln Testing add Testing/SimpleApp
```

These commands set up a basic web application named `SimpleApp` with minimal configuration and place it within a solution structure called `Testing`. The resulting project is configured to run on .NET 7.0.

x?
---
#### Opening the Project in Visual Studio or Visual Studio Code
Background context: This section provides instructions for opening an ASP.NET Core project using either Visual Studio or Visual Studio Code, highlighting the process of selecting and opening the solution file.
:p How do you open a project in Visual Studio or Visual Studio Code?
??x
If using **Visual Studio**:

1. Select `File > Open > Project/Solution`.
2. Navigate to the `Testing` folder and select the `Testing.sln` file.
3. Click the Open button to load the solution.

If using **Visual Studio Code**:

1. Select `File > Open Folder`.
2. Browse to the `Testing` folder containing the project files.
3. Select the folder by clicking the "Select Folder" button.

This process ensures that both Visual Studio and Visual Studio Code are aware of all projects within the solution, allowing for easy management and development of the application.
x?
---
#### Setting the HTTP Port in launchSettings.json
Background context: This section explains how to configure the HTTP port where ASP.NET Core applications will listen for incoming requests by editing the `launchSettings.json` file. This configuration is essential for running and testing web applications locally.
:p How do you set the HTTP port for an ASP.NET Core application?
??x
To set the HTTP port, you need to edit the `launchSettings.json` file in the `Properties` folder of your project. The following steps provide guidance:

1. Open the `launchSettings.json` file located in the `Properties` directory.
2. Update the `applicationUrl` field within the `profiles` section for your application.

Here is an example configuration:

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
      "SimpleApp": {
         "commandName": "Project",
         "dotnetRunMessages": true,
         "launchBrowser": true,
         "applicationUrl": "http://localhost:5000", // Set the desired port here
         "environmentVariables": {
            "ASPNETCORE_ENVIRONMENT": "Development"
         }
      },
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

In this example, the `applicationUrl` is set to `"http://localhost:5000"`, which means that the ASP.NET Core application will listen for HTTP requests on port 5000. You can change this value to any desired port number.
x?
---

#### Enabling the MVC Framework in ASP.NET Core

Background context: 
In this section, we are setting up an ASP.NET Core application to use the Model-View-Controller (MVC) framework. This is done by adding necessary services and configuring the application to handle HTTP requests.

Relevant code:
```csharp
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddControllersWithViews(); // Adds MVC controllers and views support

var app = builder.Build();

app.MapDefaultControllerRoute();
app.Run(); // Runs the application
```

:p How do you enable the MVC framework in an ASP.NET Core application?
??x
By adding `AddControllersWithViews()` to the service collection, we register the necessary services for handling HTTP requests with controllers and views. This method call is part of the configuration in the `Program.cs` file.

```csharp
builder.Services.AddControllersWithViews();
```
x??

---

#### Creating a Data Model

Background context: 
The first step in building an application that involves data handling is to define a model class. In this example, we create a simple product model with properties like Name and Price.

Relevant code:
```csharp
namespace SimpleApp.Models {
    public class Product {
        public string Name { get; set; } = string.Empty;
        public decimal? Price { get; set; }
        
        public static Product[] GetProducts() {
            return new Product[] {
                new Product { Name = "Kayak", Price = 275M },
                new Product { Name = "Lifejacket", Price = 48.95M }
            };
        }
    }
}
```

:p How do you define a simple data model in ASP.NET Core?
??x
By creating a class named `Product` with properties like `Name` and `Price`, and a static method called `GetProducts()` that returns an array of products.

```csharp
public class Product {
    public string Name { get; set; } = string.Empty;
    public decimal? Price { get; set; }

    public static Product[] GetProducts() {
        return new Product[] {
            new Product { Name = "Kayak", Price = 275M },
            new Product { Name = "Lifejacket", Price = 48.95M }
        };
    }
}
```
x??

---

#### Creating a Controller and View

Background context: 
Controllers handle the application's logic, while views render the user interface. Here we create a simple controller `HomeController` that returns a list of products from the model.

Relevant code:
```csharp
using Microsoft.AspNetCore.Mvc;
using SimpleApp.Models;

namespace SimpleApp.Controllers {
    public class HomeController : Controller {
        public ViewResult Index() {
            return View(Product.GetProducts());
        }
    }
}
```

:p How do you create a controller and view in ASP.NET Core?
??x
By creating a `HomeController` that inherits from `Controller`, and defining an action method `Index()` which returns a view with data. The `Index()` method calls the static `GetProducts()` method to fetch product data.

```csharp
public class HomeController : Controller {
    public ViewResult Index() {
        return View(Product.GetProducts());
    }
}
```
x??

---

#### Creating a Razor View

Background context: 
The view is responsible for rendering the output of the controller. In this case, we use Razor syntax to iterate over the product list and display each product's name and price.

Relevant code:
```html
@using SimpleApp.Models @model IEnumerable<Product>

@{ Layout = null; }

<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width">
    <title>Simple App</title>
</head>
<body>
    <ul>
        @foreach (Product p in Model ?? Enumerable.Empty<Product>()) {
            <li>Name: @p.Name, Price: @p.Price</li>
        }
    </ul>
</body>
</html>
```

:p How do you create a Razor view for displaying product data?
??x
By creating an `Index.cshtml` file inside the `Views/Home` folder and using Razor syntax to render a list of products. The code iterates over each product in the model and displays its name and price.

```html
@using SimpleApp.Models @model IEnumerable<Product>

@{ Layout = null; }

<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width">
    <title>Simple App</title>
</head>
<body>
    <ul>
        @foreach (Product p in Model ?? Enumerable.Empty<Product>()) {
            <li>Name: @p.Name, Price: @p.Price</li>
        }
    </ul>
</body>
</html>
```
x??

---

---
#### Starting an ASP.NET Core Application
Background context: The provided text explains how to start an ASP.NET Core application using the `dotnet run` command, which runs the application on a local server. This is done within the SimpleApp folder.

:p How do you start an ASP.NET Core application in the given example?
??x
To start the ASP.NET Core application, navigate to the SimpleApp folder and execute the following command:
```
dotnet run
```
This command starts the development server, which listens on `http://localhost:5000`. You can open a web browser or use `curl` to access this URL.
x??

---
#### Creating a Unit Test Project for ASP.NET Core Applications
Background context: The text describes the process of creating a separate Visual Studio project dedicated to holding unit tests. It also explains that using separate projects helps in deployment by not including test files with the application.

:p What is the purpose of creating a separate project for unit tests in an ASP.NET Core application?
??x
The purpose of creating a separate project for unit tests is to keep the testing code organized and separate from the production application code. This separation allows for easier management, deployment, and maintenance. Unit test projects can be created using templates provided by the .NET Core SDK, which support popular testing frameworks like MSTest, NUnit, or xUnit.
x??

---
#### Using XUnit as a Test Framework
Background context: The text mentions that the .NET Core SDK includes templates for unit tests using three popular frameworks—MSTest, NUnit, and xUnit. It recommends starting with xUnit due to its ease of use.

:p Which testing framework is recommended to start with in ASP.NET Core applications?
??x
The recommendation is to start with xUnit because it is the test framework that is found to be easiest to work with. However, MSTest and NUnit are also available as alternatives.
x??

---
#### Naming Conventions for Test Projects
Background context: The text explains that unit test projects follow a naming convention where the project name includes ".Tests".

:p What is the standard naming convention for unit test projects in ASP.NET Core applications?
??x
The standard naming convention for unit test projects in ASP.NET Core applications is to include ".Tests" at the end of the application's name. For example, if your application is named "SimpleApp", the corresponding test project should be named "SimpleApp.Tests".
x??

---
#### Creating an XUnit Test Project
Background context: The text provides a command sequence for creating and configuring an xUnit test project in the Testing folder.

:p How do you create an xUnit test project using the .NET Core CLI?
??x
To create an xUnit test project named "SimpleApp.Tests", you can use the following commands:
```
dotnet new xunit -o SimpleApp.Tests --framework net7.0
dotnet sln add SimpleApp.Tests
dotnet add SimpleApp.Tests reference SimpleApp
```
These commands first generate a new xUnit test project in the `SimpleApp.Tests` directory, then add this project to an existing solution and create a reference between the application and the test project.
x??

---
#### Removing Default Test Classes
Background context: The text explains that when using the template for creating unit tests, a default class file "UnitTest1.cs" is created which can be confusing. It provides options on how to remove this default class.

:p How do you remove the default `UnitTest1.cs` file from your test project?
??x
You can either manually delete the "UnitTest1.cs" file using the Solution Explorer or File Explorer, or use the following command:
```
Remove-Item SimpleApp.Tests/UnitTest1.cs
```
This command removes the specified file and cleans up the initial default configuration.
x??

---
#### Writing Unit Tests in xUnit
Background context: The text details how to write unit tests using xUnit by defining test methods with the `[Fact]` attribute. It explains the Arrange, Act, Assert (A/A/A) pattern.

:p What is the purpose of the `[Fact]` attribute in an xUnit test?
??x
The `[Fact]` attribute in xUnit indicates that a method is intended to be executed as a unit test. This attribute marks the method as executable and ensures it gets run by the testing framework.
x??

---
#### Example Unit Test Class
Background context: The text provides an example of a simple `ProductTests.cs` class containing two test methods for a `Product` model.

:p What are the names of the test methods in the provided `ProductTests.cs` class?
??x
The test methods in the provided `ProductTests.cs` class are:
- `CanChangeProductName`
- `CanChangeProductPrice`
x??

---
#### Understanding Arrange, Act, Assert Pattern
Background context: The text explains that unit tests follow a pattern called arrange, act, assert (A/A/A). It breaks down what each part of the pattern means and provides examples.

:p What does the "Arrange" step in a unit test involve?
??x
The "Arrange" step in a unit test involves setting up the conditions for the test to run. This includes initializing objects and preparing any data that will be used during the test.
x??

---
#### Example of Arrange, Act, Assert Pattern
Background context: The text provides an example of a simple unit test where `Product` is being tested.

:p How does the "Act" step work in the provided unit test example?
??x
The "Act" step in the provided unit test involves performing the action that needs to be verified. For instance, it changes the `Name` and `Price` properties of a `Product` object.
x??

---

