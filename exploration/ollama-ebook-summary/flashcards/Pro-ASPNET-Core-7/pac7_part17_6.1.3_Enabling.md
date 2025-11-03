# Flashcards: Pro-ASPNET-Core-7_processed (Part 17)

**Starting Chapter:** 6.1.3 Enabling the MVC Framework

---

---
#### Creating a New Project Using Dotnet CLI
Background context: The provided text explains how to create a new ASP.NET Core web application using the .NET CLI. This involves creating a global.json file, setting up the project with minimal configuration for an ASP.NET Core app, and organizing it within a solution folder.

:p How do you use the dotnet CLI to create a new ASP.NET Core web application?
??x
To create a new ASP.NET Core web application using the .NET CLI, follow these steps:

1. Create a global.json file specifying the SDK version:
   ```sh
   dotnet new globaljson --sdk-version 7.0.100 --output Testing/SimpleApp
   ```
2. Create a solution folder and add it to the solution.
3. Create a web project inside the solution folder:
   ```sh
   dotnet new web --no-https --output Testing/SimpleApp --framework net7.0
   ```
4. Add the project to the solution:
   ```sh
   dotnet sln Testing add Testing/SimpleApp
   ```

This command sequence sets up a basic ASP.NET Core web application, which can be developed and managed within a Visual Studio or Visual Studio Code environment.
x??

---
#### Opening the Project in Visual Studio or VSCode
Background context: The text describes how to open an existing project created via .NET CLI using either Visual Studio or Visual Studio Code. This involves navigating through the file system to find the solution file and opening it.

:p How do you open a project in Visual Studio?
??x
To open a project in Visual Studio, follow these steps:

1. Go to File > Open > Project/Solution.
2. Navigate to the Testing folder and select the `Testing.sln` file.
3. Click on the Open button to open both the solution file and its referenced projects.

:p How do you open a project in Visual Studio Code?
??x
To open a project in Visual Studio Code, follow these steps:

1. Go to File > Open Folder.
2. Navigate to the Testing folder containing your project files.
3. Click on the Select Folder button to load the project into VSCode.

This allows developers to work with the project using their preferred development environment.
x??

---
#### Setting the HTTP Port in launchSettings.json
Background context: The text explains how to configure the HTTP port for an ASP.NET Core application by editing the `launchSettings.json` file. This is crucial for running and testing the application locally.

:p How do you set the HTTP port for an ASP.NET Core application?
??x
To set the HTTP port for an ASP.NET Core application, edit the `launchSettings.json` file located in the `Properties` folder. The configuration should look like this:

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
      "applicationUrl": "http://localhost:5000",
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

This configuration ensures that the application runs on `http://localhost:5000` when debugging or running locally.

:p How do you specify a different HTTP port in launchSettings.json?
??x
To specify a different HTTP port in the `launchSettings.json`, modify the `"applicationUrl"` value to use the desired port number. For example, to run the application on port 5001:

```json
{
  "iisSettings": {
    "windowsAuthentication": false,
    "anonymousAuthentication": true,
    "iisExpress": {
      "applicationUrl": "http://localhost:5001",
      "sslPort": 0
    }
  },
  "profiles": {
    "SimpleApp": {
      "commandName": "Project",
      "dotnetRunMessages": true,
      "launchBrowser": true,
      "applicationUrl": "http://localhost:5001", // Changed port here
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

This change will ensure that the application runs on port 5001 instead of the default 5000.
x??

---

---
#### Enabling the MVC Framework in Program.cs
Background context: In this chapter, the author explains how to enable the MVC (Model-View-Controller) framework for testing purposes. The `Program.cs` file is where services are configured and routes are set up.

:p What does adding `builder.Services.AddControllersWithViews();` do?
??x
This line of code registers the necessary services required by the ASP.NET Core MVC framework, enabling the creation of controllers and views in the application.
x??

---
#### Creating the Data Model
Background context: The author introduces a simple model class called `Product` to create data objects for demonstration purposes. This setup is crucial for testing methods that involve manipulating or displaying data.

:p How does the `Product` class look, and what static method does it contain?
??x
The `Product` class has the following structure:

```csharp
namespace SimpleApp.Models 
{ 
    public class Product 
    { 
        public string Name { get; set; } = string.Empty; 
        public decimal? Price { get; set; }

        public static Product[] GetProducts() 
        { 
            Product kayak = new Product { 
                Name = "Kayak", 
                Price = 275M 
            }; 

            Product lifejacket = new Product { 
                Name = "Lifejacket", 
                Price = 48.95M 
            }; 

            return new Product[] { kayak, lifejacket }; 
        } 
    }
}
```

This class includes a static method `GetProducts` that returns an array of `Product` objects containing two sample products: a kayak and a lifejacket.
x??

---
#### Creating the Controller
Background context: The next step is to create a controller class (`HomeController`) that will handle requests for rendering views. This example uses basic methods to demonstrate the MVC pattern.

:p What does the `Index` action method in `HomeController.cs` do?
??x
The `Index` action method in `HomeController.cs` returns a view result with products obtained from the static method `Product.GetProducts()`:

```csharp
using Microsoft.AspNetCore.Mvc;
using SimpleApp.Models;

namespace SimpleApp.Controllers 
{ 
    public class HomeController : Controller 
    { 
        public ViewResult Index() 
        { 
            return View(Product.GetProducts()); 
        } 
    }
}
```

This method tells ASP.NET Core to render the default view and passes it a collection of `Product` objects.
x??

---
#### Creating the View
Background context: The view file (`Index.cshtml`) is responsible for rendering the data passed from the controller. This example uses Razor syntax to display product information.

:p What does the `Index.cshtml` file contain, and how does it render the products?
??x
The `Index.cshtml` file contains Razor code to display a list of products:

```html
@using SimpleApp.Models @model IEnumerable<Product> 

{ Layout = null; } 
<!DOCTYPE html> 
<html> 
<head>     <meta name="viewport" content="width=device-width" />     <title>Simple App</title> </head> 
<body>     <ul>         @foreach (Product p in Model ?? Enumerable.Empty<Product>()) {             <li>Name: @p.Name, Price: @p.Price</li>         }     </ul> </body> 
</html>
```

The `@model IEnumerable<Product>` line sets the model type to a collection of `Product` objects. The `foreach` loop iterates over each product and displays its name and price in an unordered list (`<ul>`).
x??

---

---
#### Running the ASP.NET Core Application
Background context: The chapter explains how to run an example application using ASP.NET Core. This involves running a specific command to start the application and then making a request via HTTP to see its output.

:p How do you start an ASP.NET Core application?
??x
You start an ASP.NET Core application by running the `dotnet run` command in the terminal, typically from within the project's directory. For example, if your project is named "SimpleApp," navigate to that folder and execute:
```bash
dotnet run
```
x??

---
#### Creating a Unit Test Project for ASP.NET Core Applications
Background context: The text explains how to create a separate Visual Studio project for unit testing in an ASP.NET Core application. This allows the tests to be deployed separately from the application code.

:p How do you create a new XUnit test project using .NET Core?
??x
To create a new XUnit test project, use the `dotnet` command with the template for XUnit:
```bash
dotnet new xunit -o SimpleApp.Tests --framework net7.0
```
This creates a folder named "SimpleApp.Tests" with a basic structure ready for unit tests.

Then add this project to your solution and reference it from the main application project:
```bash
dotnet sln add SimpleApp.Tests
dotnet add SimpleApp.Tests reference SimpleApp
```

If you're using Visual Studio, after running these commands, you might need to reload the solution to see the new test project.

x??

---
#### Removing Default Test Class File
Background context: The default template for a new XUnit project includes a sample test class. This can be confusing and should be removed or adjusted as needed for clarity in testing examples.

:p How do you remove the default UnitTest1.cs file from your project?
??x
You can delete the default `UnitTest1.cs` file using either Solution Explorer or File Explorer:

Using Solution Explorer:
- Open Solution Explorer.
- Navigate to the `SimpleApp.Tests` folder.
- Delete `UnitTest1.cs`.

Or using command line:
```bash
Remove-Item SimpleApp.Tests/UnitTest1.cs
```

This ensures that your testing examples are not affected by a confusing default test case.

x??

---
#### Writing and Running Unit Tests in ASP.NET Core
Background context: The text explains how to write unit tests for an ASP.NET Core application, focusing on using the XUnit framework. It details the Arrange-Act-Assert pattern commonly used in unit testing.

:p What is the Arrange-Act-Assert (A/A/A) pattern in unit testing?
??x
The Arrange-Act-Assert (A/A/A) pattern describes the structure of a unit test:
- **Arrange**: Set up the conditions and objects needed for the test.
- **Act**: Perform the action to be tested.
- **Assert**: Check if the result matches expectations.

Example:
```csharp
public class ProductTests {
    [Fact]
    public void CanChangeProductName() {
        // Arrange
        var p = new Product { Name = "Test", Price = 100M };
        
        // Act
        p.Name = "New Name";
        
        // Assert
        Assert.Equal("New Name", p.Name);
    }
}
```

x??

---

