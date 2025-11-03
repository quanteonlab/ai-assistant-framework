# Flashcards: Pro-ASPNET-Core-7_processed (Part 19)

**Starting Chapter:** 7.1.1 Creating the unit test project

---

#### Unit Testing Overview
Background context: This section explains how unit tests are typically organized and executed within an application. It highlights the benefits of using a dedicated project for unit testing, as well as the importance of isolating components to ensure that each part can be tested independently.

:p What is the typical structure of unit tests mentioned in this text?
??x
The typical structure involves defining unit tests within a separate and dedicated unit test project. This setup helps maintain and extend the codebase more effectively by providing robust support for unit testing.
x??

---

#### Arrange/Act/Assert Pattern
Background context: The Arrange/Act/Assert pattern is a common structure used in writing unit tests to ensure that each part of the system under test (SUT) is isolated and tested independently. This pattern helps in organizing the test by preparing the test environment, executing the code, and validating the results.

:p What are the three steps involved in the Arrange/Act/Assert pattern for a unit test?
??x
The three steps in the Arrange/Act/Assert pattern are:
1. **Arrange**: Set up the initial state or context of the system.
2. **Act**: Perform an action on the system, such as calling a method.
3. **Assert**: Check that the result of the action meets expectations.

Example code for this pattern might look like:

```csharp
[Test]
public void ExampleTestMethod()
{
    // Arrange: Set up test data and objects
    var mockObject = new Mock<MockClass>();
    
    // Act: Perform an action with the setup object
    var result = mockObject.Object.MethodToTest();
    
    // Assert: Validate the outcome of the method call
    Assert.AreEqual(expectedResult, result);
}
```
x??

---

#### ASP.NET Core Application Overview
Background context: This chapter introduces the creation of a more realistic e-commerce application called SportsStore. The application will include features such as an online product catalog, shopping cart, checkout process, and administrative area for managing products.

:p What is the primary goal of building the SportsStore application?
??x
The primary goal of building the SportsStore application is to provide a realistic example of real ASP.NET Core development by creating a simplified but functional e-commerce platform. This includes features like an online product catalog, shopping cart, checkout process, and administrative area for managing products.
x??

---

#### Unit Testing in SportsStore Application
Background context: Throughout the development of the SportsStore application, various unit tests will be created to isolate and test different components of the ASP.NET Core framework. These tests help ensure that each part of the system works as expected without relying on external dependencies.

:p What is the main purpose of including sections on unit testing in the SportsStore application?
??x
The main purpose of including sections on unit testing in the SportsStore application is to demonstrate how different components can be isolated and tested effectively. This approach ensures that the ASP.NET Core framework's features are used correctly and that the overall system behaves as intended.
x??

---

#### Creating Projects for SportsStore
Background context: The initial steps involve setting up a minimal ASP.NET Core project and progressively adding necessary features using commands in PowerShell. These commands create solution folders, add projects, and manage dependencies.

:p What command is used to add the SportsStore project to the solution?
??x
The command used to add the SportsStore project to the solution is:

```powershell
dotnet sln SportsSln add SportsSln/SportsStore
```

This command adds the SportsStore project folder to the existing solution (SportsSln).
x??

---

#### Minimal ASP.NET Core Project Setup
Background context: The setup involves creating a minimal ASP.NET Core project with specific template and framework configurations using dotnet CLI commands. These commands ensure that the project is set up correctly for development.

:p What command is used to create a new web project in .NET 7?
??x
The command used to create a new web project in .NET 7 is:

```powershell
dotnet new web --no-https --output SportsSln/SportsStore --framework net7.0
```

This command creates a new ASP.NET Core web application project with the specified framework version and outputs it into the `SportsSln/SportsStore` folder.
x??

---

#### Creating a Unit Test Project
Background context: The process of creating a unit test project for an application involves setting up a new project using a testing framework like xUnit. This ensures that individual units or components of the code can be tested independently.

:p How do you create a unit test project in .NET using xUnit?
??x
To create a unit test project, use the `dotnet new` command with the xUnit template.
```sh
dotnet new xunit -o SportsSln/SportsStore.Tests --framework net7.0
```
This command creates a new folder and adds the necessary files for testing.

You then add this project to your solution:
```sh
dotnet sln SportsSln add SportsSln/SportsStore.Tests
```
And reference it in your application project:
```sh
dotnet add SportsSln/SportsStore.Tests reference SportsSln/SportsStore
```
x??

---

#### Installing the Moq Package
Background context: The Moq package is a popular mocking framework used for unit testing. Mock objects are used to simulate dependencies, allowing tests to be isolated from external systems or services.

:p How do you install the Moq package into your test project?
??x
To install the Moq package, use the `dotnet add` command with the `package` and `version` parameters:
```sh
dotnet add SportsSln/SportsStore.Tests package Moq --version 4.18.4
```
This command installs the specified version of Moq into your unit test project.

x??

---

#### Opening the Projects in Visual Studio Code
Background context: To open and work on the projects created, you need to import the folder containing the solution file into an integrated development environment (IDE) like Visual Studio Code. This process allows you to view and build all components of the application.

:p How do you open a project or solution in Visual Studio Code?
??x
To open a project or solution in Visual Studio Code, select `File > Open Folder`, navigate to the SportsSln folder, and click the "Select Folder" button. Visual Studio Code will discover the solution and project files and prompt you to install required assets.

If prompted, click Yes to install the necessary tools. Select SportsStore as the project to run if given a choice.

x??

---

#### Configuring HTTP Port in ASP.NET Core
Background context: The `launchSettings.json` file configures how your application will be launched during development, including the port number on which it listens for incoming requests. This is crucial for debugging and testing purposes.

:p How do you configure the HTTP port for an ASP.NET Core project?
??x
To set the HTTP port in the `SportsStore/Properties/launchSettings.json` file, modify the JSON content as shown:
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
    "SportsStore": {
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
This configuration specifies that the application will listen on `http://localhost:5000`.

x??

---

#### Creating Application Project Folders
Background context: Organizing code into logical folders is a best practice in software development. This helps in managing complexity and reusability of components.

:p What are the key folders you need to create for an ASP.NET Core project?
??x
You should create the following folders:
- `Models`: Contains data models and access classes.
- `Controllers`: Contains controller classes that handle HTTP requests.
- `Views`: Contains all Razor files, grouped into subfolders like `Home` and `Shared`.

To create these folders in Visual Studio Code or Visual Studio, right-click on the `SportsStore` project in Solution Explorer and select "Add > New Folder".

x??

---

#### Preparing Services and Request Pipeline
Background context: The `Program.cs` file is where you configure services and middleware for an ASP.NET Core application. This includes setting up dependency injection, routing, and serving static files.

:p How do you configure the basic features of an ASP.NET Core application in the Program.cs file?
??x
In the `Program.cs` file, use the following configuration:
```csharp
var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllersWithViews(); // Sets up MVC services

var app = builder.Build();

app.UseStaticFiles(); // Enables serving static files from wwwroot

app.MapDefaultControllerRoute(); // Maps default routes to controllers

app.Run(); // Starts the application
```
This code sets up the basic features needed for an ASP.NET Core web application.

x??

---

#### Configuring the Razor View Engine
Background context: The Razor view engine processes .cshtml files to generate HTML responses. Proper configuration of this engine ensures that views are easier to create and maintain.

:p How do you configure the Razor view engine in your ASP.NET Core project?
??x
To configure the Razor view engine, add `_ViewImports.cshtml` and `_ViewStart.cshtml` files:
- `_ViewImports.cshtml`: Contains `@using` statements for namespaces.
```cshtml
@using SportsStore.Models @addTagHelper *, Microsoft.AspNetCore.Mvc.TagHelpers
```
- `_ViewStart.cshtml`: Sets the default layout file.
```cshtml
@{ Layout = "_Layout"; }
```
Also, create a layout file in `Views/Shared/_Layout.cshtml` with HTML structure and placeholders for rendering content:
```html
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width" />
    <title>SportsStore</title>
</head>
<body>
    <div>
        @RenderBody()
    </div>
</body>
</html>
```
x??

---
#### Creating a HomeController and Index View
Background context: This section explains how to create a minimal controller for handling HTTP requests in an ASP.NET Core application. It covers creating a class file named `HomeController.cs` in the `SportsStore/Controllers` folder and defining its functionality through the `Index()` action method.

:p What is the purpose of the `HomeController` class?
??x
The purpose of the `HomeController` class is to handle HTTP requests, specifically for the "Home" controller. The `Index()` action method returns a view that will be rendered by ASP.NET Core when a request is made to the appropriate URL.

```csharp
using Microsoft.AspNetCore.Mvc;

namespace SportsStore.Controllers {
    public class HomeController: Controller {
        public IActionResult Index() => View();
    }
}
```
x??

---
#### Mapping Default Routes and URLs
Background context: This concept explains how routing works in ASP.NET Core. The `MapDefaultControllerRoute` method is used to match incoming HTTP requests to the appropriate controller action.

:p How does the `MapDefaultControllerRoute` method work?
??x
The `MapDefaultControllerRoute` method configures the default routes for a given application, allowing URLs like `/Home/Index` to be matched to specific actions in controllers. In this case, it sets up a route that maps requests to the `Index()` action of the `HomeController`.

```csharp
// Example configuration code
public void Configure(IApplicationBuilder app) {
    app.UseRouting();
    app.UseEndpoints(endpoints => {
        endpoints.MapDefaultControllerRoute();  // This method call is implied in the context.
    });
}
```
x??

---
#### Creating a Product Model Class
Background context: This section details how to create and define a simple model class for representing products in an e-commerce application. The `Product` class will be used to store data about individual items available for sale.

:p What is the purpose of the `Product` class?
??x
The purpose of the `Product` class is to represent a product entity within the application, storing necessary attributes such as name, description, price, and other relevant information. This model class will be used throughout the application to manage and manipulate data related to products.

```csharp
using System.ComponentModel.DataAnnotations.Schema;

namespace SportsStore.Models {
    public class Product {
        // Class definition goes here.
    }
}
```
x??

---
#### Rendering a View with Razor Syntax
Background context: This part explains how to create a simple HTML view using Razor syntax. The `Index.cshtml` file is created in the `Views/Home` folder, which contains the basic structure of an HTML page that will be rendered by ASP.NET Core.

:p What does the `Index.cshtml` file contain?
??x
The `Index.cshtml` file contains a simple HTML structure with Razor syntax for embedding dynamic content. It includes a heading to welcome users to the SportsStore application.

```html
<h4>Welcome to SportsStore</h4>
```
x??

---

