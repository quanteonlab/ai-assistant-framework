# Flashcards: Pro-ASPNET-Core-7_processed (Part 54)

**Starting Chapter:** 3.2.1 Preparing the project

---

#### Creating an ASP.NET Core Project
Background context: In this section, you'll learn how to create a basic ASP.NET Core project and set up the initial structure for a simple web application. The commands provided will ensure that your project is configured correctly with the required .NET framework version.

:p What are the steps to create a new ASP.NET Core MVC project named "PartyInvites" using the `dotnet` CLI?
??x
To create a new ASP.NET Core MVC project named "PartyInvites," you need to run several commands in PowerShell. First, use `dotnet new globaljson --sdk-version 7.0.100 --output PartyInvites` to set up the correct SDK version for your project. Then, use `dotnet new mvc --no-https --output PartyInvites --framework net7.0` to create a new ASP.NET Core MVC application with .NET 7.0. Finally, run `dotnet new sln -o PartyInvites` to create a solution file named "PartyInvites.sln" and add the project using `dotnet sln PartyInvites add PartyInvites`.

```powershell
# Example PowerShell commands
dotnet new globaljson --sdk-version 7.0.100 --output PartyInvites
dotnet new mvc --no-https --output PartyInvites --framework net7.0
dotnet new sln -o PartyInvites
dotnet sln PartyInvites add PartyInvites
```
x??

---

#### Setting the Port in launchSettings.json
Background context: The `launchSettings.json` file is used to configure how your application starts and listens for HTTP requests. By default, it sets up a local development environment with specific port numbers.

:p How do you change the port that will be used by the ASP.NET Core application?
??x
To change the port in the `launchSettings.json` file, navigate to the `Properties` folder of your project and open the `launchSettings.json`. Replace the existing content with the JSON provided in Listing 3.2, which sets the HTTP application URL to "http://localhost:5000".

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
     "PartyInvites": {
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
x??

---

#### HomeController and Index.cshtml
Background context: The `HomeController` is responsible for handling actions related to the home page. In this case, it will return a view that can be customized with specific content.

:p What does the `Index()` method in the `HomeController` do?
??x
The `Index()` method in the `HomeController` returns a `ViewResult`, which tells ASP.NET Core to render a specified view for the user. This is a basic setup to get you started, allowing further customization of what's displayed on the home page.

```csharp
using Microsoft.AspNetCore.Mvc;
namespace PartyInvites.Controllers {
    public class HomeController : Controller {
        public IActionResult Index() { 
            return View();
        }
    }
}
```
x??

---

#### Customizing the Home Page with Index.cshtml
Background context: The `Index.cshtml` file in the `Views/Home` folder is where you can add custom content to be displayed on your home page. This example sets up a simple HTML structure and adds some placeholder text.

:p What does the updated `Index.cshtml` contain?
??x
The updated `Index.cshtml` contains basic HTML structure and a welcome message for party invitees. It uses Razor syntax to define layout null, set the document type, meta tags, title, and body content.

```razor
@{ Layout = null; }
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width" />
    <title>Party.</title>
</head>
<body>
    <div>
        <div>
            We're going to have an exciting party.<br />
            (To do: sell it better. Add pictures or something.)
        </div>
    </div>
</body>
</html>
```
x??

---

#### Compiling and Running a Project
Background context: The text explains how to compile and run an ASP.NET Core application using the `dotnet watch` command. This process allows for automatic recompilation and updating of the browser when changes are made, which is useful during development.

:p How do you compile and run the project in this example?

??x
To compile and run the project, use the following command:
```
dotnet watch
```
This command watches for changes in your code and automatically rebuilds the application. If a change is detected, it will update the running application without requiring manual restart.

Additionally, if you make a mistake or there are issues that prevent automatic updates, you can restart the application using the prompt:
```
watch : Do you want to restart your app
- Yes (y) / No (n) / Always (a) / Never (v)?
```

Selecting `Always (a)` will ensure that the project is always rebuilt automatically.
x??

---

#### Adding a Data Model in ASP.NET Core
Background context: In an ASP.NET Core application, the data model represents real-world objects and processes. For this project, you need to create a simple domain class called `GuestResponse` that will be used for storing RSVP information.

:p What is the purpose of creating a data model in an ASP.NET Core application?

??x
The purpose of creating a data model in an ASP.NET Core application is to represent real-world objects and processes. In this context, the `GuestResponse` class serves as a domain object that will store information about RSVPs for guests.

Here’s how you can create and define the `GuestResponse` class:
```csharp
namespace PartyInvites.Models {
    public class GuestResponse {
        public string? Name { get; set; }
        public string? Email { get; set; }
        public string? Phone { get; set; }
        public bool? WillAttend { get; set; }
    }
}
```
The properties of the `GuestResponse` class are nullable to accommodate cases where some information might not be provided.
x??

---

#### Adding an Action Method and View
Background context: To add functionality, you need to define action methods in your controller that will handle requests. In this case, a new action method is added to the `HomeController` to serve an RSVP form.

:p How do you add an action method for handling an RSVP form in ASP.NET Core?

??x
To add an action method for handling an RSVP form, you need to define it within the appropriate controller. For example, adding a method named `RsvpForm()` in the `HomeController`:

```csharp
using Microsoft.AspNetCore.Mvc;

namespace PartyInvites.Controllers {
    public class HomeController : Controller {
        // Existing actions

        public ViewResult RsvpForm() {
            return View();
        }
    }
}
```

This action method returns a `ViewResult`, which tells the Razor view engine to look for a corresponding view named `RsvpForm.cshtml` in the `Views/Home` folder.

You then create this view by adding a new Razor view with the specified name, as shown below:
```html
@{ Layout = null; }
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width" />
    <title>RsvpForm</title>
</head>
<body>
    <div>
        This is the RsvpForm.cshtml View
    </div>
</body>
</html>
```
x??

---

#### Handling Different Result Types in Controllers
Background context: ASP.NET Core controllers can return different result types. For this example, a `ViewResult` was returned from an action method to render a view.

:p What are the different result types that a controller in ASP.NET Core can return?

??x
In ASP.NET Core, controllers can return various result types depending on what kind of response is needed. Some common result types include:

- **ViewResult**: To render a Razor view.
- **JsonResult**: To return JSON data.
- **RedirectResult**: To redirect the user to another URL.
- **FileResult**: To send a file to the client.

For example, in the `HomeController`, you defined an action method that returns a `ViewResult`:
```csharp
public ViewResult RsvpForm() {
    return View();
}
```

This tells the view engine to render the `RsvpForm.cshtml` view located at `Views/Home/RsvpForm.cshtml`.
x??

---

#### Using Layouts in Razor Views
Background context: The layout file (`_Layout.cshtml`) is used as a base for other views. In this example, you have set the layout to null using `@{ Layout = null; }`, which means no layout will be applied to this view.

:p How do you create and use a layout in Razor views?

??x
In ASP.NET Core, layouts are used to share common HTML content across multiple views. You can define a base layout with shared sections such as headers and footers. To apply a layout to a view, set the `Layout` property within the view.

For example:
```html
@{ Layout = "_Layout"; }
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width" />
    <title>RsvpForm</title>
</head>
<body>
    <!-- View-specific content here -->
</body>
</html>
```

In this example, the `RsvpForm.cshtml` view is using a layout specified in `_Layout.cshtml`, which includes shared sections like headers and footers.

If you want to exclude any layout from being applied (like for simple static HTML), you can set the `Layout` property to null:
```html
@{ Layout = null; }
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width" />
    <title>RsvpForm</title>
</head>
<body>
    This is the RsvpForm.cshtml View
</body>
</html>
```

This setup ensures that no layout file is used, and only the view's own content is displayed.
x??

---

#### ASP.NET Core Tag Helpers
Background context: In an ASP.NET Core application, tag helpers are used to simplify HTML generation and enable better separation of concerns by allowing server-side logic within your views. Tag helpers transform attributes on a tag into server-side code that can modify the behavior or content of the tag at runtime.

:p What is the role of the `asp-action` attribute in the context of ASP.NET Core?
??x
The `asp-action` attribute is used to generate URLs for action methods within your application. It dynamically creates an `href` attribute for anchor (`<a>`) elements based on the action method and controller specified.

For example, if you have a controller named `Home`, and inside that controller, there's an action method called `RsvpForm`. When you use `asp-action="RsvpForm"` in your view, it will generate a URL like `/Home/RsvpForm`.

```html
<a asp-action="RsvpForm">RSVP Now</a>
```

This approach allows the application configuration to change without affecting any views, providing flexibility and maintainability.
x??

---
#### Model Binding in ASP.NET Core
Background context: In an MVC (Model-View-Controller) framework like ASP.NET Core, model binding is used to automatically bind form data sent from a view into objects that can be passed to your action methods. The `@model` directive at the top of a Razor view file specifies the type of object that will receive the bound data.

:p How does the `@model` directive in a Razor view work?
??x
The `@model` directive in a Razor view sets the model for the current view, which is an instance of the specified class. This allows the server-side code to access and manipulate this model directly within the view. For example, if you have defined a `GuestResponse` object as your model, it means that all form inputs will be automatically bound to properties of `GuestResponse`.

```csharp
@model PartyInvites.Models.GuestResponse
```

In this case, any input fields like name, email, and phone would be mapped directly to the corresponding properties in the `GuestResponse` class.
x??

---
#### Form Submission Handling in ASP.NET Core
Background context: In an ASP.NET Core application, you can use Razor tag helpers to create HTML forms that automatically handle submission and model binding. The form's action is determined by the value of the `asp-action` attribute, which points to the target controller and action method.

:p How does the `<form>` element with `asp-action` work in an ASP.NET Core application?
??x
The `<form>` element with the `asp-action` attribute works as follows:

1. It automatically generates a form submission URL that points to the specified action method.
2. When the form is submitted, the data from the input fields are bound to the model object and passed to the controller's action method.

Here's an example of how it looks in code:
```html
<form asp-action="RsvpForm" method="post">
    <div>
        <label asp-for="Name">Your name:</label>
        <input asp-for="Name" />
    </div>
    <!-- other form fields -->
</form>
```

In this example, the `asp-action` attribute on the `<form>` tag ensures that when the form is submitted, it will send data to the `RsvpForm` action method in the same controller.
x??

---

#### ASP.NET Core Tag Helpers for Model Binding
Background context explaining how tag helpers simplify model binding and form handling. Mention that `asp-for` attribute is used to bind form elements to a specific property of a view model, which helps in data submission and processing.

:p What is the purpose of the `asp-for` attribute in ASP.NET Core forms?
??x
The `asp-for` attribute simplifies the process of binding form elements directly to properties within your view models. When you use `asp-for`, it automatically sets the `id` and `name` attributes of input elements, making sure that the data is correctly submitted to the server when a form is posted.

For example:
```html
<p>
    <label for="Name">Your name:</label>
    <input type="text" asp-for="Name">
</p>

// This generates HTML like this:
<!-- <p>
    <label for="Name">Your name:</label>
    <input type="text" id="Name" name="Name">
</p> -->
```

x??

---
#### ASP.NET Core Routing and Action Methods
Background context explaining how routing is used in ASP.NET Core to map URLs to action methods. Mention that `asp-action` attribute helps configure the form submission URL based on application’s routing configuration.

:p How does the `asp-action` attribute help in routing within an ASP.NET Core application?
??x
The `asp-action` attribute simplifies configuring the form's `action` attribute by using the application’s URL routing configuration. This means that when you change the system of URLs used in your application, the content generated by tag helpers will automatically reflect these changes.

For example:
```html
<form method="post" asp-action="RsvpForm">
    <!-- Form elements here -->
</form>
```

This generates a form with an action attribute set to `/Home/RsvpForm`, which targets the `RsvpForm` action in the `HomeController`. If you change the routing configuration, these URLs will be updated automatically.

x??

---
#### Handling GET and POST Requests
Background context explaining that ASP.NET Core can handle different types of HTTP requests (GET and POST) with separate C# methods within a controller. Mention that this separation helps in maintaining clean and organized code by separating concerns.

:p How does ASP.NET Core differentiate between handling GET and POST requests using action methods?
??x
ASP.NET Core differentiates between handling GET and POST requests through the use of separate action methods within a controller. The `RsvpForm` action method can be defined to handle both GET and POST requests, but it needs to check which type of request was made.

Example:
```csharp
public class HomeController : Controller
{
    public IActionResult RsvpForm()
    {
        // Handle GET request - display the form
        return View();
    }

    [HttpPost]
    public IActionResult RsvpForm(GuestResponse guest)
    {
        // Handle POST request - process submitted data
        if (ModelState.IsValid)
        {
            // Save or process the data
        }
        return View(); // or some other action based on processing result
    }
}
```

Here, the `RsvpForm` method handles GET requests by returning a view to display the form. The `[HttpPost]` attribute ensures that only POST requests are processed by this method.

x??

---

#### Adding HttpGet and HttpPost Attributes to RsvpForm Method
Background context: The `HomeController` now has two methods named `RsvpForm`. One is for handling GET requests, and the other for POST requests. This setup allows the controller to serve different purposes during form submissions and data retrieval.

:p How does ASP.NET Core differentiate between the GET and POST requests in the `RsvpForm` method?
??x
In ASP.NET Core, you can specify which HTTP methods a method should handle by using attributes such as `[HttpGet]` and `[HttpPost]`. The `RsvpForm()` with no parameters is decorated with `[HttpGet]`, indicating it should process GET requests. The overloaded version of `RsvpForm(GuestResponse guestResponse)` is decorated with `[HttpPost]`, indicating it should process POST requests.

Here's the code for clarity:

```csharp
[HttpGet]
public ViewResult RsvpForm() {
    return View();
}

[HttpPost]
public ViewResult RsvpForm(GuestResponse guestResponse) {
    // TODO: store response from guest
    return View();
}
```
x??

---

#### Model Binding in ASP.NET Core
Background context: Model binding is a feature that parses incoming HTTP data and populates properties of domain model types. In the `RsvpForm` method, the `GuestResponse` object is automatically populated with form field values.

:p How does model binding work in the context of the `RsvpForm` action methods?
??x
Model binding works by automatically mapping form fields to properties of a C# class (in this case, `GuestResponse`). When a POST request is made to the `RsvpForm` method that accepts a `GuestResponse` parameter, ASP.NET Core parses the incoming HTTP data and populates the `guestResponse` object with values from the form.

Here's an example of how model binding works in code:

```csharp
[HttpPost]
public ViewResult RsvpForm([FromBody] GuestResponse guestResponse) {
    // The guestResponse object is automatically populated with form field values.
    return View();
}
```

The `[FromBody]` attribute ensures that the model binder looks at the request body to find the form data. This allows you to work directly with `guestResponse` rather than dealing with individual form fields.

x??

---

#### Repository Class for Storing Responses
Background context: A repository class is used to store and retrieve guest responses in memory, which simplifies development during early stages of application building. The repository uses a static list to keep track of the responses.

:p What is the purpose of the `Repository` class?
??x
The `Repository` class is designed to manage and provide access to guest responses. It keeps a list of all received responses in memory, which can be useful for demonstration purposes or when developing an application where persistent storage isn't necessary yet.

Here's how the `Repository` class is defined:

```csharp
namespace PartyInvites.Models {
    public static class Repository {
        private static List<GuestResponse> responses = new();
        public static IEnumerable<GuestResponse> Responses => responses;
        public static void AddResponse(GuestResponse response) {
            Console.WriteLine(response);
            responses.Add(response);
        }
    }
}
```

The `responses` list is a private static member, allowing multiple parts of the application to access and modify it. The `AddResponse` method adds new responses to this list, and `Responses` returns an enumerable collection of all stored responses.

x??

---

