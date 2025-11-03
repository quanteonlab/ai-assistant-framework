# Flashcards: Pro-ASPNET-Core-7_processed (Part 9)

**Starting Chapter:** 4.2 Adding code and content to projects

---

---
#### Opening a Project in Visual Studio Code and Visual Studio
Background context: The text describes how to open an existing project in both Visual Studio and Visual Studio Code, highlighting the differences in their interfaces.

:p How do you open a project in Visual Studio Code?
??x
To open a project in Visual Studio Code:
1. Select File > Open Folder.
2. Navigate to the MySolution folder.
3. Click Select Folder to open the project.

This method allows you to load an existing project into Visual Studio Code, which displays files alphabetically by default.
x??

---
#### Adding Code and Content to Projects (VSCode)
Background context: The text explains how to add new items, such as folders and files, to a project using Visual Studio Code. It emphasizes the importance of correct file extensions.

:p How do you add a folder named `wwwroot` in Visual Studio Code?
??x
To add a folder named `wwwroot` in Visual Studio Code:
1. Right-click on the My Project item.
2. Select New Folder from the context menu.
3. Enter `wwwroot` as the name and press Enter.

This creates a new folder within your project, which can be used to store static content.
x??

---
#### Adding Code and Content to Projects (Visual Studio)
Background context: The text provides an alternative method for adding items to a project in Visual Studio, using its Solution Explorer with additional templates.

:p How do you add a file named `demo.html` in Visual Studio?
??x
To add a file named `demo.html` in Visual Studio:
1. Right-click the MyProject item.
2. Select Add > New Item from the context menu.
3. In the window that appears, search for and select "HTML Page" as the template.
4. Enter `demo.html` as the name and click Add.

This creates a new HTML file within your project with the correct extension based on the selected template.
x??

---
#### Creating Folders in Visual Studio
Background context: The text describes how to create folders using the Solution Explorer in Visual Studio, highlighting the use of templates for added functionality.

:p How do you add a folder named `wwwroot` via Visual Studio's Solution Explorer?
??x
To add a folder named `wwwroot` via Visual Studio's Solution Explorer:
1. Right-click on the MyProject item.
2. Select Add > New Folder from the context menu.
3. Enter `wwwroot` as the name and press Enter.

Visual Studio will create the new folder within your project, which can be used to store static content such as HTML files.
x??

---
#### Adding an HTML File via Visual Studio
Background context: The text provides a specific example of adding an HTML file named `demo.html`, including its contents.

:p How do you add and edit the `demo.html` file in Visual Studio?
??x
To add and edit the `demo.html` file in Visual Studio:
1. Right-click on the new wwwroot folder.
2. Select Add > New Item from the context menu.
3. In the window that appears, search for and select "HTML Page" as the template.
4. Enter `demo.html` as the name and click Add.
5. Replace the default content with:
   ```html
   <!DOCTYPE html>
   <html>
   <head>
       <meta charset="utf-8" />
       <title></title>
   </head>
   <body>
       <h3>HTML File from MyProject</h3>
   </body>
   </html>
   ```

This process adds and customizes an HTML file within your project.
x??

---

---
#### Understanding Item Scaffolding in Visual Studio
Item templates provided by Visual Studio, such as those for C# classes, automatically set up namespaces and class names. However, scaffolded items are more complex and can guide you through creating advanced features like views and controllers.

:p What are the potential downsides of using scaffolded items in Visual Studio?
??x
Using scaffolded items in Visual Studio can lead to unnecessary complexity and may introduce unwanted code that doesn't align with your project's needs. It’s recommended to use only necessary templates and customize them as needed.
x??

---
#### Building and Running Projects Using Command-Line Tools
To build and run a .NET Core project using command-line tools, you first need to configure the environment properly by adding specific statements in `Program.cs` and setting up `launchSettings.json`.

:p What are the steps involved in building and running a .NET Core project via the command line?
??x
1. Add necessary configuration in `Program.cs`:
   ```csharp
   var builder = WebApplication.CreateBuilder(args);
   var app = builder.Build();
   
   app.MapGet("/", () => "Hello World.");
   
   app.UseStaticFiles();
   
   app.Run();
   ```
2. Configure the HTTP port and other settings in `launchSettings.json`:
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
           "MyProject": {
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
3. Build the project using `dotnet build` in the command line.

x??

---
#### Adding a Statement to Program.cs
In .NET Core projects, adding statements like those for handling HTTP requests and serving static files is crucial for setting up basic routing and functionality.

:p What does the statement `app.MapGet("/", () => "Hello World.");` do?
??x
The statement `app.MapGet("/", () => "Hello World.");` maps a GET request to the root URL ("/") of your application, responding with the string "Hello World." when this route is accessed. This sets up a basic HTTP endpoint for testing purposes.
x??

---
#### Configuring Static Files and Routing in .NET Core
.NET Core applications need configuration to handle static files such as HTML or CSS served from specific folders.

:p What does `app.UseStaticFiles();` do?
??x
The statement `app.UseStaticFiles();` enables serving of static files (such as those stored in the `wwwroot` folder) directly from a web server. This is essential for hosting and serving client-side resources.
x??

---
#### Setting Up Development Environment Variables
Environment variables set up in `launchSettings.json` influence how your application behaves during development.

:p How do you configure the environment for a .NET Core project using `launchSettings.json`?
??x
You can configure the environment by setting the `ASPNETCORE_ENVIRONMENT` variable. For example:
```json
"environmentVariables": {
    "ASPNETCORE_ENVIRONMENT": "Development"
}
```
This sets the application's environment to Development, which is useful for running in a local development setup.
x??

---

---

#### Using Command Line to Run a Project
Background context: Running projects directly from the command line is an efficient way to develop and test applications. The `dotnet run` command compiles and runs the project seamlessly.

:p How do you build and run a .NET project using the command line?
??x
To build and run a .NET project, you use the following command in the terminal within your project directory:
```bash
dotnet run
```
This command first builds the project (if necessary) and then runs it. If no specific entry point is specified, `dotnet run` will default to running `Program.Main()`.

x??

---

#### Building a Static HTML File into an ASP.NET Core Application
Background context: ASP.NET Core projects can serve static files like HTML. By adding these files under the `wwwroot` folder, they can be served directly by the server.

:p How do you access a static HTML file added to your project?
??x
To view a static HTML file in an ASP.NET Core application, you need to place it under the `wwwroot` folder. You can then access it via its URL, which is typically `http://localhost:5000/{filename}`.

For example, if you have added a file named `demo.html`, you would request:
```
http://localhost:5000/demo.html
```

x??

---

#### Hot Reload Feature in ASP.NET Core
Background context: The .NET hot reload feature allows changes to the codebase to be reflected immediately without needing to restart the application. This is particularly useful for developing web applications where frequent small changes are made.

:p What is the command to start an ASP.NET Core project with hot reload?
??x
To start an ASP.NET Core project with hot reload, you use the following command:
```bash
dotnet watch
```
This command monitors the project files for changes and automatically recompiles them when changes are detected. It also ensures that any open browsers will be refreshed to show updated content.

x??

---

#### Example of Hot Reload in Action
Background context: The hot reload feature can significantly speed up development by allowing developers to see updates in real-time without manually stopping and restarting the application.

:p How do you test the hot reload feature with a simple change?
??x
To test the hot reload feature, make a small change to an HTML file (e.g., `demo.html`), save it, and then refresh your browser. For example:

1. Modify `wwwroot/demo.html`:
```html
<DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title></title>
</head>
<body>
    <h3>New Message</h3>
</body>
```

2. Save the file.
3. Refresh your browser to see the updated content.

x??

---

#### Hot Reload Limitations
Background context: While hot reload is a powerful feature, not all changes can be handled automatically. Some complex edits require an application restart, which can disrupt state in certain applications.

:p What happens when you make a namespace change that cannot be handled by hot reload?
??x
When you make a namespace change (e.g., moving `MyClass` into a new namespace), the dotnet watch command might not be able to handle this change automatically and may prompt for an application restart. This is because such changes are considered "rude edits" that cannot be seamlessly applied.

For example, if you change:
```csharp
namespace MyProject {
    public class MyClass {}
}
```
to:
```csharp
namespace MyProject.MyNamespace {
    public class MyClass {}
}
```

The dotnet watch tool might produce output similar to this when the file is saved:
```
watch : File changed: C:\MySolution\MyProject\MyClass.cs.
watch : Unable to apply hot reload because of a rude edit. Application will be restarted.
```

x??

---

