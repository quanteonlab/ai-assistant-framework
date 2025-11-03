# Flashcards: Pro-ASPNET-Core-7_processed (Part 57)

**Starting Chapter:** 4.2 Adding code and content to projects

---

#### Opening a Project in Visual Studio Code and Visual Studio
Background context: The text describes how to open an existing project in both Visual Studio Code and Visual Studio, highlighting differences in file display methods between the two IDEs. It also mentions additional steps required when opening a project for the first time in Visual Studio Code.

:p How do you start working with a project in Visual Studio Code?
??x
To start working with a project in Visual Studio Code, follow these steps:
1. Open Visual Studio Code.
2. Select `File > Open Folder` from the menu.
3. Navigate to and select the folder containing your project (e.g., `MySolution`).
4. Click `Select Folder`.

If you open a project for the first time in Visual Studio Code, it might prompt you to add assets necessary for building and debugging. In such cases, click `Yes` to proceed.

```plaintext
// No specific code example needed here.
```
x??

---

#### Adding Code and Content to Projects (Visual Studio Code)
Background context: The text explains how to add files and folders to a project using Visual Studio Code, emphasizing the need for correct file extensions.

:p How do you add a new folder to your project in Visual Studio Code?
??x
To add a new folder to your project in Visual Studio Code:
1. Right-click on the folder that should contain the new folder.
2. Select `New File` from the context menu (or `New Folder` if applicable).
3. Set the name of the new item and press Enter.

For example, to create a folder named `wwwroot`, you would:
```plaintext
Right-click My Project > New Folder (set name: wwwroot) > Press Enter
```

x??

---

#### Adding Code and Content to Projects (Visual Studio)
Background context: The text provides an alternative method for adding files and folders using Visual Studio, which is more comprehensive but can be used selectively.

:p How do you add a new folder to your project in Visual Studio?
??x
To add a new folder to your project in Visual Studio:
1. Right-click the `MyProject` item in the Solution Explorer.
2. Select `Add > New Folder` from the context menu.
3. Set the name of the new item and press Enter.

For example, to create a folder named `wwwroot`, you would:
```plaintext
Right-click MyProject > Add > New Folder (set name: wwwroot) > Press Enter
```

x??

---

#### Adding an HTML File in Visual Studio Code
Background context: The text describes the process of adding a new file with content to your project using Visual Studio Code.

:p How do you add and populate an `demo.html` file in Visual Studio Code?
??x
To add and populate an `demo.html` file in Visual Studio Code:
1. Right-click the `wwwroot` folder.
2. Select `New File`.
3. Set the name to `demo.html` and press Enter.
4. Add the HTML content as shown in Listing 4.3.

Here's what the code looks like:
```html
DOCTYPE html>
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

x??

---

#### Adding an HTML File in Visual Studio
Background context: The text provides a method for adding and populating an `demo.html` file using the more comprehensive approach available in Visual Studio.

:p How do you add and populate an `demo.html` file in Visual Studio?
??x
To add and populate an `demo.html` file in Visual Studio:
1. Right-click the new `wwwroot` folder.
2. Select `Add > New Item`.
3. In the window that appears, search for or filter by "HTML Page".
4. Enter `demo.html` in the Name field.
5. Click the Add button to create the new file and replace its contents with the HTML content as shown in Listing 4.3.

Here's what the code looks like:
```html
DOCTYPE html>
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

x??

---

---
#### Item Templates and Scaffolding
Background context: Visual Studio provides item templates for creating files like HTML pages, C# classes with auto-generated namespaces, and scaffolded items that guide you through adding complex components. The choice between these is important as they can significantly affect your project structure.

:p What are the differences between item templates and scaffolded items in Visual Studio?
??x
Item templates provide a basic file structure for common types of files like HTML pages or C# classes with auto-generated namespaces. Scaffolded items, on the other hand, guide you through creating more complex components and automatically add relevant code based on your selections.

For example, selecting "HTML Page" from item templates creates an .html file that includes a basic structure suitable for web content. In contrast, scaffolded items like "View" or "Controller" lead to more complex files with pre-defined logic and structures tailored to specific ASP.NET Core functionalities such as routing or data binding.

Scaffolded items are recommended against using until you fully understand their contents due to potential complexity that might introduce unnecessary dependencies.
x??

---
#### Building and Running Projects Using Command Line
Background context: You can build and run projects from the command line for simplicity. This involves adding necessary statements in your code, setting up the application URL, and running commands like `dotnet build` and `dotnet run`.

:p What are the steps to build and run a project using the dotnet CLI?
??x
To build and run a project using the .NET Core Command Line Interface (CLI), you need to follow these steps:

1. **Add Required Statements**: Modify your `Program.cs` file with statements that configure your application, such as setting up routes and static files.
2. **Set HTTP Port in `launchSettings.json`**: Configure the application URL in the `Properties\launchSettings.json` file to specify the port where your project will run.

Here is an example of modifying the `Program.cs` file:

```csharp
var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();
app.MapGet("/", () => "Hello World.");
app.UseStaticFiles();
app.Run();
```

This code sets up a simple web server that responds to GET requests at the root URL with "Hello World." and serves static files from the `wwwroot` folder.

Next, set the HTTP port in the `Properties\launchSettings.json` file:

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

Finally, build the project using the following command in your terminal:

```bash
dotnet build MyProject.csproj
```

And run it with:

```bash
dotnet run
```
x??

---

#### Building and Running a Project with `dotnet run`
Background context: This section explains how to build and run an ASP.NET Core project using the `dotnet` CLI command. The process involves compiling the project and starting an integrated HTTP server that listens on port 5000.

:p How do you start building and running your ASP.NET Core project?
??x
To start building and running your ASP.NET Core project, use the following command in the terminal from within the `MyProject` folder:

```bash
dotnet run
```

This command will compile the project and launch an integrated ASP.NET Core HTTP server. You can then access the application by navigating to `http://localhost:5000/demo.html`.

x??

---

#### Hot Reload Feature in .NET
Background context: The hot reload feature allows real-time updates to applications without requiring a full restart, enhancing development productivity.

:p What is the hot reload feature and how does it work with ASP.NET Core?
??x
The hot reload feature in .NET enables automatic recompilation of code on changes and automatically refreshes the browser window. For ASP.NET Core applications, this means that updates to your project can be seen instantly without stopping the application or manually running `dotnet run`.

To start a project with hot reloading, use:

```bash
dotnet watch
```

This command opens a new browser tab, which ensures the browser loads a small piece of JavaScript enabling real-time updates. Changes made to the code are detected and recompiled on the fly.

x??

---

#### Hot Reload for HTML Files
Background context: This section demonstrates how changes in static files like `demo.html` can be automatically reflected in the running application without restarting it.

:p How do you modify a static file (`demo.html`) and see the updates instantly?
??x
To demonstrate hot reload for an HTML file, follow these steps:

1. Modify the content of `wwwroot/demo.html` to include:
   ```html
   <h3>New Message</h3>
   ```

2. Save the changes.

When you save the `demo.html` file, the `dotnet watch` tool will detect the change and automatically update the browser window.

x??

---

#### Handling Namespace Changes with Hot Reload
Background context: This section explains why some changes, such as modifying namespaces, cannot be handled by hot reload and require a full restart of the application.

:p What happens if you make changes to a namespace in an ASP.NET Core project?
??x
If you attempt to modify a namespace in a class file while using `dotnet watch`, it may result in a message like:

```
watch : File changed: C:\MySolution\MyProject\MyClass.cs.
watch : Unable to apply hot reload because of a rude edit. The application is restarted instead.
```

This occurs because the changes cannot be safely applied without a full restart due to potential issues with dependencies and state management.

x??

---

