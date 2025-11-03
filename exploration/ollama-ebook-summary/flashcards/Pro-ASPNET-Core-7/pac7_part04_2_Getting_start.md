# Flashcards: Pro-ASPNET-Core-7_processed (Part 4)

**Starting Chapter:** 2 Getting started. 2.1.1 Installing Visual Studio

---

#### Choosing a Code Editor for ASP.NET Core Development
Microsoft provides tools like Visual Studio and Visual Studio Code for ASP.NET Core development. Both editors are capable, but they have different features and resource requirements.

:p Which code editor is recommended for new .NET developers?
??x
Visual Studio is recommended because it offers more structured support for creating the necessary files in ASP.NET Core development, ensuring better compatibility with the examples provided.
x??

---

#### Installing Visual Studio 2022 Community Edition
To develop ASP.NET Core applications using Visual Studio 2022, you need to install the appropriate version of Visual Studio. The free Community Edition can be downloaded from Microsoft's official website.

:p How do you start the installation process for Visual Studio 2022 Community Edition?
??x
1. Go to the Visual Studio website: <https://visualstudio.microsoft.com/>.
2. Click on "Download Visual Studio" and select "Community" edition.
3. Run the installer, which will prompt you with a setup screen (Figure 2.1).
4. Follow the instructions by clicking the Continue button.

After downloading, proceed to install the necessary workloads and components as described in subsequent steps.
x??

---

#### Selecting Workload for ASP.NET Core Development
During the Visual Studio installation process, ensure that the "ASP.NET and web development" workload is selected. This ensures you have the right tools to develop ASP.NET Core applications.

:p What should be done during the Visual Studio installer setup to support ASP.NET Core development?
??x
Ensure that the "ASP.NET and web development" workload is checked (Figure 2.3). Additionally, check the SQL Server Express 2019 LocalDB option under the "Individual components" section (Figure 2.4) because it will be used in later chapters for storing data.
x??

---

#### Ensuring LocalDB Installation
LocalDB is a Windows-only feature provided by SQL Server that needs to be installed during Visual Studio setup. It ensures compatibility with the examples and exercises in this book.

:p How do you ensure LocalDB is installed during the Visual Studio 2022 installation?
??x
Ensure that the "ASP.NET and web development" workload is selected, which includes the necessary components for ASP.NET Core development. Specifically, check the SQL Server Express 2019 LocalDB option under the "Individual components" section (Figure 2.4).
x??

---

#### Installing .NET SDK
After installing Visual Studio, you need to install the .NET Software Development Kit (SDK) if it wasn't installed automatically by Visual Studio. You can download the required version from Microsoft's official website.

:p How do you ensure that the correct .NET SDK is installed for this book?
??x
Go to <https://dotnet.microsoft.com/download/dotnet-core/7.0> and download the installer for .NET SDK version 7.0.0, which is the current release at the time of writing.
x??

---

#### Installing .NET SDK
Background context: The .NET SDK is essential for developing applications using the .NET framework. It includes tools and libraries necessary to build, run, and debug .NET applications.

:p How do you check which versions of the .NET SDK are installed on your machine?
??x
To check which versions of the .NET SDK are installed, you can use the command `dotnet --list-sdks`. This command lists all the installed SDKs along with their paths.
```powershell
dotnet --list-sdks
```
The output will display a list like:
```
7.0.100 [C:\Program Files\dotnet\sdk]
5.0.100 [C:\Program Files\dotnet\sdk] 
6.0.100 [C:\Program Files\dotnet\sdk]
...
```
Ensure that there is at least one entry for the 7.0.x version.

x??

---

#### Installing Visual Studio Code
Background context: Visual Studio Code (VSCode) is a popular code editor that can be used to write .NET applications. It provides features like syntax highlighting, debugging, and extensions to enhance development productivity.

:p How do you install Visual Studio Code?
??x
To install Visual Studio Code, visit the official website at <https://code.visualstudio.com> and download the installer for your operating system. During installation, make sure to check the "Add to PATH" option to ensure that VSCode is accessible from the command line.
```powershell
// Example of adding VSCode to PATH
echo 'export PATH="$PATH:/path/to/visual-studio-code/bin"' >> ~/.bashrc
```
x??

---

#### Installing .NET SDK via Visual Studio Code Installer
Background context: While the Visual Studio Code installer provides a convenient way to install other tools, it does not include the .NET SDK. Therefore, you need to download and install the .NET SDK separately.

:p How do you ensure that the 7.x version of the .NET SDK is installed via the official website?
??x
To install the 7.x version of the .NET SDK from the official Microsoft site, visit <https://dotnet.microsoft.com/download/dotnet-core/7.0> and download the appropriate installer for your operating system. Run the installer and ensure that it includes the necessary components.

Once installed, verify the installation by opening a new PowerShell command prompt and running:
```powershell
dotnet --list-sdks
```
The output should confirm that 7.x is among the installed versions.
x??

---

#### Installing SQL Server LocalDB
Background context: SQL Server LocalDB is a lightweight version of SQL Server that can be used for development purposes without requiring a full SQL Server installation.

:p How do you install SQL Server Express with LocalDB?
??x
To install SQL Server Express with LocalDB, download the installer from <https://www.microsoft.com/en-in/sql-server/sql-server-downloads>. During the installation, select "Custom" as your setup type. This allows you to choose which components are installed.

After selecting Custom, follow these steps:
1. Choose a download location for the SQL Server Express installation files.
2. Click Install to start the download and installation process.
3. When prompted, select the option to create a new SQL Server installation.
4. On the Feature Selection page, ensure that LocalDB is checked.

Follow the default options for the rest of the installation steps.
x??

---

#### Creating an ASP.NET Core Project Using Command Line
Background context: This section explains how to create a basic ASP.NET Core project using the command line interface (CLI) provided by the .NET SDK. It covers essential commands for setting up the environment and creating a new MVC project.

:p How do you create a new ASP.NET Core project using the dotnet CLI?
??x
To create a new ASP.NET Core project, follow these steps:
1. Open a PowerShell command prompt.
2. Navigate to the desired folder where you want to create your project.
3. Use the following commands to set up and create the project:

```powershell
# Create global.json file specifying .NET SDK version
dotnet new globaljson --sdk-version 7.0.100 --output FirstProject

# Create a new ASP.NET Core MVC project using net7.0 framework
dotnet new mvc --no-https --output FirstProject --framework net7.0

# Create a solution file named FirstProject.sln
dotnet new sln -o FirstProject

# Add the newly created project to the solution
dotnet sln FirstProject add FirstProject
```

These commands ensure that your project uses the specified .NET SDK version, creates an MVC template-based ASP.NET Core project, and sets up a solution file for managing multiple projects.

x??

---
#### Opening the ASP.NET Core Project in Visual Studio
Background context: This section details how to open an existing ASP.NET Core project created earlier using either Visual Studio or Visual Studio Code. It explains navigating to the project directory and opening it with respective IDEs.

:p How do you open a newly created ASP.NET Core project in Visual Studio?
??x
To open the project in Visual Studio, follow these steps:

1. Start Visual Studio.
2. Click on the "Open a project or solution" button.
3. Navigate to the `FirstProject` folder where your project is located.
4. Select the `FirstProject.sln` file (solution file) and click the Open button.

Visual Studio will open the project, and its contents will be displayed in the Solution Explorer window.

x??

---
#### Opening the ASP.NET Core Project in Visual Studio Code
Background context: This section explains how to open an existing ASP.NET Core project using Visual Studio Code. It details navigating to the folder containing the project files and opening them within VSCode's environment.

:p How do you open a newly created ASP.NET Core project in Visual Studio Code?
??x
To open the project in Visual Studio Code, follow these steps:

1. Start Visual Studio Code.
2. Select File > Open Folder from the main menu.
3. Navigate to the `FirstProject` folder containing your project files.
4. Click the Select Folder button.

Visual Studio Code will open the project and display its contents in the Explorer pane. For better visibility, you might want to switch to the light theme if it's not already set.

x??

---
#### Using Different IDEs for ASP.NET Core Projects
Background context: This section highlights the flexibility of working with ASP.NET Core projects across different Integrated Development Environments (IDEs) like Visual Studio and Visual Studio Code. It emphasizes that both environments support ASP.NET Core development, making it easier to choose based on personal preferences or specific project needs.

:p What are some benefits of using Visual Studio versus Visual Studio Code for developing an ASP.NET Core project?
??x
Both Visual Studio and Visual Studio Code offer their own unique advantages:

- **Visual Studio**: 
  - Provides a more robust development environment with extensive debugging tools, code analysis features, and integration with other Microsoft products.
  - Better support for complex projects and large-scale solutions.

- **Visual Studio Code**:
  - Lightweight and fast, making it an excellent choice for smaller projects or when developing on non-Windows operating systems.
  - Offers rich extension support through the marketplace, allowing customization according to specific needs.

:x??
---

---
#### Installing Visual Studio Code C# Features
Visual Studio Code will prompt you to install necessary features when opening a `.cs` file or creating a new project. This step is crucial for running and debugging C# projects within Visual Studio Code.

:p How does Visual Studio Code notify users about installing the required assets for C# development?
??x
When you open a `.cs` file in Visual Studio Code, it will prompt to install features related to C#. If this is your first time opening a C# project, there will be an additional prompt asking whether you want to install the necessary assets. This step ensures that all required tools and extensions are available for developing .NET projects.

For example:
```
Do you want to install .NET Core SDK and other required features?
```

x??
---
#### Configuring the `launchSettings.json` File
The `Properties/launchSettings.json` file in an ASP.NET Core project determines which HTTP port the application will use. This file can be edited to change the default settings.

:p How does one modify the default HTTP port for running an ASP.NET Core application using Visual Studio Code?
??x
To change the default HTTP port, open the `Properties/launchSettings.json` file and update the URLs in the `profiles` section. Specifically, you should set the `applicationUrl` to use a different port number.

For example:
```json
{
  "iisSettings": {
    "windowsAuthentication": false,
    "anonymousAuthentication": true,
    "iisExpress": {
      "applicationUrl": "http://localhost:5001", // Changed from :5000 to :5001
      "sslPort": 0
    }
  },
  "profiles": {
    "FirstProject": {
      "commandName": "Project",
      "dotnetRunMessages": true,
      "launchBrowser": true,
      "applicationUrl": "http://localhost:5001", // Changed from :5000 to :5001
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
#### Starting the ASP.NET Core Application
Using command-line tools to run an ASP.NET Core application provides more reliability and consistency compared to integrated development environments (IDEs).

:p How can one start the example ASP.NET Core application using `dotnet run` in a PowerShell window?
??x
To start the ASP.NET Core application, open a new PowerShell command prompt from the Windows Start menu. Navigate to the project folder containing the `.csproj` file and execute the following command:
```sh
dotnet run
```

This command compiles the code and starts the application.

For example:
```powershell
PS C:\path\to\FirstProject> dotnet run
```

x??
---
#### Understanding Endpoints in ASP.NET Core
In an ASP.NET Core application, incoming HTTP requests are handled by endpoints. These endpoints consist of actions, which are methods written in C#. Actions are defined within controllers, derived from `Microsoft.AspNetCore.Mvc.Controller`.

:p What is the relationship between controllers and actions in an ASP.NET Core application?
??x
Controllers in ASP.NET Core are classes that inherit from `Microsoft.AspNetCore.Mvc.Controller`. Public methods defined in these controller classes act as actions. These actions handle incoming HTTP requests.

For example:
```csharp
using Microsoft.AspNetCore.Mvc;

public class FirstController : Controller {
    public IActionResult Index() {
        return Content("Hello, World!");
    }
}
```

In this code snippet, `Index` is an action that returns a simple text response when accessed via the browser. The `Controller` base class provides methods to handle different HTTP verbs (GET, POST, etc.).

x??
---

#### ASP.NET Core Project Structure and Controllers
Background context: In ASP.NET Core projects, controllers play a crucial role in handling HTTP requests. The project template typically includes a `Controllers` folder with a default `HomeController`. This controller handles common routing such as `/Home/Index`.
:p What is the purpose of the `Controllers` folder in an ASP.NET Core project?
??x
The `Controllers` folder houses the C# classes that handle HTTP requests and return responses, making it easier to manage different routes and actions within the application.
x??

---

#### HomeController Class Overview
Background context: The `HomeController` class is a typical example of how controllers are structured in ASP.NET Core. It inherits from the `Controller` base class provided by Microsoft's framework and contains methods (actions) that correspond to different HTTP request endpoints.
:p What does the `HomeController` class do?
??x
The `HomeController` class handles HTTP requests through its action methods, which return views or content based on the incoming URL path. For example, an action method named `Index()` might be triggered when a user visits `/Home/Index`.
x??

---

#### Index Action Method in HomeController
Background context: The `Index` action method is a common starting point for ASP.NET Core applications. It returns a simple string or view based on the request.
:p What does the `Index` action method do?
??x
The `Index` action method processes an HTTP GET request to `/Home/Index` and returns a response (in this case, a string "Hello World") to be displayed in the browser.
x??

---

#### Using Statements for Namespaces
Background context: In C#, using statements are used at the top of a file to import namespaces. These imports allow you to use classes and methods from different libraries without needing to fully qualify their names.
:p What is the purpose of using statements in `HomeController.cs`?
??x
Using statements in `HomeController.cs` provide access to necessary namespaces that contain classes like `ILogger<HomeController>` and `ActionResult`. This allows for more readable and concise code within the controller class.
x??

---

#### Replacing HomeController Code
Background context: The original `HomeController` contains multiple action methods. To simplify, the code is replaced to include only one method that returns a string "Hello World" when called through an HTTP request.
:p How does replacing the content of `HomeController.cs` affect the application's behavior?
??x
Replacing the content simplifies the controller by removing unnecessary complexity and focuses on returning a static response ("Hello World") for the `/Home/Index` route. This makes it easier to understand how requests are handled in ASP.NET Core.
x??

---

#### Running the Application
Background context: After modifying the `HomeController.cs`, running the application with the command `dotnet run` deploys the changes and starts the web server on `http://localhost:5000`. Browsing to this URL triggers the `Index` action method in the `HomeController`.
:p What happens when you navigate to `http://localhost:5000` after running the application?
??x
Navigating to `http://localhost:5000` sends an HTTP request to the server. The `HomeController` processes this request with its `Index` action method, which returns "Hello World" as a response. This string is then displayed in the browser.
x??

---

#### Action Method Returning String
Background context: In ASP.NET Core, action methods can return different types of results, including simple strings or more complex views. Here, an action method simply returns a string to be shown directly in the browser.
:p How does returning a string from an action method work?
??x
Returning a string from an action method means that when this method is called (e.g., via an HTTP request), the response sent back to the client is exactly the string returned. In this case, "Hello World" is displayed directly in the browser.
x??

---

