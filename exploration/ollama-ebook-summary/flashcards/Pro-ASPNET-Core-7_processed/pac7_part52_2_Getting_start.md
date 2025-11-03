# Flashcards: Pro-ASPNET-Core-7_processed (Part 52)

**Starting Chapter:** 2 Getting started. 2.1.1 Installing Visual Studio

---

#### Choosing a Code Editor for ASP.NET Core Development
Background context: The book provides options for using either Visual Studio or Visual Studio Code to develop ASP.NET Core applications. Both editors are capable, but they have different features and resource requirements.

:p What code editor is recommended for new .NET developers?
??x
Visual Studio is the preferred choice for new .NET developers due to its structured support for creating various files used in ASP.NET Core development, which helps ensure that the examples provided will work as expected. 
??x
The answer explains that Visual Studio provides a more structured environment, making it easier for beginners to follow along with the book's code examples.
```csharp
// Example of setting up a basic ASP.NET Core project in C#
public class Startup {
    public void ConfigureServices(IServiceCollection services) {
        // Add framework services.
        services.AddControllersWithViews();
    }
}
```
x??

---

#### Installing Visual Studio 2022 for ASP.NET Core Development
Background context: To develop ASP.NET Core applications using Visual Studio, you need to install the appropriate version of Visual Studio. The book recommends using Visual Studio 2022 Community Edition.

:p How do you start the installation process for Visual Studio 2022?
??x
You can start the installation process by visiting the official Microsoft website and downloading the installer from `www.visualstudio.com`. After downloading, run the installer to proceed.
??x
The answer explains how to access the download page and initiate the installation process. No specific code is needed here as it's a step-by-step guide.
---
#### Selecting Workloads in Visual Studio Installer
Background context: During the installation of Visual Studio 2022, you need to select the appropriate workloads to install. The book specifically mentions the "ASP.NET and web development" workload.

:p Which workload should be selected during the Visual Studio installer?
??x
The "ASP.NET and web development" workload should be selected during the Visual Studio installer to ensure that all necessary components for ASP.NET Core development are installed.
??x
The answer explains why this particular workload is essential for developing ASP.NET Core applications. No code is needed as it's about selecting a specific option in the installer.
---
#### Installing SQL Server Express 2019 LocalDB
Background context: The book mentions that you need to ensure that the SQL Server Express 2019 LocalDB component is installed, as it will be used for storing data in later chapters.

:p How do you ensure that SQL Server Express 2019 LocalDB is included during installation?
??x
During the Visual Studio installer, navigate to the "Individual components" section and make sure the "SQL Server Express 2019 LocalDB" option is checked. This ensures that the database component necessary for storing data in later chapters is installed.
??x
The answer explains how to check the specific component during the installation process. No code is needed as it's about navigating the installer interface.
---
#### Installing .NET SDK
Background context: The book notes that while Visual Studio installs the .NET SDK, you might need to install a specific version for the examples in the book.

:p How do you ensure the correct version of the .NET SDK is installed?
??x
Visit the official Microsoft website and download the installer for .NET SDK 7.0.0 from `https://dotnet.microsoft.com/download/dotnet-core/7.0`. This ensures that you have the correct version to follow along with the book's examples.
??x
The answer provides a step-by-step guide on how to download and install the required version of the .NET SDK, ensuring compatibility with the book's examples. No code is needed as it involves navigating a website and downloading software.
---

#### Running .NET Installer and Listing SDKs
Background context: This section explains how to install and verify the installation of the .NET SDK on a Windows machine. The `dotnet --list-sdks` command is used to check which versions of the .NET SDK are installed.

:p How do you list the installed .NET SDKs on a Windows machine?
??x
To list the installed .NET SDKs, you can use the following command in PowerShell:
```powershell
dotnet --list-sdks
```
This will output a list of all installed SDK versions and their paths. Ensure that version 7.0.x is present in the list.
x??

---
#### Installing Visual Studio Code with PATH Option
Background context: This section describes how to install Visual Studio Code, ensuring it adds the .NET SDK path to the system's PATH environment variable.

:p How do you ensure Visual Studio Code includes the .NET SDK path in the PATH environment variable?
??x
When installing Visual Studio Code from https://code.visualstudio.com, make sure to check the "Add to PATH" option during installation as shown in Figure 2.5. This step is crucial for ensuring that tools like `dotnet` can be accessed from any command prompt.

To verify, open a new PowerShell window and run:
```powershell
where.exe dotnet
```
This should return the path where `dotnet` is installed.
x??

---
#### Installing .NET SDK Separately
Background context: This section explains that while Visual Studio Code does not include the .NET SDK by default, it must be downloaded separately from https://dotnet.microsoft.com/download/dotnet-core/7.0.

:p How do you install the .NET SDK version 7.0 on a Windows machine?
??x
To install the .NET SDK version 7.0, follow these steps:
1. Go to https://dotnet.microsoft.com/download/dotnet-core/7.0.
2. Download and run the installer.
3. Once installed, verify the installation by running the command:
```powershell
dotnet --list-sdks
```
Ensure that version 7.0.x is listed.

Example output:
```powershell
7.0.100 [C:\Program Files\dotnet\sdk]
```
x??

---
#### Installing SQL Server LocalDB
Background context: This section explains how to install the SQL Server LocalDB, a lightweight version of SQL Server that can be used for database examples in this book.

:p How do you install SQL Server Express with LocalDB?
??x
To install SQL Server Express with LocalDB:
1. Go to https://www.microsoft.com/en-in/sql-server/sql-server-downloads.
2. Download and run the Express edition installer, selecting "Custom" as shown in Figure 2.6.
3. In the Feature Selection page, ensure that the LocalDB option is checked (see Figure 2.8).
4. Follow through with the installation process using default options.

To verify the installation, you can check for the presence of SQL Server services or run:
```sql
SELECT @@SERVERNAME;
```
from a SQL Server Management Studio or PowerShell window.
x??

---

#### Creating an ASP.NET Core Project Using Command Line
Background context: To create a new project, you can use the command line. This method is useful for setting up projects without needing to open any IDEs or editors.

:p How do you create a new ASP.NET Core project using the dotnet CLI?
??x
To create a new ASP.NET Core project, follow these steps:

1. Open PowerShell from the Windows Start menu.
2. Navigate to the folder where you want to create your project.
3. Run the following commands in sequence:
   - `dotnet new globaljson --sdk-version 7.0.100 --output FirstProject` 
   - `dotnet new mvc --no-https --output FirstProject --framework net7.0`
   - `dotnet new sln -o FirstProject`
   - `dotnet sln FirstProject add FirstProject`

The first command creates a folder named "FirstProject" and adds a global.json file, specifying the version of .NET to be used. The second command sets up an ASP.NET Core MVC project using the provided framework version. The remaining commands create a solution file that can manage multiple projects.

??x
---
#### Opening the Project in Visual Studio
Background context: After creating your project, you may want to open it in Visual Studio for development and debugging purposes.

:p How do you open an ASP.NET Core project created with the dotnet CLI in Visual Studio?
??x
To open a project that was created using the command line in Visual Studio:

1. Start Visual Studio.
2. Click on "Open a project or solution."
3. Navigate to the folder containing your FirstProject.sln file.
4. Select the FirstProject.sln file and click "Open."

Visual Studio will load the project, and you can see its contents in the Solution Explorer window.

??x
---
#### Opening the Project in Visual Studio Code
Background context: If you prefer using a code editor like Visual Studio Code, you can also open your ASP.NET Core project there for development.

:p How do you open an ASP.NET Core project created with the dotnet CLI in Visual Studio Code?
??x
To open a project that was created using the command line in Visual Studio Code:

1. Start Visual Studio Code.
2. Select File > Open Folder.
3. Navigate to the FirstProject folder and click "Select Folder."

Visual Studio Code will load the project, displaying its contents in the Explorer pane.

Additional configuration might be needed the first time you open a .NET project in VSCode.

??x
---
#### Using MVC Template for ASP.NET Core Projects
Background context: The MVC template is one of the options available when creating an ASP.NET Core application. It sets up the project to use the Model-View-Controller architecture, which separates concerns such as data handling and user interface logic.

:p Why might someone choose the MVC template when creating a new ASP.NET Core project?
??x
The MVC (Model-View-Controller) template is chosen because it provides a structured way to build web applications. This template creates a project that is configured for the MVC Framework, which allows developers to separate concerns like data handling and user interface logic.

While this concept can seem intimidating at first, by the end of the book, you will understand how each feature fits together and what benefits they offer. The MVC framework helps in managing complexity by dividing responsibilities among different components.

??x
---

---
#### Installing Visual Studio Code C# Features
Background context: When you open a `.cs` file or create a new project, Visual Studio Code prompts to install necessary features for C# development. This step is crucial for ensuring that all required assets and tools are available.

:p How do you start the installation process for C# in Visual Studio Code?
??x
To start the installation of C# features, click on the `.cs` file in the Explorer pane. If prompted, select "Install" or "Yes" to proceed with downloading and installing the necessary assets.
x??

---
#### `launchSettings.json` File Configuration
Background context: The `launchSettings.json` file is used to configure how your application starts up, including the HTTP port it listens on for incoming requests.

:p What changes need to be made in the `launchSettings.json` file?
??x
You should change the URLs in the `profiles` section of the `launchSettings.json` file to use port 5000. Here is how you can do it:

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
     "FirstProject": {
       "commandName": "Project",
       "dotnetRunMessages": true,
       "launchBrowser": true,
       "applicationUrl": "http://localhost:5000", // Changed to port 5000
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
#### Running ASP.NET Core Application via Command Line
Background context: Using the command line tools to run projects is preferred over IDEs for consistency and reliability. This ensures that you get predictable results from your development environment.

:p How do you start an ASP.NET Core application using the command line?
??x
To start the ASP.NET Core application, open a new PowerShell window, navigate to the project directory containing `FirstProject.csproj`, and run:

```sh
dotnet run
```
This command compiles and starts the application.

:p How do you stop the running application when it is finished?
??x
Once the application has started, use `Ctrl+C` in the PowerShell window where the application is running to stop it.
x??

---
#### Understanding Endpoints in ASP.NET Core Applications
Background context: In an ASP.NET Core application, incoming HTTP requests are handled by endpoints. These endpoints can be actions (methods written in C#) defined within controllers.

:p What are actions and how do they work?
??x
Actions are methods that handle incoming HTTP requests in an ASP.NET Core application. They are typically found inside controller classes derived from `Microsoft.AspNetCore.Mvc.Controller`. Each public method in a controller class is considered an action, which means you can invoke these methods to handle specific HTTP request types such as GET or POST.

:p What is the role of controllers in this process?
??x
Controllers act as the central point where actions are defined and executed. They inherit from `Microsoft.AspNetCore.Mvc.Controller` and contain public methods that represent actions. Controllers help organize logic related to handling requests, responses, and other business operations.

Example:
```csharp
public class WeatherForecastController : Controller
{
    [HttpGet]
    public IActionResult Index()
    {
        // Logic for the GET request
        return View();
    }

    [HttpPost]
    public IActionResult Save(WeatherForecast forecast)
    {
        // Logic for the POST request
        return Json(new { message = "Saved successfully" });
    }
}
```
x??

---
Each flashcard follows the exact format and covers a specific concept from the provided text, ensuring clarity and understanding of the key points.

#### Controllers in ASP.NET Core Projects
Background context explaining that controllers are essential components in ASP.NET Core projects, typically placed in a folder named `Controllers`. They handle HTTP requests and return responses. The template usually includes a default controller like `HomeController`.

:p What is the purpose of the `HomeController` class in an ASP.NET Core project?
??x
The `HomeController` class serves as the primary entry point for handling HTTP requests. It contains action methods that correspond to specific URLs or endpoints, allowing you to define how these requests are processed and what response they should return.

```csharp
public class HomeController : Controller {
    public IActionResult Index() {         return View();     }
}
```
x??

---

#### Action Methods in HomeController
Explanation of action methods within the `HomeController`. These methods are responsible for processing specific HTTP requests and returning appropriate responses. In ASP.NET Core, actions can be defined as simple methods that return an `IActionResult` or a view.

:p What is the role of an action method in the `HomeController`?
??x
An action method defines how an HTTP request should be handled by the controller. It typically returns a result such as a view to render HTML content or a string containing plain text. For example, the `Index` method in `HomeController` simply returns "Hello World" when called.

```csharp
public class HomeController : Controller {
    public IActionResult Index() {         return "Hello World";     }
}
```
x??

---

#### Using Statements and Namespaces
Explanation of using statements and namespaces used in C#. `using` statements are directives that allow you to bring types from a namespace into the current code file without qualifying them with their fully qualified names.

:p What is the role of `using` statements in C#?
??x
Using statements help manage namespaces by bringing them into your current code file. This reduces typing and makes it easier to use classes, interfaces, and other members defined in those namespaces. For example, using `Microsoft.AspNetCore.Mvc;` allows you to directly reference types from the MVC namespace.

```csharp
using Microsoft.AspNetCore.Mvc;
```
x??

---

#### HomeController Class After Modifications
Explanation of changes made to the `HomeController` class in the provided text, including removal of unused code and simplification for demonstration purposes. The modified `Index` method returns a simple string "Hello World".

:p What is the new implementation of the `Index` action method in the `HomeController`?
??x
The new implementation of the `Index` action method now simply returns the string "Hello World". This change makes it easier to see how the HTTP request is processed and what response is sent back to the browser.

```csharp
public class HomeController : Controller {
    public string Index() {         return "Hello World";     }
}
```
x??

---

#### Running ASP.NET Core Project with `dotnet run` Command
Explanation of running an ASP.NET Core project using the `dotnet run` command and accessing it via a web browser. This command starts the development server, making your application available at `http://localhost:5000`.

:p How do you start the ASP.NET Core project using the `dotnet run` command?
??x
To start the ASP.NET Core project with the `dotnet run` command, navigate to the root directory of your project in the terminal and execute the following command:

```bash
dotnet run
```

Once the server starts, you can access the application via a web browser at `http://localhost:5000`. The server processes HTTP requests according to the configuration set by the template.

x??

---

