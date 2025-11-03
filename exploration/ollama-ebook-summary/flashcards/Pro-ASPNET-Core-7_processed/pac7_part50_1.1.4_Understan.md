# Flashcards: Pro-ASPNET-Core-7_processed (Part 50)

**Starting Chapter:** 1.1.4 Understanding the utility frameworks

---

#### Patterns and Their Application in ASP.NET Core

Background context: The provided text discusses the importance of design patterns in software development, specifically within the context of ASP.NET Core. It highlights how patterns can be useful but should not be applied blindly.

:p What are some key points regarding the use of design patterns according to this passage?
??x
The text emphasizes that while design patterns can provide solutions to common problems encountered in projects, they should be used freely and adapted as necessary. Overzealous adherence to a pattern can be detrimental if it doesn't fit the specific project's needs.
??x

---

#### Understanding Razor Pages in ASP.NET Core

Background context: The passage explains how Razor Pages improve upon the MVC framework by offering simpler application development compared to Web Forms, though they may still face challenges with complex projects.

:p How does Razor Pages differ from the MVC Framework in terms of simplicity and complexity?
??x
Razor Pages simplify application development by mixing code and content into self-contained pages. This approach allows for faster development similar to Web Forms but avoids some underlying technical issues faced by traditional ASP.NET applications.
??x

---

#### Introduction to Blazor

Background context: The text introduces Blazor, a framework that aims to bridge the gap between C# developers and JavaScript client-side frameworks, offering two main versions: Blazor Server and Blazor WebAssembly.

:p What are the two main versions of Blazor mentioned in this passage?
??x
The two main versions of Blazor discussed are Blazor Server and Blazor WebAssembly. 
Blazor Server relies on a persistent HTTP connection to an ASP.NET Core server, where C# code is executed.
Blazor WebAssembly executes the application’s C# code directly within the browser.
??x

---

#### Using Razor Pages with MVC Framework

Background context: The passage describes how Razor Pages can be used alongside the MVC Framework in certain scenarios, such as for secondary features like administration tools.

:p How does the author recommend using Razor Pages and MVC Framework together?
??x
The author suggests writing the main parts of the application using the MVC Framework while using Razor Pages for secondary features, such as administration or reporting tools. This approach is demonstrated in chapters 7-11 of a hypothetical ASP.NET Core application called SportsStore.
??x

---

#### Scaling Up Complex Projects with Razor Pages

Background context: The text acknowledges that although Razor Pages offer simplicity, scaling up complex projects can still be an issue.

:p What potential challenge does the author mention regarding complex projects and Razor Pages?
??x
The author notes that while Razor Pages provide a simpler development process, scaling up complex projects can still be problematic.
??x

---

#### Introduction to Blazor (Continued)

Background context: The text continues by explaining how Blazor aims to make client-side applications more accessible for C# developers.

:p What is the primary goal of using Blazor according to this passage?
??x
The primary goal of using Blazor is to bridge the gap between C# and JavaScript client-side frameworks, allowing C# to be used for writing client-side applications.
??x

---

#### Blazor Server vs. WebAssembly

Background context: The text provides details on the two versions of Blazor and their unique execution environments.

:p How do Blazor Server and Blazor WebAssembly differ in terms of where the application's C# code is executed?
??x
Blazor Server relies on a persistent HTTP connection to an ASP.NET Core server, executing the application’s C# code on the server.
Blazor WebAssembly executes the application’s C# code directly within the browser using WebAssembly technology.
??x

---

---
#### Entity Framework Core Overview
Entity Framework Core is Microsoft’s object-relational mapping (ORM) framework, used to represent data stored in a relational database as .NET objects. It simplifies database access by abstracting the underlying SQL operations.

:p What does Entity Framework Core do?
??x
Entity Framework Core maps .NET classes to tables in a relational database and enables developers to interact with the database using object-oriented principles rather than writing raw SQL queries.

Code Example:
```csharp
public class Customer {
    public int Id { get; set; }
    public string Name { get; set; }
}

using (var context = new MyDbContext()) {
    var customer = new Customer { Name = "John Doe" };
    context.Customers.Add(customer);
    context.SaveChanges();
}
```
x??

---
#### ASP.NET Core Identity Overview
ASP.NET Core Identity is a framework provided by Microsoft for handling user authentication and authorization in .NET applications. It helps validate user credentials, manage roles, and control access to application features.

:p What does ASP.NET Core Identity provide?
??x
ASP.NET Core Identity manages user authentication (login, logout) and authorization (role-based access control). It allows developers to define custom user claims and roles, and restrict access to specific application features based on these roles.

Code Example:
```csharp
public void ConfigureServices(IServiceCollection services) {
    services.AddDbContext<ApplicationDbContext>(options =>
        options.UseSqlServer(Configuration.GetConnectionString("DefaultConnection")));
    services.AddDefaultIdentity<IdentityUser>()
        .AddEntityFrameworkStores<ApplicationDbContext>();
}

public class Startup {
    public void Configure(IApplicationBuilder app, IWebHostEnvironment env) {
        // Middleware for authentication
        app.UseAuthentication();
        app.UseAuthorization();

        app.UseEndpoints(endpoints => {
            endpoints.MapControllerRoute(
                name: "default",
                pattern: "{controller=Home}/{action=Index}/{id?}");
        });
    }
}
```
x??

---
#### ASP.NET Core Platform Overview
The ASP.NET Core platform contains the low-level features required to process HTTP requests and create responses. It includes an integrated HTTP server, middleware components for handling requests, and core features like URL routing and the Razor view engine.

:p What are the key components of the ASP.NET Core platform?
??x
The key components include:
- An HTTP server that listens for incoming requests.
- Middleware to handle request processing in a pipeline.
- URL routing to map URLs to controller actions.
- The Razor view engine to render HTML content dynamically.

Code Example:
```csharp
public class Startup {
    public void Configure(IApplicationBuilder app, IWebHostEnvironment env) {
        if (env.IsDevelopment()) {
            app.UseDeveloperExceptionPage();
        } else {
            app.UseExceptionHandler("/Home/Error");
        }

        // Middlewares for static files and routing
        app.UseStaticFiles();
        app.UseRouting();

        // Middleware for authentication and authorization
        app.UseAuthentication();
        app.UseAuthorization();

        app.UseEndpoints(endpoints => {
            endpoints.MapControllerRoute(
                name: "default",
                pattern: "{controller=Home}/{action=Index}/{id?}");
        });
    }
}
```
x??

---
#### SignalR Overview
SignalR is a framework that enables real-time web functionality, allowing for low-latency communication between applications. It provides the foundation for Blazor Server but is rarely used directly due to more modern alternatives like Azure Event Grid or Service Bus.

:p What does SignalR enable?
??x
SignalR enables real-time bidirectional communication between clients and servers over WebSockets, Server-Sent Events (SSE), and long polling. It simplifies the implementation of features such as chat applications, live updates, and collaborative editing.

Code Example:
```csharp
public class ChatHub : Hub {
    public async Task SendMessage(string user, string message) {
        await Clients.All.SendAsync("ReceiveMessage", user, message);
    }
}
```
x??

---

#### gRPC Overview and Limitations
gRPC is an emerging standard for cross-platform remote procedure calls (RPCs) over HTTP, initially created by Google. It provides efficiency and scalability benefits but cannot be used directly in web applications due to requirements on low-level control of HTTP messages that browsers do not support.
:p What does gRPC offer and why can't it be used directly in web applications?
??x
gRPC offers efficiency and scalability for cross-platform RPCs over HTTP. However, its use requires low-level control of HTTP messages which browsers don’t allow directly. There's a browser library allowing gRPC via a proxy server but that undermines some benefits.
```csharp
// Example C# code using gRPC in the back-end
public class GreeterService : Greeter.GreeterBase {
    public override Task<HelloReply> SayHello(HelloRequest request, ServerCallContext context) {
        return Task.FromResult(new HelloReply { Message = "Hello " + request.Name });
    }
}
```
x??

---

#### Understanding This Book
This book requires familiarity with web development basics, HTML/CSS understanding, and C#. It emphasizes ASP.NET Core and C#, so JavaScript for client-side work is not a necessity. Chapter 5 covers important C# features relevant to ASP.NET Core.
:p What background knowledge does one need to follow this book?
??x
One needs to be familiar with the basics of web development, understand how HTML and CSS function, and have a working knowledge of C#. The book focuses on C# and ASP.NET Core, so detailed client-side JavaScript skills are not required. Chapter 5 outlines essential C# features for ASP.NET Core.
```csharp
// Example C# code snippet
public class MyController : Controller {
    public IActionResult Index() {
        return View();
    }
}
```
x??

---

#### Required Software for Examples
For following examples, you need a code editor (Visual Studio or Visual Studio Code), .NET Core SDK, and SQL Server LocalDB. These are available free from Microsoft. Chapter 2 provides installation instructions.
:p What software is required to follow the book's examples?
??x
You need a code editor like Visual Studio or Visual Studio Code, .NET Core SDK, and SQL Server LocalDB. All these tools can be downloaded for free from Microsoft. Chapter 2 includes detailed instructions on how to install them.
```csharp
// Example C# code snippet using the .NET Core SDK
public class Program {
    public static void Main(string[] args) {
        CreateHostBuilder(args).Build().Run();
    }

    public static IHostBuilder CreateHostBuilder(string[] args) =>
        Host.CreateDefaultBuilder(args)
            .ConfigureWebHostDefaults(webBuilder => { webBuilder.UseStartup<Startup>(); });
}
```
x??

---

#### Platform Requirements
The book is written for Windows, but it should work on any version supported by Visual Studio, VS Code, and .NET Core. SQL Server LocalDB is specific to Windows.
:p What platform requirements are specified for following the examples?
??x
The book is designed for Windows, though other versions of Windows compatible with Visual Studio, VS Code, and .NET Core should work. However, some examples rely on SQL Server LocalDB, which is specific to Windows. For non-Windows platforms, contact the author for general guidance.
```csharp
// Example C# code snippet using Entity Framework with SQL Server LocalDB
public class ApplicationDbContext : DbContext {
    public DbSet<User> Users { get; set; }
}
```
x??

---

#### Troubleshooting Examples
If problems occur, start again from the beginning of the chapter. Most issues arise from missing steps or not following listings thoroughly. Check the errata list available in the GitHub repository.
:p What should you do if you encounter difficulties while following examples?
??x
Start over from the beginning of the relevant chapter. Most issues stem from missing a step or not fully following the code listings. Refer to the errata/corrections list included in the book’s GitHub repository for known errors and solutions.
```csharp
// Example C# code snippet showing how to follow instructions carefully
public void ProcessRequest() {
    // Ensure all steps are followed correctly
    // Highlighted text in listings show necessary changes
}
```
x??

