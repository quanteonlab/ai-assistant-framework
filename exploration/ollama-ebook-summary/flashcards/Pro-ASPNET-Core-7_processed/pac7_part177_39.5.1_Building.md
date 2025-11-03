# Flashcards: Pro-ASPNET-Core-7_processed (Part 177)

**Starting Chapter:** 39.5.1 Building a simple JavaScript client

---

#### ASP.NET Core Identity Setup
Background context: This section covers how to set up and use ASP.NET Core Identity in a Blazor WebAssembly application. ASP.NET Core Identity is a component for adding user authentication and authorization features to .NET applications.

:p What are the main steps involved in setting up ASP.NET Core Identity?
??x
The main steps involve building the application, configuring middleware, mapping routes, and seeding initial data.

```csharp
var app = builder.Build();
app.UseStaticFiles();
app.UseAuthentication();  // Middleware for authentication
app.UseAuthorization();   // Middleware to authorize requests based on user roles
app.MapControllers();
app.MapControllerRoute("controllers", "controllers/{controller=Home}/{action=Index}/{id?}");
app.MapRazorPages();
app.MapBlazorHub();
app.MapFallbackToPage("_Host");
app.UseBlazorFrameworkFiles("/webassembly");
app.MapFallbackToFile("/webassembly/{*path:nonfile}", "/webassembly/index.html");

var context = app.Services.CreateScope().ServiceProvider
    .GetRequiredService<DataContext>();
SeedData.SeedDatabase(context);
IdentitySeedData.CreateAdminAccount(app.Services, app.Configuration);

app.Run();
```
x??

---

#### Authentication Configuration with AddAuthentication
Background context: The `AddAuthentication` method is used to configure authentication in ASP.NET Core applications. It allows the selection of different authentication mechanisms such as cookies-based authentication.

:p What does the `AddAuthentication` method do?
??x
The `AddAuthentication` method configures the selected authentication mechanism, typically replacing default event handlers that would otherwise trigger redirections.

```csharp
app.UseAuthentication();
```
x??

---

#### Building a Simple JavaScript Client
Background context: This section explains how to build a simple JavaScript client to consume data from an ASP.NET Core API. The focus is on demonstrating web service authentication using a basic HTML and JavaScript setup.

:p How can you create a simple JavaScript client for consuming data?
??x
You can create a simple JavaScript client by adding an HTML file (`webclient.html`) to the `wwwroot` folder and writing JavaScript code in a corresponding `.js` file. This setup allows interaction with web services using basic HTTP methods like `fetch`.

```html
<!DOCTYPE html>
<html>
<head>
    <title>Web Service Authentication</title>
    <link href="/lib/bootstrap/css/bootstrap.min.css" rel="stylesheet" />
    <script type="text/javascript" src="webclient.js"></script>
</head>
<body>
    <div id="controls" class="m-2"></div>
    <div id="data" class="m-2 p-2">No data</div>
</body>
</html>
```

```javascript
const username = "bob";
const password = "secret";

window.addEventListener("DOMContentLoaded", () => {
    const controlDiv = document.getElementById("controls");
    createButton(controlDiv, "Get Data", getData);
    createButton(controlDiv, "Log In", login);
    createButton(controlDiv, "Log Out", logout);
});

function login() { /* do nothing */ }
function logout() { /* do nothing */ }

async function getData() {
    let response = await fetch("/api/people");
    if (response.ok) {
        let jsonData = await response.json();
        displayData(...jsonData.map(item => `${item.surname}, ${item.firstname}`));
    } else {
        displayData(`Error: ${response.status}: ${response.statusText}`);
    }
}

function displayData(...items) {
    const dataDiv = document.getElementById("data");
    dataDiv.innerHTML = "";
    items.forEach(item => {
        const itemDiv = document.createElement("div");
        itemDiv.innerText = item;
        itemDiv.style.wordWrap = "break-word";
        dataDiv.appendChild(itemDiv);
    })
}

function createButton(parent, label, handler) {
    const button = document.createElement("button");
    button.classList.add("btn", "btn-primary", "m-2");
}
```
x??

---

#### ASP.NET Core Identity Overview
Background context: The provided text demonstrates how to use ASP.NET Core Identity for authentication and authorization in web applications. It explains setting up basic authentication features, handling cookie-based authentication, and restricting access to specific endpoints.

:p What is ASP.NET Core Identity used for?
??x
ASP.NET Core Identity is a framework that provides user management functionality such as authentication (logging in) and authorization (determining what users can access). It supports various authentication mechanisms including cookies.
x??

---
#### Implementing Authorization in Web Services
Background context: The text shows how to apply the `[Authorize]` attribute to restrict access to web service endpoints. This ensures that only authenticated users can access certain data.

:p How does the `[Authorize]` attribute work?
??x
The `[Authorize]` attribute is used to restrict access to controllers or actions within an ASP.NET Core application. When applied, it requires the user to be authenticated before they can access the protected endpoint.
```csharp
[ApiController]
[Route("/api/people")]
[Authorize]
public class DataController : ControllerBase { }
```
x??

---
#### Defining Authentication Actions in a Controller
Background context: The `ApiAccountController` defines actions for logging in and out. These actions use the `SignInManager` to handle authentication logic.

:p How do you define login and logout actions in a controller?
??x
You define actions in a controller to handle authentication using methods from the `SignInManager`. For example, the `Login` action signs in the user while the `Logout` action signs them out.
```csharp
[ApiController]
[Route("/api/account")]
public class ApiAccountController : ControllerBase {
    private SignInManager<IdentityUser> signinManager;

    public ApiAccountController(SignInManager<IdentityUser> mgr) { 
        signinManager = mgr; 
    }

    [HttpPost("login")]
    public async Task<IActionResult> Login([FromBody] Credentials creds) {
        // PasswordSignInAsync checks the password and signs in the user if successful
        Microsoft.AspNetCore.Identity.SignInResult result = await signinManager.PasswordSignInAsync(
            creds.Username, creds.Password, false, false);
        if (result.Succeeded) {
            return Ok();
        }
        return Unauthorized();
    }

    [HttpPost("logout")]
    public async Task<IActionResult> Logout() {
        // SignOutAsync signs the user out
        await signinManager.SignOutAsync(); 
        return Ok();
    }

    public class Credentials {
        public string Username { get; set; } = string.Empty;
        public string Password { get; set; } = string.Empty;
    }
}
```
x??

---
#### Handling HTTP Requests for Authentication and Data
Background context: The JavaScript client sends HTTP requests to the server using `fetch` to authenticate and retrieve data. It uses cookies to maintain authentication state.

:p How does the JavaScript client handle authentication?
??x
The JavaScript client handles authentication by sending a POST request to `/api/account/login` with user credentials. If successful, it receives a cookie that is included in future requests.
```javascript
async function login() {
    let response = await fetch("/api/account/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: username, password: password })
    });
    if (response.ok) {
        displayData("Logged in");
    } else {
        displayData(`Error: ${response.status}: ${response.statusText}`);
    }
}
```
x??

---
#### Restricting Access to Endpoints with HTTP Methods
Background context: The text demonstrates how the `GET` method is used to retrieve data while the `POST` methods are used for authentication actions.

:p What HTTP methods are used in the provided code?
??x
The provided code uses different HTTP methods:
- GET: Used by `getData()` function to fetch data from the server.
- POST: Used by `login()` and `logout()` functions to authenticate and deauthenticate users, respectively.
```javascript
async function getData() {
    let response = await fetch("/api/people");
    if (response.ok) {
        let jsonData = await response.json();
        displayData(...jsonData.map(item => `${item.surname}, ${item.firstname}`));
    } else {
        displayData(`Error: ${response.status}: ${response.statusText}`);
    }
}

async function logout() {
    let response = await fetch("/api/account/logout", { method: "POST" });
    if (response.ok) {
        displayData("Logged out");
    } else {
        displayData(`Error: ${response.status}: ${response.statusText}`);
    }
}
```
x??

---
#### Displaying Data and Handling Responses
Background context: The JavaScript client displays data in the UI based on the HTTP response received from the server.

:p How does the JavaScript client display data?
??x
The JavaScript client uses a function `displayData` to handle displaying different types of responses, whether it's a list of names or an error message.
```javascript
function displayData(...items) {
    const dataDiv = document.getElementById("data");
    dataDiv.innerHTML = "";
    items.forEach(item => {
        const itemDiv = document.createElement("div");
        itemDiv.innerText = item;
        itemDiv.style.wordWrap = "break-word";
        dataDiv.appendChild(itemDiv);
    })
}
```
x??

---

#### Using Cookie Authentication
Background context: ASP.NET Core Identity can use cookie-based authentication, which stores session information on the client side using cookies. This method is simple and works well for most web applications but has limitations as not all clients support cookies.

:p What is cookie authentication used for in web services?
??x
Cookie authentication is a method of maintaining user sessions by storing session information on the client side through cookies. It enables the server to recognize returning users based on the cookie that contains the necessary session data.
x??

---

#### Using Bearer Token Authentication
Background context: Cookie-based authentication may not be suitable for all web services, especially when clients do not support cookies or require more secure methods of authentication.

:p What is bearer token authentication and why might it be preferred over cookie authentication?
??x
Bearer token authentication uses a string (bearer token) that the client receives and includes in requests to authenticate themselves. Unlike cookies, tokens are opaque and do not contain any information about the user's state on the server side. This method is more secure as it can prevent interception of the token, especially when used with HTTPS.
x??

---

#### JSON Web Token (JWT)
Background context: JWT is a compact, URL-safe means of representing claims to be transferred between parties as a JSON object.

:p What is JWT and how does it differ from other tokens?
??x
JWT is a standard for securely transmitting information between parties as a JSON object. Unlike opaque tokens used in bearer token authentication, JWT contains the username or unique identifier within its payload, making it easier for the server to verify the user's identity without storing session data on the client side.
x??

---

#### Adding Configuration Settings for JWT
Background context: To use JWT with ASP.NET Core, you need to configure a key that will be used for encrypting and decrypting tokens.

:p How do you add configuration settings for JWT in the `appsettings.json` file?
??x
Add the following configuration setting to your `appsettings.json` file:
```json
{
  "jwtSecret": "jwt_secret"
}
```
Ensure that you replace `"jwt_secret"` with a secure and unique key. For production use, it is recommended to store this key securely outside of the project.
x??

---

#### Generating Tokens
Background context: When implementing JWT-based authentication, you need an endpoint to generate tokens based on user credentials.

:p How do you generate tokens in the `ApiAccountController`?
??x
In the `ApiAccountController`, add a method that receives user credentials and generates a JWT token. Here is how it can be implemented:
```csharp
[HttpPost("token")]
public async Task<IActionResult> Token(
    [FromBody] Credentials creds)
{
    if (await CheckPassword(creds))
    {
        JwtSecurityTokenHandler handler = new JwtSecurityTokenHandler();
        byte[] secret = Encoding.ASCII.GetBytes(Configuration["jwtSecret"]);
        SecurityTokenDescriptor descriptor = new SecurityTokenDescriptor
        {
            Subject = new ClaimsIdentity(new Claim[]
            {
                new Claim(ClaimTypes.Name, creds.Username)
            }),
            Expires = DateTime.UtcNow.AddHours(24),
            SigningCredentials = new SigningCredentials(
                new SymmetricSecurityKey(secret),
                SecurityAlgorithms.HmacSha256Signature
            )
        };
        SecurityToken token = handler.CreateToken(descriptor);
        return Ok(new { success = true, token = handler.WriteToken(token) });
    }
    return Unauthorized();
}
```
This method checks the password against the stored credentials and generates a JWT if successful.
x??

---

#### Checking Passwords with Identity
Background context: The `CheckPassword` method in the controller uses the `SignInManager` to verify the user's password.

:p How does the `CheckPassword` method validate passwords?
??x
The `CheckPassword` method validates the password using the `IdentityUserManager`. Here is how it works:
```csharp
private async Task<bool> CheckPassword(Credentials creds)
{
    IdentityUser? user = await UserManager.FindByNameAsync(creds.Username);
    if (user != null)
    {
        return (await SignInManager.CheckPasswordSignInAsync(user, creds.Password, true)).Succeeded;
    }
    return false;
}
```
This method first finds the user by username and then uses `CheckPasswordSignInAsync` to validate the password. If successful, it returns `true`; otherwise, it returns `false`.
x??

---

#### Example of Credentials Class
Background context: The `Credentials` class is used to hold the username and password for authentication.

:p What is the purpose of the `Credentials` class?
??x
The `Credentials` class serves as a data structure to hold the user's credentials (username and password) that will be passed in the request body. Here is its implementation:
```csharp
public class Credentials 
{
    public string Username { get; set; } = string.Empty;
    public string Password { get; set; } = string.Empty;
}
```
This class simplifies the process of handling user credentials and makes it easier to validate them.
x??

---

