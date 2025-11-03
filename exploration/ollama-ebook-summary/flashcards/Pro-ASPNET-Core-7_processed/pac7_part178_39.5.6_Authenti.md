# Flashcards: Pro-ASPNET-Core-7_processed (Part 178)

**Starting Chapter:** 39.5.6 Authenticating with tokens

---

#### Configuring Authentication with Tokens
This section explains how to set up token-based authentication for an ASP.NET Core application. The configuration involves using `JwtBearer` middleware and setting up the necessary services and configurations in the `Program.cs` file.

:p How do you configure the application to use JWT (JSON Web Token) for authentication?
??x
To configure the application to use JWT for authentication, follow these steps:

1. **Add Authentication Services**: Use the `builder.Services.AddAuthentication()` method to register authentication services and set default schemes.
2. **Configure JWT Bearer Options**: Configure the JWT bearer options with necessary parameters like `IssuerSigningKey` and `ValidateAudience`.
3. **Token Validation Events**: Implement token validation events to handle user principal creation based on the JWT claims.

Here is a detailed breakdown of how this can be implemented in code:

```csharp
builder.Services.AddAuthentication(opts => {
    opts.DefaultScheme = CookieAuthenticationDefaults.AuthenticationScheme;
    opts.DefaultChallengeScheme = CookieAuthenticationDefaults.AuthenticationScheme;
}).AddCookie()  // Configure cookie authentication if needed
.AddJwtBearer(opts => { 
    opts.RequireHttpsMetadata = false;  // Disable secure metadata check for testing purposes
    opts.SaveToken = true;              // Save the token in the context to pass it on
    opts.TokenValidationParameters = new TokenValidationParameters {
        ValidateIssuerSigningKey = true,
        IssuerSigningKey = new SymmetricSecurityKey(
            Encoding.ASCII.GetBytes(builder.Configuration["jwtSecret"])),
        ValidateAudience = false,  // Disable audience validation for simplicity
        ValidateIssuer = false     // Disable issuer validation for simplicity
    };
    opts.Events = new JwtBearerEvents {
        OnTokenValidated = async ctx => { 
            var usrmgr = ctx.HttpContext.RequestServices.GetRequiredService<UserManager<IdentityUser>>();
            var signinmgr = ctx.HttpContext.RequestServices.GetRequiredService<SignInManager<IdentityUser>>();
            string? username = ctx.Principal?.FindFirst(ClaimTypes.Name)?.Value;
            if (username != null) { 
                IdentityUser? idUser = await usrmgr.FindByNameAsync(username);
                if (idUser != null) {
                    ctx.Principal = await signinmgr.CreateUserPrincipalAsync(idUser);
                }
            }
        }
    };
});
```

In this example, the `JwtBearer` middleware is configured to validate JWT tokens. The token validation parameters include setting up a symmetric security key derived from an environment variable and disabling audience and issuer validation for simplicity.

x??

---

#### Handling Token Validation Events
This section focuses on handling events during token validation to manage user principal creation based on JWT claims.

:p How do you handle the `OnTokenValidated` event in `JwtBearerEvents`?
??x
In the `OnTokenValidated` event, you can access the JWT token claims and use them to create or update a user's principal. This is done by injecting necessary services like `UserManager<IdentityUser>` and `SignInManager<IdentityUser>` from the request services.

Here’s how it works:

1. **Retrieve Principal**: Extract the username claim from the JWT token.
2. **Find User**: Use the `UserManager` to find a user by their username (or other unique identifier).
3. **Create Principal**: If the user is found, use the `SignInManager` to create or update a principal for the current HTTP context.

Example implementation:

```csharp
opts.Events = new JwtBearerEvents {
    OnTokenValidated = async ctx => {
        var usrmgr = ctx.HttpContext.RequestServices.GetRequiredService<UserManager<IdentityUser>>();
        var signinmgr = ctx.HttpContext.RequestServices.GetRequiredService<SignInManager<IdentityUser>>();
        string? username = ctx.Principal?.FindFirst(ClaimTypes.Name)?.Value;
        if (username != null) {
            IdentityUser? idUser = await usrmgr.FindByNameAsync(username);
            if (idUser != null) {
                ctx.Principal = await signinmgr.CreateUserPrincipalAsync(idUser);
            }
        }
    }
};
```

In this code, the `OnTokenValidated` event handler retrieves the username claim from the JWT token and uses it to find or create a user in the system. The resulting principal is then set on the HTTP context.

x??

---

#### Setting Up Cookie Authentication
This section explains how to configure cookie-based authentication for the application alongside JWT-based authentication, if needed.

:p How do you add cookie-based authentication with ASP.NET Core Identity?
??x
To integrate cookie-based authentication in an ASP.NET Core application using ASP.NET Core Identity, follow these steps:

1. **Register Cookie Authentication**: Use `AddAuthentication` to register the cookie scheme and specify it as the default challenge scheme.
2. **Configure Cookie Options**: Customize the behavior of the cookie by configuring options like redirect paths for unauthorized and forbidden requests.

Example code snippet:

```csharp
builder.Services.AddAuthentication(opts => {
    opts.DefaultScheme = CookieAuthenticationDefaults.AuthenticationScheme;
    opts.DefaultChallengeScheme = CookieAuthenticationDefaults.AuthenticationScheme;
}).AddCookie(opts => {
    opts.Events.DisableRedirectForPath(e => e.OnRedirectToLogin, "/api", StatusCodes.Status401Unauthorized);
    opts.Events.DisableRedirectForPath(e => e.OnRedirectToAccessDenied, "/api", StatusCodes.Status403Forbidden);
});
```

In this example:
- The default scheme and challenge scheme are set to `CookieAuthenticationDefaults.AuthenticationScheme`.
- Custom events are configured to handle redirection for specific paths (`/api`) with appropriate HTTP status codes.

x??

---

#### Adding JWT Support to ASP.NET Core Identity
Background context: This section explains how to integrate JSON Web Tokens (JWT) with ASP.NET Core Identity. JWT is a token-based authentication mechanism that helps in securely transmitting information between parties as a JSON object. The AddJwtBearer method adds support for JWT and provides the necessary settings to decrypt tokens.

:p What does the `AddJwtBearer` method do in the context of integrating JWT with ASP.NET Core Identity?
??x
The `AddJwtBearer` method is used to add support for JWT tokens by configuring the authentication system. It sets up the validation rules for JWTs, such as issuer and audience, and provides settings required to decrypt the token payload.

```csharp
services.AddAuthentication()
    .AddJwtBearer(options =>
    {
        options.TokenValidationParameters = new TokenValidationParameters
        {
            ValidateIssuer = true,
            ValidIssuer = "yourdomain.com",
            ValidateAudience = true,
            ValidAudience = "yourdomain.com",
            ValidateLifetime = true, // Validate the expiration time of the token
            IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes("secret_key"))
        };
    });
```
x??

---

#### Configuring Authorization with Multiple Authentication Schemes
Background context: This section describes how to configure a controller in ASP.NET Core to support multiple authentication schemes, such as cookies and JWT tokens. By default, controllers might only use cookie-based authentication, but adding the `AuthenticationSchemes` attribute allows specifying additional schemes.

:p How can you enable both cookie and bearer token authentication for an API controller?
??x
To enable both cookie and bearer token authentication, you need to apply the `[Authorize]` attribute with the `AuthenticationSchemes` parameter. This specifies that the controller will accept requests authenticated by either scheme.

```csharp
[ApiController]
[Route("/api/people")]
[Authorize(AuthenticationSchemes = "Identity.Application, Bearer")]
public class DataController : ControllerBase
{
    // Controller logic here...
}
```

Here, `AuthenticationSchemes` is set to include both the default cookie scheme (`Identity.Application`) and a custom bearer token scheme.

x??

---

#### Using Tokens for Authentication in JavaScript Client
Background context: The final step involves updating the JavaScript client to authenticate using tokens. This ensures that subsequent requests from the client can be authenticated by the server, allowing access to protected resources based on the user's identity.

:p How does the `login` function in `webclient.js` handle token authentication?
??x
The `login` function in `webclient.js` sends a POST request to the `/api/account/token` endpoint with the username and password. Upon successful authentication, it extracts the token from the response and assigns it to the `token` variable.

```javascript
async function login() {
    let response = await fetch("/api/account/token", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: username, password: password })
    });
    if (response.ok) {
        token = (await response.json()).token;
        displayData("Logged in", token);
    } else {
        displayData(`Error: ${response.status}: ${response.statusText}`);
    }
}
```

The `displayData` function then uses this token to make authenticated requests to the server.

x??

---

#### Restricting Access with Tokens
Background context: Once the token-based authentication is set up, you can restrict access to certain endpoints using tokens. This ensures that only authorized users can access protected resources by including an `Authorization` header in their HTTP requests.

:p How does the `getData` function handle authorization when making API calls?
??x
The `getData` function makes a GET request to the `/api/people` endpoint with an `Authorization` header set to include the Bearer token. This ensures that the server validates the token and authorizes the user before processing the request.

```javascript
async function getData() {
    let response = await fetch("/api/people", {
        headers: { "Authorization": 'Bearer ${token}' }
    });
    if (response.ok) {
        let jsonData = await response.json();
        displayData(...jsonData.map(item => `${item.surname}, ${item.firstname}`));
    } else {
        displayData(`Error: ${response.status}: ${response.statusText}`);
    }
}
```

This approach ensures that the user must be authenticated before they can access protected data.

x??

---

#### Clearing Browser Cookies for Testing
Background context: When testing token-based authentication, it's crucial to clear browser cookies to avoid using leftover credentials from previous tests. This step is important to ensure that the new tokens are used and not cached old cookie-based sessions.

:p Why is it important to clear your browser cookies before testing JWT authentication in ASP.NET Core?
??x
Clearing your browser cookies before testing JWT authentication ensures that you start with a clean state, avoiding any potential issues from leftover cookie-based authentication. This step helps in accurately testing the behavior of token-based authentication and ensuring that the new tokens are being used correctly.

:p How can you clear browser cookies when using Chrome?
??x
To clear browser cookies in Google Chrome:
1. Open Chrome.
2. Click on the three dots (menu) in the upper-right corner.
3. Select `Settings`.
4. Scroll down to find `Privacy and security` > `Clear browsing data`.
5. In the "Time range" dropdown, select a time frame or choose `All time` for complete clearance.
6. Check the box next to `Cookies and other site data`.
7. Click on `Clear data`.

This process ensures that any leftover cookies are cleared before testing.

x??

