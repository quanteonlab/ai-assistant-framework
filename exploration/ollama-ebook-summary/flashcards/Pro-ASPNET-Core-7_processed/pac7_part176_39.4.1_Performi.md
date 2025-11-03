# Flashcards: Pro-ASPNET-Core-7_processed (Part 176)

**Starting Chapter:** 39.4.1 Performing authorization in Blazor components

---

#### AuthorizeRouteView Component in Blazor
Background context: The `AuthorizeRouteView` component in Blazor is used to implement more granular authorization for applications that require different levels of access control based on user roles or claims. This allows you to display specific content depending on whether the current user has the required permissions.

:p What does the `AuthorizeRouteView` component do?
??x
The `AuthorizeRouteView` component in Blazor is designed to provide finer-grained authorization by displaying different contents based on the user's authentication status and roles. It works within a routing context, meaning it can control access at the route level rather than applying a blanket authorization policy to all components.

Example of how this might look:
```razor
@using Microsoft.AspNetCore.Components.Authorization

<Router AppAssembly="typeof(Program).Assembly">
    <Found>
        <AuthorizeRouteView RouteData="@context" DefaultLayout="typeof(NavLayout)">
            <NotAuthorized Context="authContext">
                <h4 class="bg-danger text-white text-center p-2">Not Authorized</h4>
                <div class="text-center">You may need to log in as a different user</div>
            </NotAuthorized>
        </AuthorizeRouteView>
    </Found>
    <NotFound>
        <h4 class="bg-danger text-white text-center p-2">No Matching Route Found</h4>
    </NotFound>
</Router>
```

x??

---

#### Restricting Access to DepartmentList Component
Background context: In the provided example, access to the `DepartmentList` component is restricted to users who belong to the "Admins" role. This is achieved by applying an `Authorize` attribute directly to the component.

:p How does the `@attribute [Authorize(Roles = "Admins")]` directive work?
??x
The `@attribute [Authorize(Roles = "Admins")]` directive restricts access to the `DepartmentList` component only to users who have been assigned the "Admins" role. If a user attempts to access this page without having that specific role, they will be redirected or shown a specified unauthorized content.

Code example:
```razor
@page "/departments"
@page "/depts"

@using Microsoft.AspNetCore.Authorization

@attribute [Authorize(Roles = "Admins")]

<CascadingValue Name="BgTheme" Value="Theme" IsFixed="false">
    <TableTemplate RowType="Department" RowData="Departments"
                   Highlight="@(d => d.Name)"
                   SortDirection="@(d => d.Name)">
        <Header>
            <tr>
                <th>ID</th><th>Name</th><th>People</th><th>Locations</th>
            </tr>
        </Header>
        <RowTemplate Context="d">
            <td>@d.Departmentid</td>
            <td>@d.Name</td>
            <td>
                @(String.Join(", ", d.People..Select(p => p.Surname)))
            </td>
            <td>
                @(String.Join(", ", d.People..Select(p =>
                    p.Location..City).Distinct()))
            </td>
        </RowTemplate>
    </TableTemplate>
</CascadingValue>

<SelectFilter Title="@("Theme")" Values="Themes"
             @bind-SelectedValue="Theme" />

<button class="btn btn-primary" @onclick="HandleClick">
People
</button>

@code {
    [Inject]
    public DataContext? Context { get; set; }

    public IEnumerable<Department>? Departments => Context?.Departments?
        .Include(d => d.People.).ThenInclude(p => p.Location.);
    
    public string Theme { get; set; } = "info";
    public string[] Themes =
        new string[] { "primary", "info", "success" };
    
    [Inject]
    public NavigationManager? NavManager { get; set; }

    public void HandleClick() => NavManager?.NavigateTo("/people");
}
```

x??

---

#### Testing Authorization in Blazor
Background context: To test the authorization logic, you can log out and then attempt to access restricted pages. You will see different content depending on whether your user is authenticated or not.

:p How do you test the authorization setup?
??x
To test the authorization setup, follow these steps:
1. Log out by navigating to `http://localhost:5000/account/logout`.
2. Navigate back to `http://localhost:5000` and log in with a non-admin user.
3. Click on the "Departments" button, which should show you the unauthorized content defined in your routing configuration.

Example:
Log out using:
```
http://localhost:5000/account/logout
```

Then navigate to the application and try accessing:
```
http://localhost:5000/departments
```

You will see a message indicating that you are not authorized, as shown in Figure 39.8.

x??

---

#### Logging In as Admin User
Background context: To access restricted components, you need to authenticate with an admin user. This demonstrates how the authorization rules work and ensures only users with the correct roles can use certain features.

:p How do you log in as an admin user?
??x
To log in as an admin user, follow these steps:
1. Log out by navigating to `http://localhost:5000/account/logout`.
2. Navigate back to `http://localhost:5000` and log in with the username "admin" and password "secret".
3. After logging in, try accessing the restricted components such as the "Departments" button.

Example:
Log out using:
```
http://localhost:5000/account/logout
```

Then navigate to the application and log in with:
```
Username: admin
Password: secret
```

After successful login, you will be able to access the restricted components without seeing the unauthorized content.

x??

---

---
#### AuthorizeView Component Overview
Background context explaining how `AuthorizeView` is used to restrict content visibility based on user roles. It allows showing different contents depending on whether a user is authorized or not.

:p What is the role of the `AuthorizeView` component in Blazor applications?
??x
The `AuthorizeView` component is used to conditionally render content based on user authorization. If the current user has the specified roles, the "Authorized" section will be rendered; otherwise, the "NotAuthorized" section is shown.

```razor
<AuthorizeView Roles="Admins">
    <Authorized>
        <!-- Content for authorized users -->
    </Authorized>
    <NotAuthorized>
        <!-- Content for unauthorized users -->
    </NotAuthorized>
</AuthorizeView>
```
x??

---
#### Applying AuthorizeView to Display Location Data
Background context about how `AuthorizeView` is used within the `DepartmentList.razor` component to conditionally display location data only to admin users.

:p How does `AuthorizeView` control the visibility of location data in a Blazor application?
??x
The `AuthorizeView` component checks if the current user has the "Admins" role before displaying the location data. If the user is not an admin, it shows a message indicating that they are not authorized.

```razor
<AuthorizeView Roles="Admins">
    <Authorized>
        @(String.Join(", ",d.People.Select(p => p.Location.City).Distinct()))
    </Authorized>
    <NotAuthorized>
        (Not authorized)
    </NotAuthorized>
</AuthorizeView>
```
x??

---
#### Testing Authorization with Different Users
Background context about testing the authorization mechanism by switching between different users and observing the behavior of the application.

:p How do you test if a user is authorized to view specific content in a Blazor application?
??x
You can test the authorization by changing the authenticated user's roles. For instance, logging in as `bob` should not show location data, while logging in as `admin` should grant access to that information.

:p What steps are involved in testing with different users?
??x
1. Start ASP.NET Core.
2. Authenticate as `bob` using a password (e.g., "secret").
3. Request the URL (`http://localhost:5000/depts`). Observe that location data is not shown.
4. Then, authenticate as `admin` with the same password ("secret") and request the same URL again.
5. Notice that the admin user can see the location data.

:p What URLs are used in this testing scenario?
??x
The URLs used for testing are:
- To test unauthorized access: `http://localhost:5000/depts` (authenticate as `bob`)
- To test authorized access: `http://localhost:5000/depts` (authenticate as `admin`)

x??

---
#### Using @page Directives in Blazor
Background context about how `@page` directives are used to define the URL path for a Blazor component.

:p How do you use the `@page` directive in a Blazor component?
??x
The `@page` directive is used to specify the URL path where the component should be rendered. For example, it can be set up as follows:

```razor
@page "/departments"
@page "/depts"
```
This tells the application that this component should handle requests for both `/departments` and `/depts`.

x??

---
#### CascadingValue for Theme Management
Background context about how `CascadingValue` is used to pass values down through nested components.

:p What is the purpose of the `CascadingValue` component in Blazor?
??x
The `CascadingValue` component is used to cascade a value (such as a theme) from a parent component to child components. This allows for centralized management and consistent application-wide settings.

```razor
<CascadingValue Name="BgTheme" Value="Theme" IsFixed="false">
    <!-- Child components that can access the Theme -->
</CascadingValue>
```
The `Name` attribute specifies the name of the value being passed, while `Value` is the actual value to be passed. `IsFixed="false"` means the value can change as needed.

x??

---

#### CookieAuthenticationExtensions Class and Method
Background context explaining that this class is part of ASP.NET Core Identity, used to customize authentication behavior for web services. It involves handling redirects when a user tries to access unauthorized content.

:p What is the purpose of the `CookieAuthenticationExtensions` class?
??x
The purpose of the `CookieAuthenticationExtensions` class is to provide an extension method that allows customizing the redirection behavior in ASP.NET Core Identity, particularly for scenarios where direct HTML form-based authentication isn't feasible. This customization helps in handling web service authentication by preventing redirections when specific paths are requested.

```csharp
public static void DisableRedirectForPath(
    this CookieAuthenticationEvents events,
    Expression<Func<CookieAuthenticationEvents, Func<RedirectContext<CookieAuthenticationOptions>, Task>>> expr,
    string path, int statuscode)
```

The method `DisableRedirectForPath` takes an `events` object from the authentication system, an expression defining which event to modify, a path to check against, and a status code to return when the path is matched. This method effectively allows you to define conditions under which redirections should be disabled.

x??

---

#### Disabling Redirection for API Endpoints
Background context explaining that web services often need different authentication handling compared to traditional web applications due to their stateless nature and direct client-server communication, where HTML form-based redirects are not possible. The example provided uses the `DisableRedirectForPath` method to prevent redirections when the request path starts with `/api`.

:p How does the `Program.cs` file configure Cookie Authentication for API endpoints?
??x
In the `Program.cs` file, the configuration disables redirections for paths starting with `/api`. This is done by replacing the default handlers for `OnRedirectToLogin` and `OnRedirectToAccessDenied` events. When a request path starts with `/api`, instead of redirecting to a login page or denying access via a URL redirect, the application returns HTTP 401 Unauthorized (for unauthorized content) or HTTP 403 Forbidden status codes.

The relevant code snippet from the provided text is:

```csharp
builder.Services.AddAuthentication(opts => {
    opts.DefaultScheme = CookieAuthenticationDefaults.AuthenticationScheme;
    opts.DefaultChallengeScheme = CookieAuthenticationDefaults.AuthenticationScheme;
}).AddCookie(opts => {
    opts.Events.DisableRedirectForPath(e => e.OnRedirectToLogin, "/api", StatusCodes.Status401Unauthorized);
    opts.Events.DisableRedirectForPath(e => e.OnRedirectToAccessDenied, "/api", StatusCodes.Status403Forbidden);
});
```

This configuration ensures that when a user tries to access an API endpoint without proper authentication or authorization, the application responds with appropriate HTTP status codes instead of attempting to redirect.

x??

---

