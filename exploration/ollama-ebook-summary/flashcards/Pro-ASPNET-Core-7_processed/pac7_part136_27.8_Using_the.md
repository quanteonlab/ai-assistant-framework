# Flashcards: Pro-ASPNET-Core-7_processed (Part 136)

**Starting Chapter:** 27.8 Using the anti-forgery feature

---

#### Anti-Forgery Feature in Form Handling
The anti-forgery feature is a security measure implemented by Tag Helpers, such as `TextAreaTagHelper`, to protect against Cross-Site Request Forgery (CSRF) attacks. CSRF exploits occur when an attacker tricks a user into performing actions on a web application without their consent.
:p What does the anti-forgery feature do?
??x
The anti-forgery feature generates and validates tokens to ensure that form submissions are legitimate and not forged by malicious sites. When using `FormTagHelper`, it adds a hidden field containing an anti-forgery token, which must be validated on the server side.
```csharp
@using (Html.BeginForm())
{
    @Html.AntiForgeryToken()
}
```
The `AntiForgeryToken` method generates and includes the necessary tokens in the form. On the server-side, validation is done automatically by ASP.NET Core.
x??

---
#### Removing a Filter for Form Data
In the provided code, a filter was applied to exclude form data with keys starting with an underscore. This was done to focus on specific values from the HTML elements in the form. However, this filtering logic can be removed if all form data needs to be stored temporarily.
:p What is the impact of removing the filter for form data?
??x
By removing the filter, all form data received from the HTML form will be stored in `TempData`. This means that every piece of data submitted through the form will be available during the next request. The original logic was to exclude CSRF tokens and similar system-generated keys, but now it stores everything.
```csharp
[HttpPost]
public IActionResult SubmitForm()
{
    foreach (string key in Request.Form.Keys)
    {
        TempData[key] = string.Join(", ", (string?)Request.Form[key]);
    }
    return RedirectToAction(nameof(Results));
}
```
The revised method no longer filters out keys starting with an underscore, ensuring all form data is captured and stored.
x??

---
#### Cross-Site Request Forgery (CSRF) Exploitation
Cross-Site Request Forgery (CSRF), also known as XSRF, is a type of web application vulnerability. It occurs when an attacker tricks a user into performing actions on a web application without their knowledge or consent. This typically happens after the user has authenticated with the web application and then visits a malicious site.
:p How does CSRF work?
??x
CSRF works by exploiting the fact that cookies are used to identify which requests belong to a specific session, often associated with a user identity. A malicious site can send a crafted request (e.g., via JavaScript) to your application, including the necessary session cookie without needing explicit consent from the user. Since the browser includes the cookie in the request, and the application processes it as part of an active session, the operation is performed without the user's knowledge.
```javascript
// Example of malicious JavaScript code on a malicious site
document.getElementById("csrfForm").submit();
```
This JavaScript code submits a form to your application, performing some action such as making a purchase or changing settings. The request includes the necessary cookies, and because the session is still active, the application processes it.
x??

---
#### Handling Form Data in ASP.NET Core
In the provided `FormController`, the controller handles form submissions by storing all form data in `TempData`. This allows the stored data to be accessed during subsequent requests. The method for handling this was modified to remove a specific filter that excluded keys starting with an underscore.
:p How is form data handled in ASP.NET Core when using TempData?
??x
Form data received from the HTML form is now fully captured and stored in `TempData` without any filtering. This means all submitted values, including potential CSRF tokens or other system-generated keys, are preserved for later use.
```csharp
[HttpPost]
public IActionResult SubmitForm()
{
    foreach (string key in Request.Form.Keys)
    {
        TempData[key] = string.Join(", ", (string?)Request.Form[key]);
    }
    return RedirectToAction(nameof(Results));
}
```
By iterating over all keys and storing them, the application can access this data as needed during subsequent requests. This approach ensures that all form-related data is available for processing.
x??

---

---
#### Anti-Forgery Feature for Form Elements Without Action Attributes
Background context explaining the concept. In ASP.NET Core, when a form element does not contain an `action` attribute (meaning it is dynamically generated using routing attributes like `asp-controller`, `asp-action`, and `asp-page`), the `FormTagHelper` automatically enables an anti-CSRF feature by adding a security token as both a cookie and a hidden input field in the HTML form.
:p What happens when a form element does not have an action attribute?
??x
When a form element lacks an action attribute, the `FormTagHelper` adds a security token to the response as a cookie. This token is also included within the form as a hidden input field. This mechanism helps prevent Cross-Site Request Forgery (CSRF) attacks by ensuring that forms are submitted with valid tokens.
x??

---
#### Enabling Anti-Forgery Feature in a Controller
Background context explaining the concept. To ensure that only requests containing valid security tokens are processed, you can enable anti-forgery validation at the controller level using an attribute. By default, controllers accept POST requests without requiring these tokens, but applying certain attributes can change this behavior.
:p How do you enable the anti-forgery feature in a controller?
??x
To enable the anti-forgery feature, apply the `[AutoValidateAntiforgeryToken]` attribute to your controller class. This attribute ensures that only requests containing valid security tokens are processed by the controller methods.
```csharp
using Microsoft.AspNetCore.Mvc;

namespace WebApp.Controllers {
    [AutoValidateAntiforgeryToken]
    public class FormController : Controller {
        // Controller methods here
    }
}
```
x??

---
#### Example of Enabling Anti-Forgery in `FormController.cs`
Background context explaining the concept. The provided code snippet shows how to enable anti-forgery validation for a controller by applying the `[AutoValidateAntiforgeryToken]` attribute.
:p How is the anti-forgery feature enabled in the `FormController` class?
??x
The anti-forgery feature is enabled by adding the `[AutoValidateAntiforgeryToken]` attribute to the `FormController` class. This ensures that any POST requests made to this controller will require a valid security token.
```csharp
[AutoValidateAntiforgeryToken]
public class FormController : Controller {
    // Controller methods here
}
```
x??

---
#### Handling Form Submission in `FormController`
Background context explaining the concept. The provided code snippet includes logic for handling form submissions, including storing form data in `TempData` and redirecting to a results view.
:p What does the `SubmitForm` method do?
??x
The `SubmitForm` method handles form submission by iterating through the form keys received via POST request. It stores each key-value pair into `TempData`, which can then be used across multiple requests, and finally redirects to the `Results` action.
```csharp
[HttpPost]
public IActionResult SubmitForm() {
    foreach (string key in Request.Form.Keys) {
        TempData[key] = string.Join(", ", (string?)Request.Form[key]);
    }
    return RedirectToAction(nameof(Results));
}
```
x??

---
#### Displaying Results View After Form Submission
Background context explaining the concept. The `Results` method simply returns a view without performing any specific actions, indicating that the form submission was successful.
:p What does the `Results` action do?
??x
The `Results` action merely returns a view named "Results" without executing any additional logic. This indicates to the user that their form submission was successful and they can see the results on this page.
```csharp
public IActionResult Results() {
    return View();
}
```
x??

---

---
#### Using Tag Helpers to Handle Forms
Background context explaining how form handling works in ASP.NET Core. Tag helpers such as `FormTagHelper` are used to simplify and secure form creation, ensuring that forms target specific actions or pages.

:p What is the role of the `FormTagHelper` class?
??x
The `FormTagHelper` class simplifies the process of generating HTML forms in Razor views by targeting specific action methods or Razor pages. It ensures that the form submission correctly binds to the intended controller action and can incorporate anti-forgery tokens for security.

```html
<form asp-page-handler="submit" method="post">
    <!-- Form elements here -->
</form>
```
x??

---
#### Controlling Anti-Forgery Token Validation with Attributes
Explanation on how different attributes like `AutoValidateAntiforgeryToken`, `IgnoreAntiforgeryToken`, and `ValidateAntiForgeryToken` can be used to control the validation of anti-forgery tokens. These attributes are crucial for securing HTTP requests, especially POSTs.

:p How does the `AutoValidateAntiforgeryToken` attribute work?
??x
The `AutoValidateAntiforgeryToken` attribute is automatically applied by ASP.NET Core and ensures that all non-GET requests (including POST, PUT, DELETE) are checked against anti-forgery tokens. It simplifies security without requiring manual application of the token validation attributes.

```csharp
[AutoValidateAntiforgeryToken]
public class SomeController : Controller {
    // Action methods here
}
```
x??

---
#### Enabling Anti-Forgery in Razor Pages
Explanation on how anti-forgery is enabled by default in Razor Pages and how to disable it using the `IgnoreAntiforgeryToken` attribute. This is relevant for scenarios where custom validation logic might be necessary.

:p How can you enable request validation in a Razor Page?
??x
To enable request validation in a Razor Page, you need to remove the `[IgnoreAntiforgeryToken]` attribute from the page handler method. This ensures that the anti-forgery token is validated during form submission, enhancing security for the specific pages.

```csharp
public class FormHandlerModel : PageModel {
    // Remove this line if validation needs to be enabled
    //[IgnoreAntiforgeryToken]
}
```
x??

---
#### Using Anti-Forgery Tokens with JavaScript Clients
Explanation on how anti-forgery tokens can be used with JavaScript clients, including sending the token as a cookie and reading it in requests. This is particularly useful for web services.

:p How does the ASP.NET Core application configure the anti-forgery feature for use with JavaScript clients?
??x
The ASP.NET Core application configures the anti-forgery feature to send the anti-forgery token via a `X-XSRF-TOKEN` header and sets it as an HTTP-only cookie named `XSRF-TOKEN`. A custom middleware is used to inject this token into outgoing requests, ensuring that JavaScript clients can include it in their POST requests.

```csharp
builder.Services.Configure<AntiforgeryOptions>(opts => {
    opts.HeaderName = "X-XSRF-TOKEN";
});

app.Use(async (context, next) => {
    if (.context.Request.Path.StartsWithSegments("/api")) {
        string? token = antiforgery.GetAndStoreTokens(context).RequestToken;
        if (token != null) {
            context.Response.Cookies.Append("XSRF-TOKEN", token,
                new CookieOptions { HttpOnly = false });
        }
    }
    await next();
});
```
x??

---
#### Implementing a JavaScript Client for Anti-Forgery Tokens
Explanation on how to create a simple JavaScript client that reads the anti-forgery token from the cookie and includes it in HTTP requests.

:p How does the `JavaScriptForm.cshtml` Razor Page handle form submission with an anti-forgery token?
??x
The `JavaScriptForm.cshtml` Razor Page reads the anti-forgery token from a cookie, appends it to the request headers, and submits the form data using the Fetch API. This ensures that the request is validated by the server.

```javascript
async function sendRequest() {
    const token = document.cookie.replace(/(?:(?:^|.*;\s*)XSRF-TOKEN\s*\=\s*([^;]*).*$)|^.*$/, '$1');
    let form = new FormData();
    form.append("name", "Paddle");
    form.append("price", 100);
    form.append("categoryId", 1);
    form.append("supplierId", 1);
    let response = await fetch("@Url.Page(\"FormHandler\")", {
        method: "POST",
        headers: { "X-XSRF-TOKEN": token },
        body: form
    });
    document.getElementById("content").innerHTML = await response.text();
}
```
x??

---

