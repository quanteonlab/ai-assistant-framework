# Flashcards: Pro-ASPNET-Core-7_processed (Part 131)

**Starting Chapter:** 26.5 Working with image elements

---

#### Using Built-in Tag Helpers for CSS and JavaScript Inclusions
Background context: This section discusses how to use built-in tag helpers in ASP.NET Core to manage the inclusion of external CSS and JavaScript files. It explains dynamic loading based on the availability of CDN resources, ensuring that if a file fails to load, a fallback is provided.

:p How does the Tag Helper determine whether to use a CDN or a local fallback for Bootstrap styles?
??x
The Tag Helper determines this by checking the `display` property of an element with a specific class (in this case, `.btn`). If the property value matches what's expected (e.g., `inline-block`), it means the browser has successfully loaded the CSS from the CDN. Otherwise, it inserts a `<link>` tag for the local fallback file.

Code example:
```javascript
g = f.getElementsByTagName("SCRIPT"), 
h = g[g.length - 1].previousElementSibling,
i = f.defaultView && f.defaultView.getComputedStyle ? 
    f.defaultView.getComputedStyle(h) : h.currentStyle;
if (i && i[a] == b)
    for (e = 0; e < c.length; e++) 
        f.write('<link href="' + c[e] + '" ' + d + '/>');
```
x??

---

#### Cache Busting with ImageTagHelper
Background context: This section explains how the `ImageTagHelper` can be used to append a versioning query string to image URLs, ensuring that cached images are updated when new versions of them are available. This helps in improving performance by leveraging browser caching while still allowing for immediate reflection of changes.

:p How does ASP.NET Core ensure that an updated version of an image is fetched from the server?
??x
ASP.NET Core ensures this by using the `asp-append-version` attribute on the `<img>` tag. When present, it appends a query string to the URL with a unique identifier (version checksum), forcing the browser to fetch the latest version of the image rather than serving an outdated cached copy.

Code example:
```html
<img src="/images/city.png?v=KaMNDSZFAJufRcRDpKh0K_IIPNc7E" class="m-2" />
```
x??

---

#### ASP.NET Core Tag Helpers for Dynamic Content Management
Background context: This section describes the use of built-in tag helpers in ASP.NET Core to dynamically manage external resource loading, specifically focusing on CSS and JavaScript. It involves checking if a resource is available from a CDN or falling back to a local file.

:p How does the Tag Helper handle dynamic CSS and JavaScript inclusion?
??x
The Tag Helper checks for the availability of CSS or JavaScript files from a CDN by inspecting an element's `display` property after applying a specific class. If the property matches expected values, it indicates successful loading; otherwise, a fallback link is created to load local resources.

Code example:
```javascript
// Pseudocode
if (element.display == 'inline-block') {
    // Use CDN resource
} else {
    // Add link for local resource
}
```
x??

---

#### Checksum and Caching
Background context: The addition of a checksum ensures that any changes to a file will invalidate its cache, preventing stale content. This is particularly useful for static files like images or CSS, ensuring they are always fresh.
:p What role does the checksum play in caching?
??x
The checksum helps ensure that if any changes are made to a file (like updating an image), those changes will be recognized and the cached version of the file will be invalidated. This prevents serving stale content from caches when files have been modified.
x??

---

#### CacheTagHelper Overview
Background context: The `CacheTagHelper` is part of ASP.NET Core's built-in tag helpers, which allow for caching specific parts of a view to improve rendering performance. It uses attributes like `enabled`, `expires-on`, and `vary-by-header` among others to control the caching behavior.
:p What is the purpose of the CacheTagHelper?
??x
The purpose of the CacheTagHelper is to cache fragments of content in views or pages, which can significantly speed up rendering times. It uses various attributes like `enabled`, `expires-on`, and `vary-by-header` to manage when and how the cached content should be used.
x??

---

#### Caching Attributes Explained
Background context: The CacheTagHelper uses several attributes to configure caching behavior. These include controlling whether to enable caching, setting expiration times, and managing cache versions based on request headers or cookies.
:p What are some of the key attributes of the CacheTagHelper?
??x
Key attributes of the CacheTagHelper include:
- `enabled`: Controls whether the content is cached.
- `expires-on`: Sets an absolute time for cache expiration.
- `expires-after`: Sets a relative time after which the cache expires.
- `expires-sliding`: Specifies a sliding window period since last use before cache expiration.
- `vary-by-header`, `vary-by-query`, `vary-by-route`, `vary-by-cookie`, and `vary-by-user`: Manage different versions of cached content based on request headers, query strings, routing variables, cookies, or user authentication status.
x??

---

#### Caching in _SimpleLayout.cshtml
Background context: An example is provided where a timestamp element is cached using the CacheTagHelper. This demonstrates how caching can be applied to specific parts of a view file.
:p How does the CacheTagHelper work in the given example?
??x
In the example, the `CacheTagHelper` is used to cache an `<h6>` element containing a timestamp. When the page loads initially, both uncached and cached timestamps are shown. On subsequent reloads, only the cached content (timestamp) is displayed because it has not expired.
```html
<cache>
    <h6 class="bg-primary text-white m-2 p-2">
        Cached timestamp: @DateTime.Now.ToLongTimeString()
    </h6>
</cache>
```
x??

---

#### Caching Strategy Considerations
Background context: Effective use of caching requires careful planning. While it can improve performance, improper configuration can lead to issues like stale content and version conflicts.
:p Why is careful thought required when using the CacheTagHelper?
??x
Careful thought is necessary because improper caching strategy can result in users receiving outdated content, versioning problems where different caches have different content, or deployment issues if cached data from old application versions mixes with new ones. It's essential to have a clear performance problem that needs solving and understanding the implications of caching.
x??

---

#### Setting Cache Expiry Using `expires-*` Attributes
Background context: The `expires-*` attributes allow you to specify when cached content will expire, either as an absolute time or a relative duration. This ensures that content can be refreshed periodically based on your application's needs.

:p What is the purpose of using the `expires-after` attribute in cache management?
??x
The `expires-after` attribute allows setting a relative time interval for caching content. In Listing 26.18, it specifies that cached content should expire after 15 seconds.
```csharp
<cache expires-after="@TimeSpan.FromSeconds(15)">
    <!-- Cached content here -->
</cache>
```
x??

---
#### Fixed Expiry Point Using `expires-on` Attribute
Background context: The `expires-on` attribute lets you specify a fixed time in the future when cached content will expire. This is useful for setting an absolute expiration date that doesn't depend on current timestamps.

:p How can you set a fixed expiry point using the `expires-on` attribute?
??x
The `expires-on` attribute takes a DateTime value to indicate when cached content should expire. In Listing 26.19, it sets a very distant future time.
```csharp
<cache expires-on="@DateTime.Parse(\"2100-01-01\")">
    <!-- Cached content here -->
</cache>
```
x??

---
#### Sliding Expiry Using `expires-sliding` Attribute
Background context: The `expires-sliding` attribute defines a sliding expiry period after which cached content is discarded if it hasn't been used. This ensures that older data is refreshed more frequently.

:p What does the `expires-sliding` attribute do?
??x
The `expires-sliding` attribute sets a time interval during which unused cached content will be expired. In Listing 26.20, it specifies that cached content should expire if not used within 10 seconds.
```csharp
<cache expires-sliding="@TimeSpan.FromSeconds(10)">
    <!-- Cached content here -->
</cache>
```
x??

---
#### Using Cache Variations with `vary-by` Attributes
Background context: By default, all requests receive the same cached content. However, you can use cache variations to serve different cached contents based on specific routing parameters.

:p How can you create cache variations using the `vary-by-route` attribute?
??x
The `vary-by-route` attribute creates cache variations based on route values matched by the routing system. In Listing 26.21, it uses the `action` value to differentiate between cached content for different actions.
```csharp
<cache expires-sliding="@TimeSpan.FromSeconds(10)"
       vary-by-route="action">
    <!-- Cached content here -->
</cache>
```
x??

---

