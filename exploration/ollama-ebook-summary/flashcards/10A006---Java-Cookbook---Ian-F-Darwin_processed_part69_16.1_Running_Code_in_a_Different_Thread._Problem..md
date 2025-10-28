# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 69)

**Starting Chapter:** 16.1 Running Code in a Different Thread. Problem. Solution. Discussion

---

#### Runnable Interface
Background context explaining the `Runnable` interface, its implementation, and why it is used. This interface has only one method: `run()`, which does not return a value but runs as an independent thread of execution.

:p What is the `Runnable` interface in Java?
??x
The `Runnable` interface is designed to allow you to pass code that should be executed by another thread, without creating a new class. It has only one method: `run()`, which does not return any value but performs some computation or task.

Here’s an example of how you can implement the `Runnable` interface and run it in a separate thread:

```java
public class MyRunnable implements Runnable {
    @Override
    public void run() {
        System.out.println("Task is running in a different thread");
    }
}
```

To execute this `Runnable`, you would typically create a new thread as follows:

```java
Thread thread = new Thread(new MyRunnable());
thread.start();
```
x??

---

#### Callable Interface
Background context explaining the `Callable` interface, its implementation, and how it differs from the `Runnable` interface. The `Callable` interface introduces the `call()` method which returns a result.

:p What is the `Callable` interface in Java?
??x
The `Callable` interface extends the `Runnable` interface by adding one method: `call()`, which returns a value and can throw an exception. Unlike `Runnable`, `Callable` allows you to return data from the thread after it has executed its task.

Here’s how you might implement the `Callable` interface:

```java
public class MyCallable implements Callable<Integer> {
    @Override
    public Integer call() throws Exception {
        return 42; // Returning an integer value
    }
}
```

To use a `Callable`, it is typically passed to an `ExecutorService` via its `submit()` method, which returns a `Future`.

```java
ExecutorService executor = Executors.newSingleThreadExecutor();
Future<Integer> futureResult = executor.submit(new MyCallable());
int result = futureResult.get(); // Blocking until the task completes and retrieves the value
```
x??

---

#### ExecutorService for Threading
Background context explaining how to use an `ExecutorService` for managing threads in Java. It allows you to submit tasks (either `Runnable` or `Callable`) without directly creating thread objects.

:p How do you use an `ExecutorService` to handle threading?
??x
An `ExecutorService` is a service class that can execute code asynchronously, making it easier to manage multiple threads. You obtain an `ExecutorService` by invoking methods on the `Executors` utility class.

Here’s how you might create and use an `ExecutorService`:

```java
final ExecutorService pool = Executors.newFixedThreadPool(4); // Creating a thread pool with 4 threads

// Submitting tasks to the executor service
for (int i = 0; i < 10; i++) {
    Future<Integer> future = pool.submit(() -> {
        Thread.sleep(1000); // Simulating long-running task
        return i;
    });
    
    System.out.println("Task submitted");
}

// Waiting for all tasks to complete and then shutting down the executor service
try {
    Thread.sleep(3 * 1000);
} catch (InterruptedException e) {
    e.printStackTrace();
}
pool.shutdown();
```

In this example, `ExecutorService` manages the threads efficiently, allowing you to submit multiple tasks without worrying about thread creation and management.

??x
The key advantage of using an `ExecutorService` is that it abstracts away the details of creating and managing threads. It handles the task distribution and ensures that resources are managed effectively.

The code creates a fixed-size thread pool with 4 threads, submits tasks to this pool, and then waits for all submitted tasks to complete before shutting down the service.
x??

---

#### Future Interface
Background context explaining the `Future` interface and its methods. The `Future` represents a result that will become available in the future.

:p What is the `Future` interface used for?
??x
The `Future` interface is an abstract representation of some result to be computed later. It can return values, and indicate whether computation has completed. The main purpose is to allow you to perform asynchronous tasks without blocking your current thread until those tasks complete.

Here are some key methods in the `Future` interface:

```java
public interface Future<V> {
    public boolean isDone(); // Checks if the task is done
    public V get() throws InterruptedException, ExecutionException; // Blocks until result is available and returns it
    public V get(long timeout, TimeUnit unit) throws InterruptedException, ExecutionException, TimeoutException; // Similar to `get`, with a timeout
    public boolean cancel(boolean mayInterruptIfRunning); // Cancels the task if not already started or in progress
    public boolean isCancelled(); // Checks if the task has been cancelled
}
```

Here’s an example of using the `Future` interface:

```java
ExecutorService executor = Executors.newFixedThreadPool(1);
Future<Integer> futureResult = executor.submit(() -> {
    Thread.sleep(2000); // Simulating a long-running task
    return 42;
});

// Waiting for the result and handling potential exceptions
try {
    int result = futureResult.get();
    System.out.println("The result is: " + result);
} catch (InterruptedException | ExecutionException e) {
    e.printStackTrace();
}

executor.shutdown();
```

In this example, `Future` provides a way to get results from asynchronous tasks without blocking the main thread. It also supports cancelling tasks and checking their state.

??x
The `Future` interface allows you to handle asynchronous computations effectively by providing methods like `get()` which blocks until the result is ready or throws an exception if something goes wrong, while `cancel()` lets you stop a task that hasn't started or is still in progress.

In this example, we submit a task to an executor service and use `Future` to retrieve its result asynchronously. If the task completes normally, it prints the result; otherwise, it handles exceptions.
x??

---

#### CompletableFuture
Background context explaining `CompletableFuture`, its constructor methods, and how to complete or cancel tasks.

:p What is `CompletableFuture` in Java?
??x
`CompletableFuture` is a powerful class for asynchronous programming that extends the basic `Future` interface. It allows you to perform complex operations asynchronously by chaining various operations using lambda expressions and method references.

Here’s an example of creating and completing a `CompletableFuture`:

```java
CompletableFuture<Integer> cf = new CompletableFuture<>();
// Simulating work being done elsewhere
cf.complete(42); // Completing the future with a value

try {
    int result = cf.get(); // Blocks until the result is available
    System.out.println("Result: " + result);
} catch (InterruptedException | ExecutionException e) {
    e.printStackTrace();
}
```

`CompletableFuture` provides many methods for chaining operations:

- `thenApply`: Applies a function to the completed future.
- `thenAccept`: Accepts a consumer that will be called with the value once it’s available.

Here is an example of using `thenApply` and `thenRun`:

```java
CompletableFuture.supplyAsync(() -> 42)
    .thenApply(v -> v * 2) // Doubles the value
    .thenAccept(System.out::println); // Prints the result

// Using thenRun for side effects
CompletableFuture<Void> future = CompletableFuture.completedFuture(null);
future.thenRun(() -> System.out.println("Task completed"));
```

`CompletableFuture` is very flexible and allows you to compose complex asynchronous tasks with ease.

??x
`CompletableFuture` is a versatile class that enhances the `Future` interface by providing additional methods for composing asynchronous operations. It supports chaining operations like `thenApply`, `thenAccept`, and `thenRun`.

In this example, we create a `CompletableFuture` using `supplyAsync` to simulate a task that returns an integer value 42. We then chain further operations: doubling the value with `thenApply` and printing it with `thenAccept`. The `thenRun` method is used for side effects.

??x
The flexibility of `CompletableFuture` allows you to build complex asynchronous workflows by chaining various operations using lambda expressions or method references. This makes it easier to manage tasks that have multiple steps, where each step depends on the result of the previous one.
x??

---

#### CompletableFuture Simple Usage
Background context: `CompletableFuture` is a way to handle asynchronous tasks and their results in Java. It provides methods like `thenApply`, `thenAccept`, etc., for chaining operations based on task completion.

Example code:
```java
class CompletableFutureSimple {
    static String twice(String x) { return x + ' ' + x; }
    
    public static void main(String[] args) {
        CompletableFuture<String> cf = new CompletableFuture<>();
        
        // Chain the operations to be executed after the future is completed
        cf.thenApply(x -> twice(x))
          .thenAccept(x -> System.out.println(x));
        
        // Complete the future with a value
        cf.complete("Hello");
    }
}
```
:p What does `cf.thenApply` and `cf.thenAccept` do in this example?
??x
- `thenApply`: This method applies a function to the result of the completed `CompletableFuture`. It returns another `CompletableFuture`.
- `thenAccept`: This method accepts the result of the completed `CompletableFuture`, but does not return anything. It is used for side-effect operations.

Code explanation:
```java
// Then apply the twice function on the result and get a new CompletableFuture<String>
cf.thenApply(x -> twice(x))
  // Then accept (print) the result without returning anything
  .thenAccept(x -> System.out.println(x));
```
x??

---
#### Sprite Class in Animator Program
Background context: The `Sprite` class is used to represent an animated image that moves around a graphical display. It uses threads to update its position and repaint itself continuously.

:p What is the purpose of the `Sprite` class?
??x
- The `Sprite` class represents a moving image on the screen. It extends `Component`, which allows it to be painted, and implements `Runnable` so that it can run in a separate thread.

Code explanation:
```java
public class Sprite extends Component implements Runnable {
    protected static int spriteNumber = 0;
    // Other fields...
    
    public Sprite(Component parent, Image image, Direction direction) {
        this.parent = parent;
        this.image = image;
        this.direction = direction;
        this.number = Sprite.spriteNumber++;
        setSize(image.getWidth(this), image.getHeight(this));
    }
    
    // Other methods...
}
```
x??

---
#### Bounce Class in Animator Program
Background context: The `Bounce` class is the main program that manages a collection of `Sprite` objects and controls their animation.

:p What does the `Bounce` class do?
??x
- The `Bounce` class creates a graphical display with multiple moving images (Sprites). It uses a thread pool to handle the animations concurrently. Each `Sprite` runs in its own thread, allowing for smooth animation of the images across the screen.

Code explanation:
```java
public class Bounce extends JPanel {
    protected JPanel p;
    protected Image img;
    protected ExecutorService tp = Executors.newCachedThreadPool();
    
    public Bounce(String imgName) {
        setLayout(new BorderLayout());
        JButton b = new JButton("Add a Sprite");
        // Add action listener to add Sprites
        b.addActionListener(e -> {
            System.out.println("Creating another one.");
            Sprite s = new Sprite(this, img);
            tp.execute(s);
            p.add(s);
            v.add(s);
        });
        
        add(b, BorderLayout.NORTH);
        add(p = new JPanel(), BorderLayout.CENTER);
        // Other initialization code...
    }
    
    public void stop() {
        // Stop all Sprites
    }
}
```
x??

---
#### Volatile Keyword Usage in Sprite Class
Background context: The `volatile` keyword is used to ensure that variables are always read from the main memory and not cached by the compiler. This is crucial for multi-threaded programs where multiple threads might modify a variable.

:p What is the purpose of using the `volatile` keyword in the `Sprite` class?
??x
- Using `volatile` ensures that changes made to a variable by one thread are immediately visible to other threads, preventing inconsistent state due to caching.

Code example:
```java
protected volatile boolean done = false;
```
Explanation: The `done` field is declared as `volatile`. This means any changes made to `done` in one thread will be seen by all other threads without needing explicit synchronization.

x??

---
#### ThreadPool and ExecutorService in Bounce Class
Background context: A `ThreadPool` or `ExecutorService` manages a pool of worker threads that can execute tasks. In the `Bounce` class, a cached thread pool is used to handle multiple `Sprite` instances concurrently.

:p How does the `Bounce` class use an `ExecutorService`?
??x
- The `Bounce` class uses an `ExecutorService` (specifically a cached thread pool) to manage and execute tasks (in this case, running `Sprite` objects in separate threads). This allows multiple animations to run concurrently without creating new threads every time.

Code example:
```java
protected ExecutorService tp = Executors.newCachedThreadPool();
```
Explanation: The `ExecutorService` is created using `Executors.newCachedThreadPool()`. When a `Sprite` instance is added, it is executed by the thread pool, allowing smooth and parallel animation.

x??

---

#### Stopping a Thread: Using `stop()` Method is Deprecated
Background context explaining why using the `Thread.stop()` method is deprecated. The compiler generates deprecation warnings, and it can lead to unreliable behavior when used in multi-threaded programs.

:p What are the reasons for deprecating the use of `Thread.stop()`?
??x
The `Thread.stop()` method is deprecated because its usage can cause issues such as abrupt termination of threads without proper cleanup. This can result in resource leaks or inconsistent program states, making it unreliable to use in multi-threaded applications.

```java
public class StopBoolean {
    // volatile boolean done = false; // Note: The given code has 'volatile' removed for brevity.
    protected boolean done = false;
    Runnable r = () -> {
        while (done) { 
            System.out.println("StopBoolean running");
            try {
                Thread.sleep(720);
            } catch (InterruptedException ex) {
                // nothing to do
            }
        }
        System.out.println("StopBoolean finished.");
    };
    public void shutDown() {
        System.out.println("Shutting down...");
        done = true;
    }
    public void doDemo() throws InterruptedException {
        ExecutorService pool = Executors.newSingleThreadExecutor();
        pool.submit(r);
        Thread.sleep(1000 * 5);
        shutDown();
        pool.shutdown();
        pool.awaitTermination(2, TimeUnit.SECONDS);
    }
}
```
x??

---

#### Stopping a Thread: Using `volatile` Boolean Variable
Background context explaining the role of the `volatile` keyword in ensuring visibility of changes across threads. The main loop continuously checks this boolean variable to determine if the thread should exit.

:p How does using a `volatile` boolean variable help in stopping a thread?
??x
Using a `volatile` boolean variable ensures that any write operation to this variable is immediately visible to all other threads. This allows the main loop of the run() method to check the value of the boolean variable and exit the loop when necessary, thereby gracefully shutting down the thread.

```java
public class StopBoolean {
    // Must be volatile to ensure changes visible to other threads.
    protected volatile boolean done = false;
    
    Runnable r = () -> { 
        while (done) { 
            System.out.println("StopBoolean running");
            try { 
                Thread.sleep(720);
            } catch (InterruptedException ex) {
                // nothing to do
            }
        } 
        System.out.println("StopBoolean finished.");
    };
    
    public void shutDown() {
        System.out.println("Shutting down...");
        done = true;
    }
    
    public void doDemo() throws InterruptedException { 
        ExecutorService pool = Executors.newSingleThreadExecutor();
        pool.submit(r);
        Thread.sleep(1000 * 5);
        shutDown();
        pool.shutdown();
        pool.awaitTermination(2, TimeUnit.SECONDS);
    }
}
```
x??

---

#### Stopping a Thread: Handling Sockets and Deadlocks
Background context explaining scenarios where threads might get stuck in I/O operations. The example demonstrates intentionally creating a deadlock by reading from an HTTP socket without sending a proper request.

:p How can you terminate a thread that is blocked on a socket operation?
??x
To handle a thread that is blocked on a socket operation, you can close the socket. This interrupts the read operation, allowing the thread to exit the loop and terminate properly. However, closing the socket does not guarantee immediate termination of the I/O operation; it sends an interrupt signal.

```java
public class StopClose extends Thread {
    protected Socket io;
    
    public void run() { 
        try { 
            io = new Socket("java.sun.com", 80); // HTTP 
            BufferedReader is = new BufferedReader(new InputStreamReader(io.getInputStream()));
            System.out.println("StopClose reading");
            
            // The following line will deadlock (intentionally), since HTTP
            // enjoins the client to send a request (like "GET / HTTP/1.0") and a null line,
            // before reading the response.
            String line = is.readLine(); // DEADLOCK 
        } catch (IOException ex) { 
            System.out.println("StopClose terminating: " + ex);
        }
    }
    
    public void shutDown() throws IOException { 
        if (io != null) {
            synchronized (io) { 
                io.close();
            }
        }
        System.out.println("StopClose.shutDown() completed");
    }
}
```
x??

---

#### Stopping a Thread: Interrupting Reads
Background context explaining the use of `java.io.InterruptedIOException` to handle interrupted reads and retry logic. This approach is useful when you want to break out of blocking I/O operations without fully terminating the socket connection.

:p How can you interrupt a read operation in Java?
??x
To interrupt a read operation, you can catch an `InterruptedException` which is thrown by methods like `read()`. By throwing an interruption signal, the thread can exit the read loop and retry the read operation later. This approach allows for graceful handling of I/O operations without fully terminating the socket connection.

```java
public class Intr {
    public void handleInterruptedRead() throws IOException {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(socket.getInputStream()))) {
            String line;
            while ((line = br.readLine()) != null) {
                // Process line
            }
        } catch (InterruptedException ex) {
            System.out.println("Read interrupted: " + ex);
            // Retry or handle the interruption
        }
    }
}
```
x??

---

