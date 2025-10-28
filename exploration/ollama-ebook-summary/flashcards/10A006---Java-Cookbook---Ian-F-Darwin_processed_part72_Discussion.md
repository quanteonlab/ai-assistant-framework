# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 72)

**Starting Chapter:** Discussion

---

#### Timer Service for One-Shot Tasks
Background context: The `Timer` service from `java.util` can be used to schedule tasks that need to be executed at a specific future time. This is useful for one-time, non-recurring tasks.

Relevant code example:
```java
public class ReminderService {
    /** The Timer object */
    Timer timer = new Timer();

    // ... other methods

    protected void loadReminders() throws Exception {
        Files.lines(Path.of("ReminderService.txt")).forEach(aLine -> {
            ParsePosition pp = new ParsePosition(0);
            Date date = formatter.parse(aLine, pp);
            String task = aLine.substring(pp.getIndex());
            if (date == null) {
                System.out.println("Invalid date in " + aLine);
                return;
            }
            timer.schedule(new Item(task), date); // Schedule the item
        });
    }

    class Item extends TimerTask {
        String message;

        Item(String m) { message = m; }

        public void run() {
            message(message);
        }
    }
}
```

:p How does the `Timer` service from `java.util` work for scheduling one-shot tasks?
??x
The `Timer` service schedules a task to be executed at a specific future time, as defined by a `Date` object. The example code reads reminders from a file and schedules each reminder using a custom `Item` class that extends `TimerTask`. When the scheduled time arrives, the `run()` method of the `Item` class is called.
```java
// Example: Scheduling a task to run at 2023-12-25 10:30
Date date = formatter.parse("2023 12 25 10 30");
timer.schedule(new Item("Reminder on high noon"), date);
```
x??

---

#### Timer for Recurring Tasks
Background context: For recurring tasks, you can use a background thread that sleeps in a loop or use the `Timer` service and recompute the next time the task should run.

Relevant code example:
```java
public class AutoSave extends Thread {
    /** The FileSaver interface is implemented by the main class. */
    protected FileSaver model;

    /** How long to sleep between tries */
    public static final int MINUTES = 5;
    private static final int SECONDS = MINUTES * 60;

    public AutoSave(FileSaver m) {
        super("AutoSave Thread");
        setDaemon(true); // so we don't keep the main app alive
        model = m;
    }

    public void run() {
        while (true) { // entire run method runs forever.
            try {
                sleep(SECONDS * 1000);
            } catch (InterruptedException e) {
                // do nothing with it
            }
            if (model.wantAutoSave() && model.hasUnsavedChanges()) {
                // Save the file
            }
        }
    }
}
```

:p How can a background thread be used for recurring tasks?
??x
A background thread is created to handle recurring tasks by sleeping in a loop and periodically checking conditions. This approach ensures that the main application remains responsive while performing background operations.
```java
// Example: Auto-saving every 5 minutes if changes are detected
AutoSave autoSave = new AutoSave(model);
autoSave.start();
```
x??

---

#### Complex Scheduling with Quartz or EJB Timer Service
Background context: For more complex scheduling tasks, such as running something at high noon every second Thursday, consider using a third-party scheduling library like Quartz or the EJB Timer Service in JavaEE/Jakarta.

Relevant code example:
```java
// Example using Quartz (pseudocode)
Scheduler scheduler = new StdSchedulerFactory().getScheduler();
JobDetail job = JobBuilder.newJob(MyComplexTask.class)
    .withIdentity("myTask", "group1").build();

CronScheduleBuilder schedule = CronScheduleBuilder.cronSchedule("0 0 * ? THU#2 *");
Trigger trigger = TriggerBuilder.newTrigger()
    .withIdentity("myTrigger", "group1")
    .startNow()
    .withSchedule(schedule)
    .build();

scheduler.scheduleJob(job, trigger);
scheduler.start();
```

:p How can a third-party library like Quartz be used for complex scheduling tasks?
??x
A third-party library like Quartz allows you to define more complex schedules using cron expressions. For example, the code snippet demonstrates how to schedule a job to run at high noon every second Thursday of the month.
```java
// Example: Schedule MyComplexTask to run on 12:00 PM every second Thursday
JobDetail job = JobBuilder.newJob(MyComplexTask.class)
    .withIdentity("myTask", "group1").build();

CronScheduleBuilder schedule = CronScheduleBuilder.cronSchedule("0 0 * ? THU#2 *");
Trigger trigger = TriggerBuilder.newTrigger()
    .withIdentity("myTrigger", "group1")
    .startNow()
    .withSchedule(schedule)
    .build();

scheduler.scheduleJob(job, trigger);
```
x??

---

#### AutoSave Mechanism Overview
AutoSave is a mechanism that saves a model's data automatically at regular intervals to prevent loss of work. This implementation uses a background thread that sleeps for five minutes and checks if an auto-save or explicit save is necessary.

:p Describe the key components of the AutoSave mechanism.
??x
The key components include:
1. A background thread that sleeps for 300 seconds (five minutes).
2. Methods to load, check, and save the model's data.
3. Synchronization mechanisms to handle concurrent access and ensure data integrity.

Code snippet illustrating the sleep interval:

```java
try {
    Thread.sleep(300 * 1000); // Sleep for 5 minutes (300 seconds)
} catch (InterruptedException e) {
    Thread.currentThread().interrupt();
}
```

x??

---
#### Synchronization Requirement

In the provided context, all methods in the `FileSaver` interface must be synchronized to ensure thread safety. This is necessary because multiple threads may access these methods concurrently.

:p Why are all methods in the FileSaver interface synchronized?
??x
All methods need synchronization to prevent race conditions and ensure data consistency when accessed by multiple threads simultaneously. For instance, if `saveFile()` is not synchronized, it could lead to issues such as corrupted files or inconsistent states if two threads try to save at the same time.

Example of a synchronized method:

```java
public class FileSaverImpl implements FileSaver {
    private final Object lock = new Object();

    @Override
    public void loadFile(String fn) {
        synchronized (lock) {
            // Method implementation
        }
    }

    @Override
    public boolean wantAutoSave() {
        synchronized (lock) {
            // Method implementation
        }
    }

    @Override
    public boolean hasUnsavedChanges() {
        synchronized (lock) {
            // Method implementation
        }
    }

    @Override
    public void saveFile(String fn) {
        synchronized (lock) {
            // Method implementation
        }
    }
}
```

x??

---
#### Synchronization Object in Shutdown Process

The text mentions that the method to shut down the main program must be synchronized on the same object used by `saveFile()`. This ensures that all related operations are properly coordinated.

:p What is the importance of synchronizing shutdown methods with save methods?
??x
Synchronizing shutdown methods with save methods (using the same lock object) ensures that critical cleanup and resource management processes happen in a controlled manner. It prevents race conditions where the program might shut down before data is fully saved, which could lead to data loss or corruption.

Example of synchronized shutdown method:

```java
public void safeShutdown() {
    synchronized (lock) { // Using the same lock object as saveFile()
        // Method implementation for cleanup and saving
    }
}
```

x??

---
#### Strategy for Saving Data

The text suggests that it would be smarter to save data to a recovery file, similar to how better word processors handle autosaves. This approach can provide an extra layer of protection against data loss.

:p How does the suggested strategy improve upon regular autosaving?
??x
Saving data to a recovery file provides an additional backup in case the primary auto-save process fails or if the system crashes. If the main save operation encounters issues, there is still a chance that the data can be recovered from the recovery file. This approach enhances reliability and helps prevent accidental loss of work.

Example of saving to a recovery file:

```java
public void safeSave(String fn) {
    synchronized (lock) {
        // Save logic for the current model's data
        try (FileOutputStream fos = new FileOutputStream("recovery_file.dat")) {
            // Code to write to recovery file
        } catch (IOException e) {
            System.err.println("Failed to save to recovery file.");
        }
    }
}
```

x??

---

---
#### Dynamic Class Loading
Background context: The `java.lang.Class` and `java.lang.reflect` packages allow for dynamic loading of classes at runtime. This capability is crucial for applications that need to load or manipulate classes on-the-fly, such as Java web services or applets.

:p What does the concept of dynamic class loading enable in Java?
??x
Dynamic class loading enables programs to load and use classes that are not available at compile-time. It allows for flexibility by dynamically adding new functionalities during runtime without requiring a restart.
x??

---
#### Reflecting on Class Information
Background context: The `java.lang.reflect` package provides mechanisms to inspect the structure of any class, including its methods, fields, constructors, and more.

:p How can you use reflection to gather information about a class?
??x
Reflection allows you to query and introspect classes at runtime. You can obtain `Class` objects for types, check if they implement certain interfaces, and access their fields, methods, and constructors.
```java
// Example: Getting the Class object from an instance of a class
MyClass myInstance = new MyClass();
Class<?> clazz = myInstance.getClass();

// Checking if the class implements an interface
boolean implementsInterface = clazz.getInterfaces().length > 0;

// Accessing fields
Field[] fields = clazz.getDeclaredFields();

// Invoking methods
Method method = clazz.getMethod("myMethod", paramTypes...);
Object result = method.invoke(myInstance, args...);
```
x??

---
#### Invoking Methods Dynamically
Background context: Once you have a `Method` object from reflection, you can invoke the method dynamically on an instance of the class.

:p How do you invoke a method using reflection?
??x
To invoke a method using reflection, you first get a `Method` reference and then use its `invoke` method to execute it. The method requires the target object and any parameters needed by the method.
```java
// Example: Invoking a method on an instance of MyClass
Method method = MyClass.class.getMethod("myMethod", int.class);
Object result = method.invoke(new MyClass(), 123);
```
x??

---
#### Creating Classes Dynamically
Background context: The `ClassLoader` is responsible for loading classes into the JVM. In some cases, you might need to create a class on the fly and load it dynamically using a custom class loader.

:p How can you create a class from scratch at runtime?
??x
Creating a class dynamically involves creating a byte array with the compiled bytecode of the new class and then loading this byte array into the JVM via a `ClassLoader`. Here is an example:
```java
// Pseudocode for creating a dynamic class using a ClassLoader
byte[] classData = compileMyClass(); // Compiles MyClass to a byte array
Class<?> clazz = defineClass("MyClass", classData, 0, classData.length);
```
x??

---
#### Conclusion on Reflection and Dynamic Loading
Background context: The `java.lang.Class` and `java.lang.reflect` packages offer powerful tools for introspection and dynamic behavior in Java. These capabilities are essential for building flexible applications like web services or applets that can adapt to changing requirements.

:p What is the significance of reflection and dynamic loading in modern Java programming?
??x
Reflection and dynamic loading provide flexibility by allowing programs to inspect and manipulate classes at runtime, enabling features such as hot swapping, pluggable components, and dynamic behavior. These capabilities enhance the runtime capability and extensibility of Java applications.
x??

---

