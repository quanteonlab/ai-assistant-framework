# Flashcards: 2B005---Streaming-Systems_processed (Part 12)

**Starting Chapter:** Variations on Fixed Windows

---

#### Custom Windowing Overview
Background context explaining the concept. Custom windowing allows users to define their own windowing strategies beyond fixed, sliding, and session windows. This flexibility can be crucial for handling specific business requirements that don't fit into predefined window types.

The objective is to understand how custom windowing works in Beam, which includes defining both initial window assignment and optional merging logic.
:p What is the main purpose of custom windowing in Beam?
??x
Custom windowing allows users to define their own windowing strategies beyond the standard fixed, sliding, and session windows. This flexibility is essential for handling specific business requirements that don't fit into predefined window types.
x??

---
#### Fixed Window Assignment Logic
Background context: The stock fixed-window implementation places each element into an appropriate fixed-window based on its timestamp and the window’s size and offset parameters.

:p What is the process of assigning elements to a fixed window?
??x
The process involves determining which fixed window an element belongs to by comparing its timestamp with the start and end timestamps defined by the window's size and offset. For example, if you have a fixed window with a size of 10 minutes and an offset of 5 minutes, each element is assigned to the window that starts at 5 minutes before its timestamp.
```java
public class FixedWindowAssignment {
    private final Duration windowSize;
    private final Duration offset;

    public Window assign(WindowedValue<?> value) {
        Instant timestamp = value.timestamp();
        // Calculate start and end of the window
        Instant start = timestamp.minus(offset);
        Instant end = start.plus(windowSize);

        return new Window(start, end);
    }
}
```
x??

---
#### Session Windows with Retractions
Background context: Session windows are a type of custom window that merges adjacent time intervals where there is no activity. These windows emit values when they begin and retract them when the session ends or times out.

:p How do session windows handle late data?
??x
Session windows handle late data by emitting retractions for previously emitted windows when new events arrive. This ensures that only the most recent state of the windowed data is retained.
```java
public class SessionWindowHandling {
    public void processLateData(WindowedValue<?> value) {
        // Get the current session window for this element
        Window currentWindow = getCurrentSessionWindow(value.timestamp());

        // Emit retractions for any windows that should be ended due to late data
        emitRetractions(currentWindow);

        // Emit the new state of the window with the late event
        emitNewState(value);
    }

    private void emitRetractions(Window window) {
        // Logic to retracted old session window states
    }

    private void emitNewState(WindowedValue<?> value) {
        // Logic to emit the updated state of the current session window
    }
}
```
x??

---
#### Custom Windowing with Merging
Background context: In Beam, custom windowing can include both initial window assignment and optional merging logic. This allows windows to evolve over time, as seen in session windows.

:p What is the role of merging in custom windowing?
??x
Merging in custom windowing allows windows to combine or evolve over time based on certain conditions. For example, in a session window, adjacent inactive intervals merge into a single active interval.
```java
public class CustomWindowMerging {
    public void mergeWindows(Window currentWindow, Window otherWindow) {
        if (currentWindow.isAdjacentTo(otherWindow)) {
            // Merge the windows if they are adjacent and have no activity in between
            currentWindow.merge(otherWindow);
        }
    }
}
```
x??

---
#### Fixed Windows vs. Custom Windows
Background context: While fixed windows follow a straightforward logic of assigning elements to predefined time intervals, custom windowing allows more flexibility by defining unique assignment and merging rules.

:p How does fixed windowing differ from custom windowing?
??x
Fixed windows place each element into an appropriate fixed-window based on its timestamp and the window’s size and offset parameters. In contrast, custom windowing involves defining both initial window assignment and optional merging logic to handle more complex scenarios that don't fit into predefined window types.
```java
public class CustomWindowExample {
    public Window assign(WindowedValue<?> value) {
        // Custom logic for assigning elements to windows
    }

    public void mergeWindows(Window currentWindow, Window otherWindow) {
        if (currentWindow.isAdjacentTo(otherWindow)) {
            // Merge the windows if they are adjacent and have no activity in between
            currentWindow.merge(otherWindow);
        }
    }
}
```
x??

---
#### Stock Implementation of Fixed Windows in Beam
Background context: The stock fixed-window implementation in Beam is straightforward, focusing on placing elements into appropriate time intervals based on their timestamps.

:p How does the stock fixed window implementation work?
??x
The stock fixed-window implementation assigns each element to a fixed window based on its timestamp and predefined window size and offset parameters. This allows for easy partitioning of data into regular time intervals.
```java
public class FixedWindowAssignment {
    private final Duration windowSize;
    private final Duration offset;

    public Window assign(WindowedValue<?> value) {
        Instant timestamp = value.timestamp();
        // Calculate start and end of the window
        Instant start = timestamp.minus(offset);
        Instant end = start.plus(windowSize);

        return new Window(start, end);
    }
}
```
x??

---

---
#### Aligned Fixed Windows
Background context: In a typical fixed-windows implementation, windows are aligned across all data points. For instance, if you have a team with window intervals from noon to 1 PM and another team also has window intervals from noon to 1 PM, these windows align perfectly. This alignment is beneficial for comparing like windows between different dimensions, such as between teams.

However, this alignment comes with a subtle cost: during the end of each window (e.g., 1 PM), all active windows across multiple keys become complete simultaneously. Consequently, the system faces a massive load of window materialization at regular intervals, which can cause synchronization issues and bursty workloads.

:p What is the issue associated with aligned fixed windows?
??x
The issue is that during the end of each window, all active windows for different keys (e.g., teams) become complete simultaneously. This leads to a synchronized burst of window materialization events in the system.
```java
// Example of how this might be handled internally
public Collection<IntervalWindow> assignWindow(AssignContext c) {
    long start = c.timestamp().getMillis() - c.timestamp()
            .plus(size)
            .minus(offset)
            .getMillis();
    return Arrays.asList(new IntervalWindow(new Instant(start), size));
}
```
x??

---
#### Load Burstiness in Fixed Windows
Background context: The load burstiness mentioned refers to the fact that, in a fixed-windows implementation with multiple keys (like teams), all windows for different keys become complete at the same time. This leads to a sudden influx of window materialization tasks during each window's end, which can overwhelm the system.

This is especially problematic when there are many keys involved, potentially causing significant load spikes every window ends.

:p How does load burstiness affect the system in fixed windows?
??x
Load burstiness affects the system by causing it to handle a large number of window materialization tasks simultaneously at the end of each window. This can lead to performance bottlenecks and resource contention, particularly when dealing with a high number of keys.

For example, if you have 1000 teams, all their windows might become complete at the same time, causing a massive load spike every two minutes.
```java
// Example of how this might be handled internally during watermark processing
public PCollection<KV<Team, Integer>> processWatermark(PCollection<KV<Team, Integer>> input) {
    return input.apply(Window.into(FixedWindows.of(TWO_MINUTES))
            .triggering(AfterWatermark())
            .discardingFiredPanes())
            .apply(Sum.integersPerKey());
}
```
x??

---
#### Window Alignment for Comparison
Background context: Fixed windows are often used to compare data across different dimensions, such as between teams or in time series analysis. The alignment of these windows ensures that the comparison is meaningful and consistent.

For instance, if you have a dataset with multiple teams, each team's window from noon to 1 PM will align perfectly, allowing for easy comparison at this specific interval.

:p Why are aligned fixed windows useful for comparisons?
??x
Aligned fixed windows are useful for comparisons because they ensure that data across different dimensions (e.g., teams) is grouped into the same intervals. This alignment allows you to compare like windows between different keys directly, ensuring consistency in your analysis.
```java
// Example of windowing strategy with aligned fixed windows
public class FixedWindows extends WindowFn<Object, IntervalWindow> {
    private final Duration size;
    private final Duration offset;

    public Collection<IntervalWindow> assignWindow(AssignContext c) {
        long start = c.timestamp().getMillis() - c.timestamp()
                .plus(size)
                .minus(offset)
                .getMillis();
        return Arrays.asList(new IntervalWindow(new Instant(start), size));
    }
}
```
x??

---

#### Unaligned Fixed Windows Implementation
Background context: The system aims to make load more predictable by allowing unaligned fixed windows. This can reduce provisioning requirements for handling peak loads, but it comes at a cost of reduced ability to compare across keys.

Code changes involve modifying the default fixed-windowing strategy to support unaligned windows. The key idea is that while elements with the same key have aligned windows, elements with different keys will typically have unaligned windows.

:p What is the primary change made in the `UnalignedFixedWindows` implementation?
??x
The primary change involves adjusting the window start time based on a hash function of the key and current timestamp. The `assignWindow` method calculates the start of the interval window by incorporating the key's hash value, ensuring that windows for elements with the same key are aligned.

```java
public class UnalignedFixedWindows extends WindowFn<KV<K, V>, IntervalWindow> {
    private final Duration size;
    private final Duration offset;

    public Collection<IntervalWindow> assignWindow(AssignContext c) {
        long perKeyShift = (hash(c.element().key()) % 0);
        long start = perKeyShift + c.timestamp().getMillis()
                     - c.timestamp().plus(size).minus(offset);
        return Arrays.asList(IntervalWindow.of(new Instant(start), size));
    }
}
```
x??

---
#### Pipeline Using Unaligned Fixed Windows
Background context: To implement the new windowing strategy, the pipeline needs to be updated. This involves changing the existing fixed windows configuration to `UnalignedFixedWindows`.

:p How does one switch a pipeline from default fixed windows to unaligned fixed windows?
??x
To switch the pipeline, you would use the `Window.into()` method with the custom `UnalignedFixedWindows` implementation and apply triggers as needed. For example:

```java
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.into(UnalignedFixedWindows.of(TWO_MINUTES)))
    .triggering(AfterWatermark())
    .apply(Sum.integersPerKey());
```

This configuration ensures that windows for elements with the same key are aligned but may be unaligned across different keys. This approach can help in reducing peak resource provisioning requirements.

x??

---
#### Visualization of Unaligned Fixed Windows
Background context: The implementation allows for better load distribution by spreading window completion tasks out, which is beneficial when processing large datasets efficiently. However, this comes at the cost of reduced comparability between windows from different keys.

:p How does using unaligned fixed windows affect the load pattern compared to traditional aligned fixed windows?
??x
Using unaligned fixed windows results in a more even distribution of window completion tasks over time. This is because elements with different keys may have windows that start at different times, reducing the peak load on the system during window processing.

For example:
- With traditional aligned fixed windows: Multiple panes might be emitted simultaneously for different keys.
- With unaligned fixed windows: Panes from different keys arrive more evenly spaced out in time.

This can significantly reduce the need for high provisioning to handle peak loads, making it easier to manage resources efficiently.

x??

---
#### Per-Element/Key Fixed Windows
Background context: This variation on fixed windows is specifically tailored to the data being processed. It's an advanced feature that Cloud Dataflow supports and was initially adopted by early users of the service.

:p What is the key characteristic of per-element/key fixed windows?
??x
The key characteristic of per-element/key fixed windows is that they are designed to be more closely aligned with the specific patterns or characteristics in the data being processed. This means that each element or key can have its own customized windowing strategy, potentially providing a better fit for the underlying data distribution and processing requirements.

This approach allows for fine-grained control over how different elements are grouped and processed within windows, optimizing performance based on the nature of the input data.

x??

---

#### Custom Window Sizes per Customer

Background context: This concept explains how a company generates analytics data for its customers, where each customer can configure their own window size for aggregating metrics. The challenge arises when supporting arbitrary fixed windows versus predefined options.

:p How does the company support different window sizes per customer?
??x
The solution involves modifying the `FixedWindows` implementation to use per-element window sizes based on metadata in each record. By doing this, the pipeline can dynamically assign intervals according to the specific needs of each customer.

The modified code changes how windows are assigned by utilizing the `HasWindowSize` interface and adjusting the timestamp calculation accordingly:

```java
public class PerElementFixedWindows<T extends HasWindowSize> 
    extends WindowFn<T, IntervalWindow> {
    private final Duration offset;

    public Collection<IntervalWindow> assignWindow(AssignContext c) { 
        long perElementSize = c.element().getWindowSize(); 
        long start = perKeyShift + c.timestamp().getMillis() 
            - c.timestamp()
            .plus(perElementSize)
            .minus(offset)
            .getMillis();
        
        return Arrays.asList(new IntervalWindow(
            new Instant(start), perElementSize));
    }
}
```
x??

---

#### Pipeline Code for Custom Window Sizes

Background context: After implementing the custom `PerElementFixedWindows` class, integrating it into the pipeline code is straightforward. The example shows how to apply this windowing strategy and aggregate data accordingly.

:p How does the pipeline code handle custom per-element fixed windows?
??x
The pipeline code uses the new `PerElementFixedWindows` implementation by specifying a duration (like 2 minutes) and applying it with a watermark trigger mechanism. This ensures that elements are processed within their designated window sizes as defined in the metadata.

Example pipeline code:

```java
PCollection<KV<Team, Integer>> totals = input 
    .apply(Window.into(PerElementFixedWindows.of(TWO_MINUTES)))
    .triggering(AfterWatermark())
    .apply(Sum.integersPerKey());
```

In this example:
- `input` is the original PCollection.
- `Window.into(...)` specifies the custom window function.
- `AfterWatermark()` is used as the trigger strategy.
- `Sum.integersPerKey()` aggregates values per key.

This setup allows processing elements with different window sizes based on metadata, making the pipeline flexible to customer-specific requirements.
x??

---

#### Visual Representation of Custom Window Sizes

Background context: The visual representation in Figure 4-10 illustrates how keys (e.g., A and B) have distinct window sizes. Key A uses a two-minute window, while Key B uses a one-minute window.

:p How does the visualization demonstrate custom window sizes?
??x
The figure shows that different elements grouped under various keys are processed within windows of varying durations as dictated by metadata associated with each element. For instance, all elements for Key A fall into 2-minute windows, whereas those for Key B fit into 1-minute windows.

This visual depiction highlights the dynamic nature of window sizes, allowing for flexible analytics tailored to specific customer needs.
x??

---

#### Flexibility in Windowing

Background context: This section emphasizes the importance of custom windowing strategies over fixed options. It explains why built-in APIs often do not cater to such use cases but underscores the value of flexibility they offer.

:p Why is custom windowing powerful for this scenario?
??x
Custom windowing is powerful because it allows each customer to define their own window sizes based on specific requirements, rather than being constrained by predefined options. This flexibility enables more accurate and relevant analytics tailored to individual customer needs without requiring significant changes to the underlying system architecture.

While implementing such custom solutions can be complex, they provide a scalable approach that can handle a wide range of windowing scenarios.
x??

#### Session Windows and Custom Windowing

Session windowing is a type of windowing that groups elements into sessions based on their timestamps, with each session potentially lasting until there is no activity for a certain period (gap duration). The implementation involves assigning windows to individual elements and then merging overlapping or adjacent windows during the grouping phase.

:p What are the key steps in implementing session windowing?
??x
The key steps in implementing session windowing include:

1. **Assignment**: Each element is initially placed into a proto-session window that begins at its timestamp and extends for the gap duration.
2. **Merging**: During the grouping phase, all eligible windows are sorted, and any overlapping or adjacent windows are merged together.

Here's an abbreviated implementation of session windowing in Java:

```java
public class Sessions extends WindowFn<Object, IntervalWindow> {
    private final Duration gapDuration;

    public Collection<IntervalWindow> assignWindows(AssignContext c) {
        return Arrays.asList(new IntervalWindow(c.timestamp(), gapDuration));
    }

    public void mergeWindows(MergeContext c) throws Exception {
        List<IntervalWindow> sortedWindows = new ArrayList<>();
        for (IntervalWindow window : c.windows()) {
            sortedWindows.add(window);
        }
        Collections.sort(sortedWindows);

        List<MergeCandidate> merges = new ArrayList<>();
        MergeCandidate current = new MergeCandidate();
        for (IntervalWindow window : sortedWindows) {
            if (current.intersects(window)) {
                current.add(window);
            } else {
                merges.add(current);
                current = new MergeCandidate(window);
            }
        }
        merges.add(current);

        for (MergeCandidate merge : merges) {
            merge.apply(c);
        }
    }

    private class IntervalWindow {
        private final Instant start;
        private final Duration duration;

        public IntervalWindow(Instant start, Duration duration) {
            this.start = start;
            this.duration = duration;
        }

        public Instant getStart() { return start; }
        public Instant getEnd() { return start.plus(duration); }
    }

    private class MergeCandidate implements Comparable<MergeCandidate> {
        private final List<IntervalWindow> windows;

        public MergeCandidate(IntervalWindow window) {
            this.windows = new ArrayList<>();
            add(window);
        }

        public void add(IntervalWindow window) {
            // Add logic to merge or keep the window
        }

        public boolean intersects(IntervalWindow other) {
            return !other.getStart().isAfter(this.getEnd()) && !this.getStart().isAfter(other.getEnd());
        }

        public IntervalWindow union() {
            // Return a merged interval window if applicable
            return null;
        }

        @Override
        public int compareTo(MergeCandidate o) {
            return windows.size() - o.windows.size();
        }
    }
}
```

x??

---

#### Bounded Sessions

Bounded sessions are a variant of session windows where the size of each session is limited, either in time or element count. This can be useful for semantic reasons or to prevent spam.

:p How does bounded session windowing differ from regular session windowing?
??x
Bounded session windowing differs from regular session windowing by imposing limits on the total duration and/or number of elements in each session. This is achieved through additional logic that checks if a merged window exceeds these predefined bounds, thus triggering the merging process earlier.

Here's an example implementation of bounded sessions:

```java
public class BoundedSessions extends WindowFn<Object, IntervalWindow> {
    private final Duration gapDuration;
    private final Duration maxSize;

    public Collection<IntervalWindow> assignWindows(AssignContext c) {
        return Arrays.asList(new IntervalWindow(c.timestamp(), gapDuration));
    }

    public static void mergeWindows(WindowFn<?, IntervalWindow>.MergeContext c) throws Exception {
        List<IntervalWindow> sortedWindows = new ArrayList<>();
        for (IntervalWindow window : c.windows()) {
            sortedWindows.add(window);
        }
        Collections.sort(sortedWindows);

        List<MergeCandidate> merges = new ArrayList<>();
        MergeCandidate current = new MergeCandidate();
        for (IntervalWindow window : sortedWindows) {
            MergeCandidate next = new MergeCandidate(window);
            if (current.intersects(window)) {
                current.add(window);
                if (windowSize(current.union()) <= (maxSize - gapDuration)) continue;
                // Current window exceeds bounds, so flush and move to next
                next = new MergeCandidate();
            }
            merges.add(current);
            current = next;
        }
        merges.add(current);

        for (MergeCandidate merge : merges) {
            merge.apply(c);
        }
    }

    private static Duration windowSize(IntervalWindow union) {
        return union == null ? Duration.ZERO : Duration.between(union.start(), union.end());
    }

    private class MergeCandidate implements Comparable<MergeCandidate> {
        private final List<IntervalWindow> windows;

        public MergeCandidate(IntervalWindow window) {
            this.windows = new ArrayList<>();
            add(window);
        }

        public void add(IntervalWindow window) {
            // Add logic to merge or keep the window
        }

        public boolean intersects(IntervalWindow other) {
            return !other.getStart().isAfter(this.getEnd()) && !this.getStart().isAfter(other.getEnd());
        }

        public IntervalWindow union() {
            // Return a merged interval window if applicable
            return null;
        }

        @Override
        public int compareTo(MergeCandidate o) {
            return windows.size() - o.windows.size();
        }
    }
}
```

x??

---

#### Custom Windowing for Bounded Sessions

Custom windowing allows users to implement their own specific logic, tailored to unique use cases. For bounded sessions, this involves adding custom checks and merging conditions.

:p How does customizing windowing help in implementing bounded sessions?
??x
Customizing windowing helps in implementing bounded sessions by allowing the user to define precise limits on session sizes (e.g., time duration or element count). This flexibility is crucial because different use cases may require varying definitions of what constitutes a valid session.

For instance, custom logic can ensure that no single session exceeds a certain size limit. If a session does exceed this limit, it gets split into smaller sessions to meet the bounded criteria.

Here's an example of how you might customize windowing for bounded sessions:

```java
// Custom logic in mergeWindows method
List<IntervalWindow> sortedWindows = new ArrayList<>();
for (IntervalWindow window : c.windows()) {
    sortedWindows.add(window);
}
Collections.sort(sortedWindows);

List<MergeCandidate> merges = new ArrayList<>();
MergeCandidate current = new MergeCandidate();
for (IntervalWindow window : sortedWindows) {
    MergeCandidate next = new MergeCandidate(window);
    if (current.intersects(window)) {
        current.add(window);
        if (windowSize(current.union()) <= (maxSize - gapDuration)) continue;
        // Current window exceeds bounds, so flush and move to next
        next = new MergeCandidate();
    }
    merges.add(current);
    current = next;
}
merges.add(current);

for (MergeCandidate merge : merges) {
    merge.apply(c);
}

private static Duration windowSize(IntervalWindow union) {
    return union == null ? Duration.ZERO : Duration.between(union.start(), union.end());
}
```

x??

---

#### Processing-Time Windows

Background context: In stream processing, windows are used to group data for processing. While event-time windowing is based on when events occur, processing-time windows focus on timestamps at which data arrives or processes.

:p What are processing-time windows and how do they differ from event-time windowing?
??x
Processing-time windows group incoming elements into fixed-size time bins based on the timestamps recorded by the system. Unlike event-time windowing, which waits for all events to occur before processing them, processing-time windows start aggregating data immediately as new elements arrive.

For example:
- A 5-minute processing-time window would aggregate data arriving in a stream every 5 minutes.
- The formula for the watermark is not necessary here since it's based on the system clock rather than event timestamps.

```java
public class ProcessingTimeWindowExample {
    // This method processes elements grouped by their arrival time
    public void processWindowElements(List<Element> elements) {
        // Logic to aggregate or process the elements
    }
}
```
x??

---

#### Session Windows

Background context: Session windows are a type of dynamic window that merges adjacent events based on a session timeout. The system automatically groups elements into sessions, making it easier to handle complex scenarios without custom logic.

:p What is a session window and how does it work?
??x
A session window dynamically merges consecutive events that fall within a specified gap (session timeout) between them. When the gap exceeds the defined threshold, the current session ends, and a new one starts for subsequent events. This approach simplifies event grouping without requiring explicit logic.

For example:
- If the session timeout is 5 minutes, any two events arriving less than 5 minutes apart are considered part of the same session.
```java
public class SessionWindowExample {
    private long sessionTimeout = 5 * 60 * 1000; // 5 minutes in milliseconds

    public void processSessionWindow(List<Element> elements) {
        boolean isNewSession = true;
        for (Element element : elements) {
            if (isNewSession || isWithinGap(element.getTimestamp())) {
                // Process the session
            } else {
                isNewSession = true; // Start a new session
            }
        }
    }

    private boolean isWithinGap(long timestamp) {
        return System.currentTimeMillis() - timestamp < sessionTimeout;
    }
}
```
x??

---

#### Custom Windows

Background context: While many systems offer predefined window types, custom windows allow users to define more complex groupings based on specific requirements. This flexibility is crucial for handling diverse and intricate data processing needs.

:p What are custom windows and why are they important?
??x
Custom windows enable developers to create tailored grouping strategies that fit the unique needs of their applications. These can include unaligned fixed windows, per-element fixed windows, and bounded session windows, among others. Custom windows provide more control over how data is aggregated, making it easier to meet specific business requirements.

For example:
- Unaligned fixed windows ensure a consistent output distribution.
- Per-element fixed windows allow for dynamic window sizes based on element attributes.
```java
public class CustomWindowExample {
    public void applyCustomWindow(String key, List<Element> elements) {
        // Implement custom logic to define and process the window
        int windowSize = determineWindowSize(key);
        for (int i = 0; i < elements.size(); i += windowSize) {
            processWindow(elements.subList(i, Math.min(i + windowSize, elements.size())));
        }
    }

    private int determineWindowSize(String key) {
        // Logic to determine the appropriate window size based on 'key'
        return 10; // Simplified example
    }
}
```
x??

---

#### Unaligned Fixed Windows

Background context: Unaligned fixed windows group data into regular-sized bins, but unlike aligned windows, they don't align with natural boundaries and can provide a more even distribution of output over time.

:p What are unaligned fixed windows?
??x
Unaligned fixed windows divide the stream into equal-sized segments regardless of event timestamps. This approach ensures that each window has a consistent size, which is useful for maintaining a steady rate of processing or when watermark triggers need to be managed carefully.

For example:
- A 1-minute unaligned fixed window would process data in chunks of one minute, even if the first element arrives at a random time.
```java
public class UnalignedFixedWindowExample {
    private int windowSize = 60 * 1000; // 1 minute in milliseconds

    public void applyUnalignedWindow(List<Element> elements) {
        long currentTime = System.currentTimeMillis();
        for (Element element : elements) {
            if ((currentTime - element.getTimestamp()) <= windowSize) {
                processWindow(element);
            }
        }
    }

    private void processWindow(Element element) {
        // Process the element in the current window
    }
}
```
x??

---

#### Per-Element Fixed Windows

Background context: Per-element fixed windows allow for dynamic sizing of fixed windows based on specific attributes of individual elements. This can be useful for handling data with varying characteristics, such as per-user or per-ad-campaign processing.

:p What are per-element fixed windows?
??x
Per-element fixed windows dynamically determine the size of each window based on a specific attribute in the element, providing greater flexibility and customization to the pipeline's semantics according to the use case.

For example:
- A system might process user interactions with different time windows depending on the user ID.
```java
public class PerElementFixedWindowExample {
    public void applyPerElementWindow(List<Element> elements) {
        for (Element element : elements) {
            int windowSize = determineWindowSize(element.getAttribute("userId"));
            // Process 'element' in a window of size 'windowSize'
        }
    }

    private int determineWindowSize(String userId) {
        // Logic to determine the appropriate window size based on 'userId'
        return 10; // Simplified example
    }
}
```
x??

---

#### Bounded Session Windows

Background context: Bounded session windows limit the duration of a session, ensuring that sessions do not grow indefinitely. This is useful for scenarios like spam detection or setting upper bounds on processing latency.

:p What are bounded session windows?
??x
Bounded session windows ensure that each session does not exceed a specified maximum length. This can help manage resource usage and prevent long-running sessions from dominating the pipeline, especially in cases where spam or delayed processing needs to be controlled.

For example:
- A system might limit a user's session to 15 minutes for tracking activities.
```java
public class BoundedSessionWindowExample {
    private int maxSessionDuration = 15 * 60 * 1000; // 15 minutes in milliseconds

    public void applyBoundedSessionWindow(List<Element> elements) {
        long currentTime = System.currentTimeMillis();
        for (Element element : elements) {
            if ((currentTime - element.getTimestamp()) <= maxSessionDuration) {
                processSession(element);
            } else {
                startNewSession(element);
            }
        }
    }

    private void processSession(Element element) {
        // Process 'element' in the current session
    }

    private void startNewSession(Element element) {
        // Start a new session for 'element'
    }
}
```
x??
---

