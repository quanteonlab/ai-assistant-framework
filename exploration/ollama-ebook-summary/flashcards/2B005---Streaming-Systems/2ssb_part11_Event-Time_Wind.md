# Flashcards: 2B005---Streaming-Systems_processed (Part 11)

**Starting Chapter:** Event-Time Windowing

---

#### Observation Order and Coloring
Background context explaining the difference between observation order and coloring. The text mentions two different observation orders, one colored white representing "as we've seen all along," and the other colored purple with a shifted processing-time axis as shown in Figure 4-1.

:p Describe the two observation orders mentioned in the text.
??x
The two observation orders are described as follows:
- One order is colored white and represents the usual sequence of observations.
- The second order is colored purple, indicating a shift in the processing-time axis. This shift simulates a different set of distributed systems operating under slightly altered conditions (e.g., winds blowing from the east instead of the west).

The key point here is that even though the processing time is shifted, the underlying values and event times remain constant.
x??

---

#### Event-Time Windowing
Explanation of fixed windowing in event time. The text mentions using a heuristic watermark to compare with different observation orderings.

:p What does event-time windowing ensure when comparing two different processing-time orderings?
??x
Event-time windowing ensures that the final results for the four windows remain the same (14, 18, 3, and 12) despite the different orders of observations in processing time. This is because event-time windowing focuses on the sequence of events based on their timestamps rather than their arrival times.

The important takeaway is that the shape of the outputs may differ due to the shift in processing time, but the final results for each window are consistent.
x??

---

#### Processing-Time Windowing via Triggers
Explanation of processing-time methods using triggers. The text introduces comparing event-time with a processing-time method using triggers.

:p How does processing-time windowing via triggers compare to event-time windowing?
??x
Processing-time windowing via triggers involves setting specific conditions or "triggers" that determine when windows are closed and results are finalized, whereas event-time windowing focuses on the sequence of events based on their timestamps. Triggers can be used to implement more complex logic for determining window boundaries.

For example, in a system where late data is handled using triggers, you might set up conditions like "if an event hasn't arrived within 5 minutes, close the current window and open a new one."
x??

---

#### Example Code for Processing-Time Windowing
Explanation of how to implement processing-time windowing with triggers. This example uses C/Java-like pseudocode.

:p Provide an example of implementing processing-time windowing using triggers in code.
??x
```java
public class TriggerBasedWindow {
    private int watermark = 0; // Initial watermark value
    private List<Integer> buffer = new ArrayList<>(); // Buffer to hold events

    public void addEvent(int value) {
        buffer.add(value); // Add event to buffer
    }

    public int[] processWatermark() {
        if (System.currentTimeMillis() - watermark > 5000) { // Check if watermark is overdue
            int sum = buffer.stream().mapToInt(Integer::intValue).sum(); // Sum events in buffer
            System.out.println("Window closed, sum: " + sum); // Print result
            buffer.clear(); // Clear buffer for new window
            watermark = System.currentTimeMillis(); // Update watermark
        }
        return new int[]{}; // Return results if needed
    }
}
```
This pseudocode demonstrates how to implement a simple processing-time window using triggers. The `processWatermark` method checks every 5 seconds (or when the watermark times out) and processes all buffered events, then clears the buffer for the next window.
x??

---

#### Windowing via Event-Time Panes
Background context: In this approach, we emulate processing-time windows using event-time panes. This method allows for a more flexible handling of time-based computations on streaming data.

:p How does emulating processing-time windows with event-time panes work?
??x
Emulating processing-time windows with event-time panes involves defining global event-time windows and periodically triggering these windows based on the desired size of the processing-time windows. Each pane acts like an independent processing-time window, ensuring that results are computed independently for each time interval.

Example code:
```java
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.triggering(Repeatedly(AlignedDelay(ONE_MINUTE)))
             .discardingFiredPanes())
    .apply(Sum.integersPerKey());
```
x??

---

#### Processing-Time Windowing via Triggers
Background context: This method involves using triggers to define processing-time windows over event-time data. The key idea is to use a trigger mechanism that fires based on the arrival of new elements, allowing for accurate and timely computations.

:p How does processing-time windowing via triggers work?
??x
Processing-time windowing via triggers works by setting up a trigger that periodically checks if any new events have arrived within the defined time frame. Once the watermark passes the end of the window, the trigger fires, and the elements are processed.

Example code:
```java
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.info(FixedWindows.of(TWO_MINUTES)))
    .apply(Sum.integersPerKey());
```
x??

---

#### Windowing via Ingress Time
Background context: This approach involves mapping the event times of incoming data to their ingress time (processing time at arrival). By doing so, we can effectively emulate processing-time windows on top of a streaming system.

:p How does windowing via ingress time work?
??x
Windowing via ingress time works by overwriting the event times of input elements with their ingress time. This allows us to use standard event-time fixed windows and triggers based on these new timestamps, which are effectively the processing times at arrival.

Example code:
```java
PCollection<String> raw = IO.read()
    .apply(ParDo.of(new DoFn<String, String>() {
        public void processElement(ProcessContext c) {
            c.outputWithTimestamp(new Instant());
        }
    }));

PCollection<KV<Team, Integer>> totals = input
    .apply(Window.info(FixedWindows.of(TWO_MINUTES)))
    .apply(Sum.integersPerKey());
```
x??

---

---

#### Session Windows Overview
Session windows are a type of event-time window that captures periods of activity terminated by a gap of inactivity. They are particularly useful for analyzing user activities over specific time frames where correlation within sessions is needed.

:p What are session windows used for?
??x
Session windows are used to analyze data based on periods of activity, often correlating the events within these periods to understand user engagement or behavior patterns.
x??

---

#### Data-Driven Windows
Session windows exemplify a data-driven window approach where the location and size of the windows depend directly on the input data. This is different from fixed or sliding windows which are based on predefined time intervals.

:p What makes session windows unique compared to fixed or sliding windows?
??x
Session windows are unique because their boundaries (start and end) are determined by patterns in the incoming data, specifically by periods of activity followed by gaps of inactivity. Fixed and sliding windows have pre-defined sizes that do not change based on the input.
x??

---

#### Unaligned Windows
Unlike aligned windows which apply uniformly across all data, session windows are unaligned as they only apply to specific subsets of the data, such as per user sessions.

:p How does a session window differ from an aligned window?
??x
A session window differs from an aligned window in that it is applied to specific subsets of data (like per-user activities) rather than uniformly across all input. Aligned windows like fixed or sliding apply the same window size and boundaries consistently.
x??

---

#### Constructing Sessions with Out-of-Order Data
When dealing with out-of-order data, session windows can still be constructed by merging together overlapping proto-sessions within a predefined gap duration timeout.

:p How are sessions created from out-of-order data?
??x
Sessions are created from out-of-order data by identifying overlapping proto-sessions that fall within the gap duration timeout. These proto-sessions are merged to form larger, continuous session windows as new data arrives.
x??

---

#### Example Code for Session Windows
The following code snippet shows how to set up and use session windows in a PCollection with a one-minute gap duration timeout.

:p What does this code do?
??x
This code sets up a `PCollection` that uses session windows with a one-minute gap duration timeout. It triggers firings both early (aligned delay) and late (after count), ensuring robust handling of incoming data.
```java
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.into(Sessions.withGapDuration(ONE_MINUTE)))
    .triggering(
        AfterWatermark().withEarlyFirings(AlignedDelay(ONE_MINUTE))
                         .withLateFirings(AfterCount(1)))
    .apply(Sum.integersPerKey());
```
x??

---

#### Merging Proto-Sessions into Sessions
Proto-sessions are initially created for each record, with overlaps merged to form complete sessions. The key is that the final session is a composition of smaller overlapping windows separated by inactivity gaps no longer than the timeout.

:p How does merging proto-sessions work?
??x
Merging proto-sessions works by first creating individual proto-sessions for each incoming event and then combining those that overlap within the allowed gap duration. This process continues as new events arrive, ensuring that complete sessions are formed over time.
x??

---

#### Early/Future Firings in Session Windows
Session windows handle early firings using an aligned delay based on the session timeout, while late firings occur after a count of one event, allowing partial results to be materialized even if not all data has arrived.

:p What triggers early and late firings in session windows?
??x
Early firings in session windows are triggered by an aligned delay that matches the session timeout. Late firings occur after a count of one event, enabling partial or speculative materialization before all data for the session is complete.
x??

---

#### Materializing Sessions on Streaming Engines
On a streaming engine, sessions are materialized as complete when the watermark passes the end of the final proto-session window, ensuring that only full and accurate results are produced.

:p How are sessions materialized in a streaming environment?
??x
In a streaming environment, sessions are materialized once the watermark passes the end of the last proto-session window. This ensures that only complete and accurate session results are outputted, even as new data continuously arrives.
x??

---

