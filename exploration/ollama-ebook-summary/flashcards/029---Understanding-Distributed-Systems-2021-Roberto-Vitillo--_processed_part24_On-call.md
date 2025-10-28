# Flashcards: 029---Understanding-Distributed-Systems-2021-Roberto-Vitillo--_processed (Part 24)

**Starting Chapter:** On-call

---

#### Importance of Service-Specific Dashboards
Service-specific dashboards display implementation details that require a deep understanding of the service's inner workings. These dashboards are primarily used by the team that owns the service and provide an initial entry point into debugging issues. Beyond service-specific metrics, they should also include upstream and downstream dependency metrics.
:p What is the primary purpose of a service-specific dashboard?
??x
The primary purpose of a service-specific dashboard is to serve as a first entry point for debugging by displaying implementation details that require deep understanding. It includes both service-specific metrics and dependencies like load balancers, messaging queues, and data stores.
x??

---

#### Best Practices for Service-Specific Dashboards
Dashboards need to be updated regularly as new metrics are added or old ones removed. To maintain consistency across different environments (staging, production), it's best to define dashboards with a domain-specific language and version-control them like code. This ensures that updates can be made through the same pull request containing related changes.
:p How should charts and dashboards be managed when new metrics are added or old ones removed?
??x
Charts and dashboards should be managed by updating them through the same pull request that contains related code changes, using a domain-specific language. This approach helps in maintaining consistency across environments like staging and production without manual updates, which can be error-prone.
x??

---

#### Placing Important Charts at the Top
The most important charts should always be located at the very top of the dashboard for easy access. Charts should render with a default timezone (e.g., UTC) to facilitate communication between people in different parts of the world. All charts in the same dashboard should use consistent timeresolution and range.
:p Where should the most important charts be placed on a service-specific dashboard?
??x
The most important charts should always be located at the very top of the dashboard for easy access and quick reference.
x??

---

#### Time Range and Resolution Selection
Selecting an appropriate time range and resolution depends on the common use case. For ongoing incidents, a 1-hour range with 1-minute resolution is recommended. For capacity planning, a 1-year range with 1-day resolution might be more suitable.
:p How should one select the default time range and resolution for a dashboard?
??x
Select the default time range and resolution based on the most common use case for the dashboard. For ongoing incidents, a 1-hour range with 1-minute resolution is recommended. For capacity planning, a 1-year range with 1-day resolution might be more suitable.
x??

---

#### Reducing Data Points on Charts
Reducing the number of data points and metrics per chart to a minimum helps in faster downloading times and easier interpretation. Metrics with similar ranges (min and max values) should be grouped together for clarity, while those with vastly different ranges should be split into separate charts.
:p How can you manage the number of data points on a chart?
??x
To manage the number of data points, keep them to a minimum to facilitate faster downloading times and easier interpretation. Group metrics with similar ranges (min and max values) together for clarity, while those with vastly different ranges should be split into separate charts.
x??

---

#### Annotations in Charts
Charts should contain useful annotations such as descriptions linking to runbooks, alert thresholds, and deployment markers. Metrics that are only emitted when error conditions occur can show wide gaps between data points, making interpretation difficult. Emitting a value of zero or one for these metrics helps avoid confusion.
:p What kind of annotations should be included in charts?
??x
Annotations should include descriptions linking to runbooks, alert thresholds, and deployment markers. For metrics only emitted when error conditions occur, emit values of zero (in absence) or one (presence) to avoid gaps that might confuse operators.
x??

---

#### On-Call Rotation Best Practices
On-call rotations are effective when developers are responsible for operating the services they build, reducing operational overhead. Developers should be given free reign to improve on-call experiences and focus on mitigating incidents rather than fixing root causes immediately. Alerts should link to relevant dashboards and runbooks with actionable steps.
:p What is an important practice for on-call rotations?
??x
An important practice for on-call rotations is ensuring that developers, who are familiar with the system's architecture and issues, handle them. They should be given free reign to improve their on-call experience by revising dashboards or improving resilience mechanisms. Alerts should link to relevant dashboards and runbooks with actionable steps.
x??

---

#### Incident Management
When an alert triggers, it should immediately link to relevant dashboards and a runbook listing actions. All actions taken by the operator should be communicated in a shared channel like a global chat accessible by other teams. The first step is to mitigate the incident rather than fix root causes.
:p What steps should be followed when an alert triggers?
??x
When an alert triggers, it should immediately link to relevant dashboards and a runbook listing actions. All actions taken by the operator should be communicated in a shared channel like a global chat accessible by other teams. The first step is to mitigate the incident rather than fix root causes.
x??

---

#### Post-Incident Analysis
Post-incident analysis, or postmortem, aims to understand the root cause and prevent future occurrences. If an SLO's error budget is significantly impacted, there should be a dedicated team effort to restore reliability until on-call rotations are back to normal.
:p What is the purpose of conducting a post-mortem after an incident?
??x
The purpose of conducting a post-mortem after an incident is to understand the root cause and develop a set of repair items to prevent future occurrences. If an SLO's error budget is significantly impacted, the whole team should focus on reliability until on-call rotations are restored.
x??

---

#### Observability Overview
Observability is a set of tools that provide granular insights into a system's behavior, enabling operators to understand and debug complex emergent failures. These tools help minimize time-to-validate hypotheses by providing rich contextual information.

:p What are the core components of an observability platform?
??x
The core components include telemetry sources like metrics, event logs, and traces. Metrics provide high-throughput monitoring data, while event logs and traces handle high-dimensional data well.
x??

---

#### Telemetry Sources
Metrics, event logs, and traces form the foundation of observability. Each has unique strengths: metrics are used for monitoring health, whereas logs and traces aid in debugging.

:p What are some common types of telemetry sources?
??x
Common types include:
- Metrics: Time-series data stored in high-throughput stores.
- Event Logs: Immutable lists of time-stamped events that can be structured or free-form text.
- Traces: Detailed sequences of interactions, often used for debugging complex failures.
x??

---

#### Metrics
Metrics are crucial for monitoring the health and performance of a system. They provide aggregated data over time but struggle with high-dimensional data.

:p What is an example scenario where metrics would be useful?
??x
An example scenario where metrics would be useful is tracking the overall response time and request count of a service to identify trends or anomalies in its performance.
x??

---

#### Event Logs
Event logs capture detailed information about system events, which can help trace root causes and investigate long-tail behaviors.

:p How do event logs differ from metrics?
??x
Event logs provide granular insights with rich context, whereas metrics offer high-level aggregated data. Metrics are better for monitoring overall health, while logs are essential for debugging specific issues.
x??

---

#### Logs in Detail
Logs are immutable lists of time-stamped events that can be structured or free-form text. They are used extensively for debugging and investigating long-tail behaviors.

:p What are the advantages and disadvantages of using textual logs?
??x
Advantages:
- Simple to emit, especially free-form ones.
Disadvantages:
- High noise-to-signal ratio due to their fine-grained nature.
- Can introduce overhead if not managed properly (e.g., asynchronous logging).
- Potential issues if disk fills up with excessive logging.

Example of emitting a structured log in Java:
```java
public class LogEmitter {
    public void logEvent(String failureCount, String serviceRegion) {
        // Emitting a structured log
        System.out.println("{ \"failureCount\" : " + failureCount + ", \"serviceRegion\" : \"" + serviceRegion + "\" }");
    }
}
```
x??

---

#### Best Practices for Logging
Best practices include emitting logs in a single event per work unit, containing all relevant information about the operation performed.

:p How can developers ensure that logs are useful for debugging?
??x
Developers should:
- Store data about specific work units in a single log event.
- Include context and measurements (e.g., timestamps, operation times).
- Instrument every network call to log response status codes and times.
- Sanitize sensitive information before logging.

Example of sanitizing logs:
```java
public class SafeLogger {
    public void safeLog(String userInput) {
        // Sanitizing potentially sensitive data
        String sanitizedUserInput = sanitize(userInput);
        System.out.println("{ \"userInput\" : \"" + sanitizedUserInput + "\" }");
    }

    private String sanitize(String input) {
        return input.replaceAll("[^a-zA-Z0-9]", "");
    }
}
```
x??

---

#### Logging Levels and Cost Control
Background context: To understand why a remote call failed, it's important to have detailed logs. However, logging everything can be costly. Different logging levels like debug, info, warning, error help control the amount of log data generated.

If applicable, add code examples with explanations:
```java
// Example of setting logging level in Java using Log4j2
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class LoggerExample {
    private static final Logger logger = LogManager.getLogger(LoggerExample.class);

    public void someMethod() {
        // Setting log level to debug for investigation purposes
        logger.debug("Debugging this message");
    }
}
```
:p What is the purpose of having different logging levels?
??x
The purpose of having different logging levels (debug, info, warning, error) is to allow operators to increase logging verbosity for investigation purposes and reduce costs when granular logs are not needed. This helps in managing the volume of log data generated.
x??

---
#### Sampling for Cost Control
Background context: Logging all events can be expensive, so sampling techniques like only logging every nth event or prioritizing events based on their signal-to-noise ratio can help reduce verbosity.

:p What is an example of a sampling technique to reduce the verbosity of logs?
??x
An example of a sampling technique is logging only one out of every n events. For instance, if you set n=10, your service would log every 10th event.
x??

---
#### Log Collectors and Rate Limiting
Background context: As nodes scale out, the volume of logs increases. To avoid overwhelming the logging pipeline, log collectors need to be able to rate-limit requests. If using a third-party service for ingesting, storing, and querying logs, there might already be a quota in place.

:p How can you prevent your logging pipeline from being overwhelmed?
??x
To prevent your logging pipeline from being overwhelmed, log collectors should be capable of rate-limiting requests. This ensures that the volume of data does not exceed the system's capacity. If using a third-party service, check if there is already a quota in place.
x??

---
#### Tracing Requests in Distributed Systems
Background context: Tracing captures the entire lifespan of a request as it propagates through services in a distributed system. A trace represents causally related spans that show the execution flow of a request.

If applicable, add code examples with explanations:
```java
// Example of tracing in Java using OpenTracing
import io.opentracing.Span;
import io.opentracing.Tracer;

public class TracingExample {
    private static final Tracer tracer = ...; // Initialize tracer

    public void processRequest() {
        Span span1 = tracer.buildSpan("Operation1").start();
        try (Scope scope = tracer.activateSpan(span1)) {
            // Simulate operation
            doSomething();
        } finally {
            span1.finish();
        }
    }

    private void doSomething() {
        Span span2 = tracer.buildSpan("Operation2").start();
        try (Scope scope = tracer.activateSpan(span2)) {
            // Simulate another operation
            ...
        } finally {
            span2.finish();
        }
    }
}
```
:p What is a trace in the context of distributed systems?
??x
A trace in the context of distributed systems is a list of causally related spans that represent the execution flow of a request. Each span represents an interval of time mapping to a logical operation or work unit, and it contains key-value pairs.
x??

---
#### Identifying Bottlenecks with Traces
Background context: Traces help identify bottlenecks in the end-to-end request path by providing detailed information about each step taken during a request.

:p How can traces be used to identify bottlenecks?
??x
Traces can be used to identify bottlenecks in the end-to-end request path. By examining the execution flow captured in spans, developers can pinpoint where delays or issues occur. This is useful for debugging both frequent and rare issues.
x??

---
#### Resource Attribution with Traces
Background context: Traces also help attribute which clients hit which downstream services and in what proportion, which can be used for rate-limiting or billing purposes.

:p How can traces assist in attributing resource usage?
??x
Traces can assist in attributing resource usage by identifying which clients hit which downstream services and the frequency of these hits. This information is valuable for rate-limiting services based on client behavior or for implementing billing mechanisms.
x??

---

