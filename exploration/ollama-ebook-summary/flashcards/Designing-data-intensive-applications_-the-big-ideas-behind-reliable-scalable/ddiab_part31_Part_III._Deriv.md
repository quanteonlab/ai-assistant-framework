# Flashcards: Designing-data-intensive-applications_-the-big-ideas-behind-reliable-scalable_processed (Part 31)

**Starting Chapter:** Part III. Derived Data

---

#### Systems of Record vs. Derived Data
In distributed systems, data can be managed and processed by different types of data systems. A system of record is the authoritative version of your data where new data is first written and each fact is represented exactly once (typically normalized). Derived data systems transform or process existing data from a system of record to serve specific needs such as caching, indexing, materialized views, or predictive summary data.
:p How do you distinguish between a system of record and derived data?
??x
A system of record holds the authoritative version of your data where new data is written. Derived data systems are transformations or processing of existing data from a system of record to serve specific needs like caching, indexing, or analytics. The key difference lies in their purpose: a system of record is the source of truth, while derived data supports performance optimization and flexibility.
x??

---
#### Batch-Oriented Dataflow Systems (e.g., MapReduce)
Batch-oriented dataflow systems are designed for processing large-scale datasets where tasks are divided into smaller jobs that can be executed independently. Examples include MapReduce, which processes data in parallel stages: map, shuffle, sort, and reduce. This approach is well-suited for scenarios requiring high computational power and scalability.
:p What is an example of a batch-oriented dataflow system?
??x
An example of a batch-oriented dataflow system is MapReduce. It involves dividing the processing tasks into smaller jobs that can be executed in parallel stages: map, shuffle, sort, and reduce.
x??

---
#### Data Streams
Data streams are continuous flows of data that need to be processed with low latency. Unlike batch systems, stream processing deals with real-time data where each piece of data is processed as it arrives. This approach is ideal for applications requiring immediate responses or analytics on live data.
:p How does data stream processing differ from batch processing?
??x
Data streams process continuous flows of data in real-time, while batch processing handles large datasets that can be divided into smaller jobs to run in parallel. The key difference lies in latency: batch processing is suited for scenarios requiring high computational power and scalability but may have higher latency, whereas stream processing deals with immediate responses.
x??

---
#### Reliability and Scalability in Future Applications
The final chapter explores ideas on building reliable, scalable, and maintainable applications using the tools and principles discussed throughout the book. It emphasizes the importance of clear dataflow management, robust error handling, and efficient resource utilization to ensure application resilience and performance.
:p What are some key aspects covered in the final chapter?
??x
The final chapter covers key aspects such as clear dataflow management, robust error handling, and efficient resource utilization to build reliable, scalable, and maintainable applications. These principles ensure that applications can handle large-scale data processing efficiently while maintaining high availability and performance.
x??

---
#### Coherent Application Architecture with Multiple Data Systems
In complex applications, integrating multiple data systems (e.g., databases, caches, indexes) is crucial for meeting diverse access patterns and performance requirements. This involves understanding the dataflow between different components of the system to ensure seamless integration and efficient data processing.
:p Why is it important to integrate multiple data systems in a large application?
??x
Integrating multiple data systems (e.g., databases, caches, indexes) is crucial because they can serve different access patterns and performance requirements. By understanding the dataflow between these components, you ensure seamless integration and efficient data processing, leading to better overall system performance.
x??

---
#### Redundancy in Derived Data
Derived data is considered redundant because it duplicates existing information but provides benefits such as improved read performance through denormalization or caching. However, maintaining consistency between derived data and the source of truth (system of record) is essential for avoiding discrepancies.
:p Why is derived data often redundant?
??x
Derived data is redundant because it duplicates existing information from a system of record to optimize read performance. For example, caches, indexes, and materialized views store transformed or processed versions of the original data. However, maintaining consistency with the source of truth (system of record) ensures that any discrepancies are resolved.
x??

---
#### Clear Distinction Between Systems of Record and Derived Data
Making a clear distinction between systems of record and derived data in system architecture can provide clarity on dataflow and dependencies. This helps manage complexity by defining inputs, outputs, and their relationships explicitly.
:p Why is it important to make a clear distinction between systems of record and derived data?
??x
It's important to make a clear distinction because it clarifies the dataflow through your system, making explicit which parts have specific inputs and outputs and how they depend on each other. This distinction helps manage complexity in large applications by defining relationships and dependencies more clearly.
x??

---

#### Batch Processing Systems Overview
Batch processing systems take a large amount of input data, run jobs to process it, and produce some output data. These systems are often scheduled to run periodically (e.g., daily) rather than being triggered by user requests.

:p What is the primary difference between batch processing systems and online systems?
??x
Batch processing systems operate on a predefined set of inputs and do not typically have users waiting for real-time responses, unlike online systems which handle individual client requests. Instead, they are designed to process large volumes of data in batches over extended periods.
x??

---
#### Performance Measures of Batch Jobs
The primary performance measure of batch jobs is throughput, defined as the time it takes to crunch through an input dataset of a certain size.

:p What is the key metric for evaluating the performance of batch processing systems?
??x
Throughput is the key metric. It measures how efficiently and quickly a batch job can process large datasets.
x??

---
#### MapReduce Overview
MapReduce, introduced in 2004, is a programming model used for batch processing tasks that involves splitting input data into chunks, mapping those chunks to keys and values, and then reducing them.

:p What does the MapReduce framework do?
??x
MapReduce splits input data into chunks, applies a map function to transform each chunk, and then reduces the output of these maps. This allows for parallel processing of large datasets.
x??

---
#### Example Pseudocode for MapReduce
```pseudocode
function map(key, value) {
    // Generate intermediate key-value pairs from the input data
}

function reduce(key, values) {
    // Aggregate or process all the values associated with a single key
}
```

:p Provide an example of pseudocode for the Map and Reduce functions in MapReduce.
??x
Here is an example pseudocode:

```pseudocode
// Example: Counting word frequencies
function map(key, value) {
    words = split(value)
    for each word in words {
        emit(word, 1)
    }
}

function reduce(key, values) {
    sum = 0
    for each value in values {
        sum += value
    }
    emit(key, sum)
}
```

This example counts the frequency of each word in a text file.
x??

---
#### Historical Context: Punch Card Machines
Before programmable digital computers, punch card tabulating machines were used to compute aggregate statistics from large inputs.

:p What was one early form of batch processing used before digital computers?
??x
Punch card tabulating machines like the Hollerith machines used in the 1890 US Census were an early form of batch processing. These machines processed data by reading punched cards and computing aggregates, similar to modern batch processing systems.
x??

---
#### MapReduce's Role in Modern Data Systems
MapReduce was a significant step forward for achieving massive scalability on commodity hardware.

:p Why is MapReduce considered important in the context of modern data systems?
??x
MapReduce is important because it enabled the efficient and scalable processing of large datasets using commodity hardware. It provided a framework that allowed complex computations to be executed across many machines, making it suitable for big data applications.
x??

---

#### Simple Log Analysis
Background context explaining how to analyze web server logs using Unix tools. The example provided shows a log line and its fields, with an explanation of what each field represents.

:p How can you use `awk`, `sort`, `uniq`, and `head` commands in combination to find the five most popular pages on a website?
??x
To find the five most popular pages, the following pipeline can be used:

```sh
cat /var/log/nginx/access.log | \
awk '{print $7}' | \  # Extract the requested URL from each log line
sort | \              # Sort the URLs alphabetically
uniq -c | \           # Count the occurrences of each URL
sort -r -n | \        # Sort by count in reverse numerical order
head -n 5            # Take the top five lines
```

This pipeline processes each log file line, extracts and prints the requested URL (`$7`), sorts them, counts unique URLs with their respective frequencies, sorts these frequencies in descending order, and finally outputs the top five most popular pages.

??x
The answer explains how to use a series of Unix commands to analyze log files. The `awk` command is used to extract the seventh field (requested URL) from each line. Then, `sort` is used to alphabetically sort these URLs. The `uniq -c` command counts the number of occurrences for each unique URL and prints them with their respective frequencies. Finally, `sort -r -n | head -n 5` sorts these counts in descending order and outputs only the top five.

```sh
cat /var/log/nginx/access.log | \
awk '{print $7}' | \  # Extract URLs
sort | \             # Alphabetical sort
uniq -c | \          # Count occurrences
sort -r -n | \       # Sort by count in descending order
head -n 5           # Output top five lines
```
x??

---

#### Chain of Commands versus Custom Program
Background context explaining the comparison between using a chain of Unix commands and writing a custom program for log analysis. The example provided shows how to achieve the same task with Ruby.

:p What is the difference in approach when comparing a chain of Unix commands to a custom Ruby script for analyzing web server logs?
??x
The difference lies in execution flow, memory usage, and performance:

- **Chain of Unix Commands:** This approach uses `awk`, `sort`, `uniq`, and `head` to process log files. It is concise and leverages the power of Unix tools.
  
- **Custom Ruby Script:** The script processes each line individually, building an in-memory hash table to count occurrences. It sorts the hash table contents and prints the top five entries.

:p How does a custom Ruby script differ from a chain of Unix commands for log analysis?
??x
The custom Ruby script uses an in-memory hash table to keep track of URL counts:

```ruby
counts = Hash.new(0)  # Initialize count dictionary

File.open('/var/log/nginx/access.log') do |file|
  file.each do |line| 
    url = line.split[6]  # Extract the URL from the log line
    counts[url] += 1     # Increment the count for this URL
  end
end

top5 = counts.map{|url, count| [count, url] }.sort.reverse[0...5]
top5.each{|count, url| puts "#{count} #{url}" }  # Output top five URLs
```

This script reads each log line, extracts the URL (7th field), and increments its count in memory. It then sorts this hash table to find the top five most frequent URLs.

??x
The answer explains that a custom Ruby script processes each line individually, builds an in-memory hash table for counting URLs, and finally sorts these counts to get the top five entries. This approach is more verbose but provides flexibility in handling data.

```ruby
counts = Hash.new(0)  # Initialize count dictionary

File.open('/var/log/nginx/access.log') do |file|
  file.each do |line| 
    url = line.split[6]  # Extract the URL from the log line
    counts[url] += 1     # Increment the count for this URL
  end
end

top5 = counts.map{|url, count| [count, url] }.sort.reverse[0...5]
top5.each{|count, url| puts "#{count} #{url}" }  # Output top five URLs
```
x??

---

#### Sorting versus In-Memory Aggregation
Background context explaining the trade-offs between using in-memory aggregation (Ruby script) and sorting with repeated entries (Unix pipeline). The example provided explains how to modify both approaches based on different analysis requirements.

:p What are the advantages of using a Unix pipeline over an in-memory Ruby script for log analysis?
??x
Advantages of using a Unix pipeline:

1. **Scalability**: Sorting can be more efficient with large datasets since it can leverage disk space if memory is limited.
2. **Conciseness and Readability**: The pipeline approach is concise and leverages familiar tools.

:p What are the advantages of an in-memory Ruby script over a Unix pipeline for log analysis?
??x
Advantages of using an in-memory Ruby script:

1. **Flexibility**: Easier to modify and extend with custom logic.
2. **Memory Efficiency**: For small to medium-sized datasets, memory usage is manageable.

:p How can the command line approach be modified if you want to omit CSS files from the report?
??x
To omit CSS files, modify `awk` to exclude lines where the URL ends in `.css`:

```sh
cat /var/log/nginx/access.log | \
awk '{if ($7 !~ /\.css $/) {print $7}}'  # Exclude URLs ending with .css
```

:p How can the command line approach be modified if you want to count top client IP addresses instead of top pages?
??x
To count top client IP addresses, modify `awk` to print the first field:

```sh
cat /var/log/nginx/access.log | \
awk '{print $1}'  # Print the first field (client IP)
```

:p How can a Ruby script be modified if you want to omit CSS files from the report?
??x
To omit CSS files, modify the `url` extraction and count incrementing logic:

```ruby
counts = Hash.new(0)

File.open('/var/log/nginx/access.log') do |file|
  file.each do |line| 
    url = line.split[6]
    next if url =~ /\.css$/  # Skip URLs ending with .css

    counts[url] += 1         # Increment the count for this URL
  end
end

top5 = counts.map{|url, count| [count, url] }.sort.reverse[0...5]
top5.each{|count, url| puts "#{count} #{url}" }  # Output top five URLs
```

:p How can a Ruby script be modified if you want to count top client IP addresses instead of top pages?
??x
To count top client IP addresses, modify the `url` extraction and count incrementing logic:

```ruby
counts = Hash.new(0)

File.open('/var/log/nginx/access.log') do |file|
  file.each do |line| 
    url = line.split[0]  # Client IP is in first field (index 0)
    counts[url] += 1     # Increment the count for this URL
  end
end

top5 = counts.map{|url, count| [count, url] }.sort.reverse[0...5]
top5.each{|count, url| puts "#{count} #{url}" }  # Output top five IPs
```
x??

---

