# Flashcards: 2A014 (Part 7)

**Starting Chapter:** 33-Offline Server

---

#### Offline Ranker Overview
Background context: The offline ranker is a component designed to facilitate filtering and scoring before the real-time online environment. Its primary responsibilities include validation, batch processing, and integrating with human review processes for machine learning (ML) models.

:p What differentiates the offline ranker from the online ranker?
??x
The offline ranker runs validation in batches and outputs fast data structures that the online ranker can utilize. It also integrates with a human-in-the-loop ML process.
x??

---
#### Bloom Filter Usage
Background context: A bloom filter is a probabilistic data structure used to test whether an element is a member of a set. It allows quick subset selection from all possible candidates, making real-time filtering more efficient.

:p How does the bloom filter help in the offline ranker?
??x
The bloom filter helps by allowing the offline ranker to quickly select subsets of all possible candidates using a few features of the request. This reduces computational complexity and enables faster downstream processing.
x??

---
#### Filtering Process
Background context: The filtering process is crucial for reducing the number of candidate recommendations before applying more complex ranking algorithms.

:p What is the role of filtering in the offline ranker?
??x
The filtering step in the offline ranker uses techniques like index lookups or bloom filters to quickly reduce the number of candidate recommendations. This makes downstream algorithms more performant.
x??

---
#### Ranking Model Training
Background context: The ranking model training process involves using a large dataset to learn how to rank items effectively, often optimizing for specific objective functions.

:p What is the goal of the offline ranking step?
??x
The goal of the offline ranking step is to train models that can learn how to rank items to perform best with respect to the objective function. This prepares the necessary outputs for fast real-time scoring and ranking.
x??

---
#### Online Ranker Workflow
Background context: The online ranker leverages pre-built filtering infrastructure (e.g., bloom filters, indexes) to reduce the number of candidates before applying complex scoring and ranking models.

:p What does the online ranker do after filtering?
??x
After filtering, the online ranker accesses a feature store to embellish candidate recommendations with necessary details. Then it applies scoring and ranking models, often in multiple independent dimensions.
x??

---
#### Server Role
Background context: The server ensures that the ordered subset of recommendations satisfies the required data schema and business logic before returning them.

:p What is the role of the server?
??x
The server takes the ordered subset from the ranker, checks if it meets the necessary data schema (including essential business logic), and returns the requested number of recommendations.
x??

---

#### Offline Server Responsibilities
Background context: The offline server plays a crucial role in refining and enforcing high-level requirements for recommendations. It handles business logic such as schema enforcement, nuanced rules, and top-level priorities on recommendations.

:p What are some responsibilities of the offline server?
??x
The offline server is responsible for:
- Establishing and enforcing schemas.
- Implementing nuanced rules like avoiding certain item pairs in recommendations.
- Prioritizing high-level requirements on the returned recommendations (business logic).
- Conducting experiments to test recommendation systems before deployment.

The offline server acts as a bridge between raw data processing and real-time application, ensuring that all necessary business rules are integrated into the recommendation pipeline before it reaches the online server. This helps maintain system integrity and relevance of recommendations.
x??

---
#### Online Server Application
Background context: The online server takes the refined requirements from the offline server and applies them to the final ranked recommendations. It handles tasks like diversification, ensuring that the recommended items meet certain criteria for user experience.

:p What does the online server do?
??x
The online server:
- Reads the diversified requirements sent by the offline server.
- Applies these requirements to the ranked list of recommendations.
- Ensures that the number and type of recommendations are diverse enough to enhance the user experience.

For example, diversification rules might require a mix of new items and frequently viewed items. The online server implements this logic on the final recommendation list before sending it to the user.

```java
public class OnlineServer {
    public List<Item> applyDiversificationRules(List<Item> rankedItems) {
        // Logic to ensure diversity in recommendations
        int requiredNewItems = 5;
        int requiredPopularItems = 3;

        List<Item> diversifiedRecommendations = new ArrayList<>();
        
        for (Item item : rankedItems) {
            if (!diversifiedRecommendations.contains(item)) { // Ensuring no duplicate items
                if (item.isNew()) {
                    if (diversifiedRecommendations.size() < requiredNewItems) {
                        diversifiedRecommendations.add(item);
                    }
                } else if (item.getPopularity() > 50) {
                    if (diversifiedRecommendations.size() < requiredPopularItems) {
                        diversifiedRecommendations.add(item);
                    }
                }
            }
        }

        return diversifiedRecommendations;
    }
}
```
x??

---
#### Experimentation in Offline Server
Background context: The offline server is used for implementing logic to handle experiments. This allows testing and validating new recommendation systems before deploying them online, ensuring that any changes have a positive impact on the user experience.

:p How does the offline server facilitate experimentation?
??x
The offline server:
- Implements the logic necessary to conduct experiments.
- Provides experimental configurations that can be applied without affecting live traffic.
- Measures the performance and impact of new recommendations in a controlled environment before rolling them out online.

For example, you might want to test a new collaborative filtering algorithm. The offline server would handle the experiment by applying this new algorithm to a subset of data, evaluating its effectiveness against the current system.

```java
public class ExperimentationHandler {
    public void runExperiment(List<Item> items, String experimentConfig) {
        // Logic to apply different recommendation algorithms based on experimentConfig
        if (experimentConfig.equals("NEW_ALGORITHM")) {
            List<Item> newRecommendations = new NewAlgorithm().generate(items);
            // Log results or store for analysis
        } else if (experimentConfig.equals("DEFAULT_ALGORITHM")) {
            List<Item> defaultRecommendations = new DefaultAlgorithm().generate(items);
            // Log results or store for analysis
        }
    }
}
```
x??

---
#### Offline vs Online Server Summary
Background context: Understanding the differences between the offline and online servers is crucial. The offline server focuses on refining and enforcing business rules, while the online server applies these refined rules to final recommendations.

:p What are the main responsibilities of each server?
??x
Offline Server:
- Establishes and enforces schemas.
- Implements nuanced business logic like item pair restrictions.
- Prioritizes high-level requirements before sending data to the online server.
- Conducts experiments to test recommendation systems.

Online Server:
- Applies final rules from the offline server to ranked recommendations.
- Ensures diversity in recommendations based on predefined criteria.
- Delivers personalized and relevant recommendations directly to users.

The offline server is upstream, handling complex business logic and experimentation before the data reaches the online server for real-time processing. This separation ensures that both systems can be independently optimized and scaled as needed.
x??

---

#### Revision Control Software
Revision control software, like Git, keeps track of changes to source code and provides functionality for comparing versions and reverting back. Code is managed through patches uploaded to repositories such as GitHub.

:p What is revision control software?
??x
Revision control software, specifically tools like Git, manage the history of changes in a project's codebase by tracking different versions. It helps developers see differences between versions and revert to previous states if necessary.
x??

---

#### Python Build Systems
Python packages are libraries that extend functionality beyond standard Python with tools like TensorFlow or JAX. These can be managed using various build systems, such as pip.

:p What is the purpose of Python build systems?
??x
Python build systems manage dependencies and installation of third-party libraries in a project. This ensures consistent environments across different projects by handling package versions and installations.
x??

---

#### Random-Item Recommender Program Setup
The program uses absl flags to configure inputs like input files, output HTML file path, and number of items to recommend.

:p What is the purpose of using absl flags in this context?
??x
Using absl flags allows specifying configuration parameters at runtime. In this example, it configures paths for input JSON catalog files, output HTML filenames, and the number of recommended items.
x??

---

#### Reading Catalog Function
This function reads a JSON file containing product to category mappings.

:p How does the `read_catalog` function work?
??x
The `read_catalog` function reads a JSON file using Python's built-in `json.loads`. It opens the specified catalog file, parses its content as JSON, and returns it as a dictionary.
```python
def read_catalog(catalog: str) -> Dict[str, str]:
    with open(catalog, "r") as f:
        data = f.read()
    result = json.loads(data)
    return result
```
x??

---

#### Dumping HTML Function
This function writes the catalog subset to an HTML file.

:p What does the `dump_html` function do?
??x
The `dump_html` function takes a subset of items from the catalog and writes them into an HTML file. It creates a basic HTML table with key, category, and image columns.
```python
def dump_html(subset, output_html: str) -> None:
    with open(output_html, "w") as f:
        f.write("<HTML>\n")
        f.write("<TABLE><tr>")
        f.write("<th>Key</th><th>Category</th><th>Image</th></tr>")
        for item in subset:
            key, category = item
            url = pin_util.key_to_url(key)
            img_url = "<img src='%s'>" % url
            out = "<tr><td>%s</td><td>%s</td><td>%s</td></tr>\n" % (key, category, img_url)
            f.write(out)
        f.write("</TABLE></HTML>")
```
x??

---

#### Main Function Logic
The main function reads the catalog, shuffles it randomly, and writes a subset of items to an HTML file.

:p What does the `main` function do?
??x
The `main` function initializes the catalog by reading JSON data from a specified input file. It then shuffles the catalog entries and selects a random subset before writing these to an output HTML file.
```python
def main(argv):
    del argv  # Unused.
    catalog = read_catalog(_INPUT_FILE.value)
    catalog = list(catalog.items())
    random.shuffle(catalog)
    dump_html(catalog[:_NUM_ITEMS.value], _OUTPUT_HTML.value)
```
x??

---

#### Creating a Python Virtual Environment
A virtual environment is created using `python -m venv` and activated with the appropriate shell command.

:p How do you create and activate a Python virtual environment?
??x
To create and activate a Python virtual environment, use:
```sh
python -m venv pinterest_venv
source pinterest_venv/bin/activate
```
The first command creates the environment, and the second activates it. Each new shell session requires reactivating the environment.
x??

---

#### Installing Packages with pip
Packages are installed into a virtual environment using `pip install -r requirements.txt`.

:p How do you install packages in a Python virtual environment?
??x
Install required packages by navigating to the directory containing `requirements.txt` and running:
```sh
pip install -r requirements.txt
```
This command installs all dependencies listed in `requirements.txt`, including subpackages.
x??

---

