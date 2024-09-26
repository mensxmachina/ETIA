
# ETIA: A Comprehensive Automated Causal Discovery Library

## Library Overview

ETIA (Αιτία (pronounced etía): "cause" in Greek) is a cutting-edge automated causal discovery library that takes causal analysis beyond traditional methods. It is designed to tackle complex, real-world problems by automating the entire causal discovery process, offering a combination of feature selection, causal structure learning, and causal reasoning validation that is unmatched in other libraries.

ETIA provides:
- **Dimensionality Reduction** through its **Automated Feature Selection (AFS)** module, which ensures only the most relevant variables are used, even in high-dimensional datasets.
- **Causal Structure Learning** via the **Causal Learning (CL)** module, automating the discovery of the causal graph with the best possible fit to the data.
- **Causal Reasoning Validation (CRV)** offers tools for computing confidence in discovered relationships and visualizing causal paths.

Unlike existing libraries, **ETIA** does **not** simply offer isolated algorithms—it provides a **fully automated pipeline** that optimizes and customizes each step of the causal discovery process, ensuring the results are robust, interpretable, and reliable for both researchers and industrial practitioners.

## Why ETIA is Unique

ETIA goes beyond other causal discovery libraries by offering:

- **End-to-End Automation**: ETIA fully automates the discovery process, combining various algorithms and approaches to find the best configuration for a given dataset. This level of automation is **rare** in other libraries, which often leave algorithm selection and tuning to the user.
- **Out-of-Sample Causal Tuning**: A method developed specifically for ETIA, **Out-of-Sample Causal Tuning** ensures the best causal graph is selected without the need for manually tuned parameters. This is essential when working in unsupervised environments where traditional estimation methods, like K-fold cross-validation, fail.
- **Confidence Estimation and Visualization**: ETIA not only discovers causal graphs but also evaluates the confidence of each relationship within the graph. Bootstrapped confidence metrics and visualization tools help users understand which findings are most reliable.
- **Dimensionality Reduction with Causal Insight**: The AFS module uses sophisticated techniques to reduce the dimensionality of datasets without sacrificing causal accuracy, a critical need when dealing with high-dimensional, real-world data.
- **Handling Latent Variables**: Unlike many causal discovery tools, ETIA can identify causal relationships even in datasets with hidden confounders through algorithms like FCI and GFCI.

## Core Features

### 1. Automated Feature Selection (AFS)

AFS goes beyond standard feature selection by targeting the **Markov Boundary** of the outcome of interest. This approach ensures that you work only with the most causally relevant variables, preventing the noise and redundancy that plague other methods.

| Algorithm                | Description                                             | Data Type |
|--------------------------|---------------------------------------------------------|-----------|
| `FBED`                   | Forward-Backward selection with Early Dropping | Mixed     |
| `SES`                    | Statistical Equivalence Selection | Mixed     |
**Why It Matters**: In high-dimensional datasets, choosing the right features is critical. AFS uses state-of-the-art techniques to find **causally relevant features**, ensuring that the subsequent causal analysis is accurate and manageable, even in datasets with hundreds of variables.

### 2. Causal Learning (CL)

The CL module identifies causal relationships using a variety of algorithms that are automatically optimized to your data. ETIA's **causal tuning** mechanism guarantees the selection of the best possible causal structure without the need for user intervention.

| Algorithm                | Latent Variables Supported | Tests/Scores Used                           | Data Type |
|--------------------------|----------------------------|---------------------------------------------|-----------|
| `PC Algorithm`           | ✕                          | FisherZ, BIC                               | Continuous|
| `FCI`                    | ✓                          | Conditional independence tests, G2, Chi2   | Mixed     |
| `GFCI`                   | ✓                          | Conditional independence tests, CG         | Mixed     |
| `FGES`                   | ✕                          | BDeu, G2, Chi2                             | Discrete  |
| `DirectLiNGAM`           | ✕                          | ICA-based causality tests                  | Continuous|

**Why It Matters**: Traditional causal discovery libraries expect users to manually choose an algorithm. ETIA’s **automated pipeline selects the best algorithm** for your dataset, saving time and reducing the risk of suboptimal results. Additionally, it can handle datasets with latent variables, which most other systems cannot do.

### 3. Causal Reasoning Validation (CRV)

CRV provides advanced tools to evaluate the discovered causal graph, offering confidence estimates and comprehensive visualizations. It can answer specific causal queries, making it an invaluable tool for decision-makers and researchers alike.

| Functionality            | Description                                                  |
|--------------------------|--------------------------------------------------------------|
| `Edge Confidence`         | Compute confidence for causal edges using bootstrapping      |
| `Path Discovery`          | Discover and validate causal paths                          |
| `Visualization`           | Visualize graphs and causal relations using Cytoscape        |

**Why It Matters**: The ability to **compute and visualize confidence in causal relationships** sets ETIA apart from other libraries. Users can trust that the discovered causal graph is not just a hypothesis but a statistically backed structure with clearly defined confidence levels.

## Installation

You can install ETIA directly from PyPi using pip:

```bash
pip install etia
```

Alternatively, clone the repository and install the dependencies:

```bash
git clone <repository-url>
cd library directory
pip install -r requirements.txt
make all
```

### Prerequisites

- **Python 3.7+**
- **Java 17** (required for Tetrad algorithms in the Causal Learning module)
- **R 1.1+** (required for some feature selection algorithms in the AFS module)

## Usage

Once installed, ETIA can be used by importing its modules. Here is a simple example for feature selection and causal discovery:

```python
from ETIA.AFS import AFS
from ETIA.CausalLearning import CausalLearner

# Feature selection
afs = AFS()
selected_features = afs.select_features(dataset)

# Causal discovery
cl = CausalLearner()
causal_graph = cl.learn(dataset)
```

## Testing

You can run the test suite using:

```bash
pytest tests/
```

Make sure that all dependencies, including Java and R, are correctly installed before running tests.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
