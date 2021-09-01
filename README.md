# Credal sum-product network implementation
Implementation of credal sum-product networks using the [SPFlow](https://github.com/SPFlow/SPFlow) library. Credal classification is possible, as the robustness evaluation used for [Generative Forests](https://github.com/AlCorreia/GeFs) is transferred to SPFlow.

With this implementation, the results of the experiments in my master thesis can be reproduced as well as some used visualizations.

## Usage
- Datasets are placed in the `data` directory.
- The experiments can be setup and run with the file `network.py`. The folowing structures are available:
  - Tree -> LearnSPN structure
  - CD-Tree -> class-discriminative LearnSPN tree structure
  - CD-DAG -> class-discriminative RAT-SPN DAG structure

## Visualizations

Visualizations can be created in various ways. The `vis_compare.py` file creates visualizations for the comparison experiments and the `vis.py` file creates visualizations for the soft evidence experiments. Furthermore, there are various files for creating visualizations used in the report.
