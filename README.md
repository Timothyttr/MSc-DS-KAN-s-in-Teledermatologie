Markdown

# MSc Data Science: KANs in Teledermatology

Official repository for the Master's thesis investigating the use of Kolmogorov-Arnold Convolutional Networks (KCNs) to mitigate domain shift in teledermatology. 

## Installation
Ensure you are running a Python 3.10+ environment. Install all required dependencies before executing any scripts:
```bash
pip install -r requirements.txt
```

Usage
## Training

To train the standard CNN Baseline models:
main.py

To train the KCN-based architectures:
main_kcn.py

## Evaluation

To run the evaluation suite and extract metrics for your trained models:
Bash

# Evaluate Baseline models
evaluate.py 

# Evaluate KCN models
evaluate_kcn.py 

HPC Cluster Execution (Snellius)

This framework is fully compatible with SLURM workload managers. You can submit training and evaluation scripts in batches on supercomputers (e.g., the Snellius cluster) using standard sbatch commands:
Bash

sbatch [script.job]

Contact

Author: Timothy Toonen

Institution: University of Amsterdam (UvA)

Project Supervisor: Arun Mukundan

