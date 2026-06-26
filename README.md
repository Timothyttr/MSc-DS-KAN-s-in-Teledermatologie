Markdown

# MSc Data Science: KANs in Teledermatology

Official repository for the Master's thesis investigating the use of Kolmogorov-Arnold Convolutional Networks (KCNs) to mitigate domain shift in teledermatology. 

## Installation
Ensure you are running a Python 3.10+ environment. Install all required dependencies before executing any scripts:
```bash
pip install -r requirements.txt

Usage
1. Training

To train the standard CNN Baseline models:
Bash

python main.py

To train the KCN-based architectures:
Bash

python main_kcn.py

2. Evaluation

To run the evaluation suite and extract metrics for your trained models:
Bash

# Evaluate Baseline models
python evaluate.py 

# Evaluate KCN models
python evaluate_kcn.py 

HPC Cluster Execution (Snellius)

This framework is fully compatible with SLURM workload managers. You can submit training and evaluation scripts in batches on supercomputers (e.g., the Snellius cluster) using standard sbatch commands:
Bash

sbatch [your_job_script.job]

Contact

Author: Timothy Toonen

Institution: University of Amsterdam (UvA)

Project Supervisor: Arun Mukundan


*** This version looks infinitely more professional. It highlights your technical competence (like mentioning SLURM compatibility properly) and makes it effortless for your supervisor or examiners to read and run your code.
