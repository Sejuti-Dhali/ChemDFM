\# ChemDFM



This is the official repository for \*\*ChemDFM\*\*, a framework designed for drug discovery and advanced chemical molecular modeling.



\##  Overview

ChemDFM is a robust framework built to facilitate drug response prediction and molecular-conditioned learning. 



\### What is being done here?

The project focuses on predicting how specific cells respond to chemical compounds. To achieve this, it performs the following:

\* \*\*Molecular Conditioned Learning:\*\* It uses molecular structures as inputs to train models that predict biological outcomes.

\* \*\*Residual Modeling:\*\* The framework employs residual learning techniques to improve prediction accuracy by focusing on learning the differences (residuals) in drug responses.

\* \*\*Biological Validation:\*\* It includes a post-hoc analysis phase to ensure that the model's predictions align with known biological gene space and molecular interactions.

\* \*\*Systematic Diagnostics:\*\* It utilizes decision gates and diagnostic tools to evaluate model performance and identify potential failure points during the training process.



\##  Key Features

\* \*\*Deep Learning Models:\*\* Supports molecular and cell-aware residual modeling.

\* \*\*Data Processing:\*\* End-to-end pipeline from raw data to processed datasets.

\* \*\*Evaluation:\*\* Built-in notebooks for biological validation and model robustness testing.

\* \*\*Configuration-Driven:\*\* Easily modify model parameters and hyperparameters using `yaml` files.



\##  Project Structure

```text

ChemDFM/

├── configs/            # Configuration files for models and data

├── data/               # Raw, interim, and processed data

├── notebooks/          # Jupyter Notebooks for experiments and analysis

├── src/                # Core source code (models, training, evaluation)

├── scripts/            # Executable scripts

└── environment.yml     # Required environment dependencies

