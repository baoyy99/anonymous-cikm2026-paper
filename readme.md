readme init
# TSLoss: A Multi-dimensional Evaluation Framework and Benchmark for Time Series Forecasting Loss 
Code Repository (For Double-Blind Review Only)

## Project Overview
This repository contains the code implementation for the paper "TSLoss". To adhere to double-blind review principles and protect intellectual property, we adopt a **phased open-source strategy**:
- **Review Phase**: Provide all non-core code to ensure reviewers can understand the project structure of our method.
- **Acceptance Phase**: Fully open-source all code, data, and models to guarantee experimental reproducibility.

## Repository Structure
```
├── Loss_Metric/        # TSLoss Definition
├── TSLoss/             # Benchmark Adaptation
│   ├── OpenLTM/exp/    # Foundation Model Benchmark Library
│   ├── TQNet/exp       # Specific Model
│   └── TSLib/exp       # Specific Model Benchmark Library
```

## Contents to Be Added (After Acceptance)
1. Core algorithm implementation in the `Loss_Metric/` directory
2. Detailed reproduction guide and parameter specifications

## Support During Review
- Provide detailed responses to any code-related questions from reviewers.
- Provide code snippets for specific modules upon reviewer request (without disclosing the overall core logic).
- Provide additional experimental results as needed.

## Reproducibility Statement
The complete code will be uploaded to a public GitHub repository within 7 working days upon acceptance of the paper, and the final version of the paper will be updated with the corresponding Git link. All experimental results reported in this paper can be fully reproduced using the complete code provided after acceptance.
