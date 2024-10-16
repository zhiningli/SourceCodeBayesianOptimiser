# Bayesian Optimization Library

## Overview
Welcome to the **Bayesian Optimization Library**! This project is part of my final year project at the **University of Oxford**, where I aim to create a robust and adaptable optimization library tailored for black-box functions. The library will evolve through several stages, each building upon the previous to create a powerful tool for tackling complex optimization challenges.

## Project Stages

### Stage 1: Building the Bayesian Optimization Framework
In this stage, the goal is to implement a **Bayesian optimization library** capable of optimizing black-box functions. A black-box function is one where the internal workings are unknown or too complex to model directly. The library uses a **Gaussian Process** as a surrogate model and an **Expected Improvement (EI)** acquisition function to iteratively suggest the next best points to evaluate.

#### Key Features:
- **Surrogate Model**: Gaussian Process (GP) with customizable kernels, such as the **Radial Basis Function (RBF)**.
- **Acquisition Functions**: Initial implementation with **Expected Improvement (EI)** for selecting the next evaluation points.
- **Optimization**: Supports bounded search spaces and can be adapted for various types of objective functions.

### Stage 2: Kernel Tricks & Constrained Optimization
In the second stage, the focus will be on extending the capabilities of the optimizer to handle more complex scenarios:

- **Kernel Tricks**: Develop advanced kernels to handle constrained search spaces effectively. This will include techniques for incorporating prior knowledge about variable relevance or dependencies between variables.
- **Variable Identification**: Introduce methods to identify which variables are relevant for optimization or are conditionally relevant based on other variables' values. This allows for a more focused and efficient search in high-dimensional spaces.

### Stage 3: Custom Optimizer for Specific Black-Box Functions
In the final stage, the library will aim to build an optimizer tailored specifically to the source code of a given black-box function:

- **Source Code Analysis**: Implement a system that can read the source code of the black-box function and analyze its structure, preferably creating kernels that can encode the hierarchical structure of the source code.
- **Bespoke Optimizer Generation**: Use insights from the source code to generate an optimizer that is best suited to the specific characteristics of the function. This could involve selecting appropriate kernels, acquisition functions, or optimization strategies.
- **Adaptive Optimization**: Enhance the library's ability to adapt to various problem types, making it suitable for a wide range of applications, from simple functions to highly complex, multi-modal ones.

## Get Involved
Are you interested in building a **fully Bayesian AI** or collaborating on advanced optimization techniques? I'm excited to hear your thoughts and ideas! Contributions are welcome, whether it's through code, discussions, or sharing your knowledge.

Feel free to reach out with suggestions or questions. **Let's push the boundaries of what's possible with Bayesian optimization together!**

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Contact
For any inquiries or to discuss potential collaborations, please reach out to **[zhining.li@trinity.ox.ac.uk or zhiningli.dev@gmail.com]**.
