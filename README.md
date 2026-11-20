# Elman Recurrent Neural Network (MATLAB)

A custom, "from-scratch" implementation of an **Elman Recurrent Neural Network** in MATLAB. This project focuses on non-linear dynamic system identification using backpropagation through time (BPTT).

## Overview
This repository provides a clean, modular MATLAB implementation of an Elman Network. Unlike standard feedforward networks, this architecture utilizes a context layer (recurrent weights) to capture temporal dependencies in data, making it suitable for time-series forecasting and dynamic system modeling.

## Key Features
- **From-Scratch Implementation:** No reliance on MATLAB Toolboxes; all logic (Forward/Backward pass) is implemented manually.
- **Chain Rule Integration:** Precise backpropagation implementation for recurrent weights.
- **Dynamic System Identification:** Trained to approximate complex non-linear plant dynamics.
- **Data Partitioning:** Includes distinct training, testing, and checking phases for robust model validation.
- **Visual Analytics:** Real-time plotting of training and testing Mean Squared Error (MSE).

## Methodology
- **Architecture:** 2-layer network with a recurrent hidden state.
- **Training Algorithm:** Gradient Descent via Chain Rule for recurrent neural networks.
- **Activation Functions:** `tansig` for hidden layers, `purelin` for output.

## How to Run
1. Clone the repository or download the source code.
2. Open `Elman_RNN_System_Identification.m` in MATLAB.
3. Run the script. The system will generate data, train the network, and visualize the prediction performance vs. actual target data.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
