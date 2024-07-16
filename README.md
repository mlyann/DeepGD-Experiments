# DeepGD-Experiments

This repository contains the code and experiments related to the DeepGD framework for graph drawing using Graph Neural Networks (GNNs). 

## Citing the Original Repository

The original DeepGD repository can be found [here](https://github.com/yolandalalala/DeepGD). Please refer to it for the original implementation and additional details.

## Description

This repository contains:
- `deepgd.py`: The main implementation of the DeepGD model.
- `models.zip`: Pre-trained models used in the experiments.
- `run.py`: The script to train and evaluate the DeepGD model using various parameters.

## Parameters

Below are the parameters used in the experiments:

- **Batch Size**: 128
- **Learning Rate (lr)**: 0.01
- **Criteria**:
  - `dgd.Stress()`: 1
  - (Other criteria can be added as needed and set to appropriate weights)

## Usage

To run the experiments, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/mlyann/DeepGD-Experiments.git
    cd DeepGD-Experiments
    ```

2. **Install Dependencies**:
    Ensure you have `torch`, `torch_geometric`, and other necessary packages installed. You can install the required packages using:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Training Script**:
    Modify the `run.py` script if needed and then run it using:
    ```bash
    python run.py
    ```

4. **Model Saving**:
    The trained model will be saved to the path specified in the `run.py` script:
    ```python
    PATH = 'save_model_path'
    torch.save(model.state_dict(), PATH)
    ```

## Authorization Issues

Please ensure you have proper authorization to use and modify the code from the original DeepGD repository. Cite their repository as needed in your publications or derivative works.

## Acknowledgments

Special thanks to the authors of the original DeepGD framework. This repository is built upon their foundational work.

## Contact

For any issues or questions, please contact Minglai Yang at mlyang721@arizona.edu.
