# MIC-CNN-NSynth: Musical Instrument Classification using Deep Learning

## 🚀 Project Overview

This project focuses on the automatic classification of musical instruments within audio recordings using deep learning techniques. The primary goal is to identify the dominant instrument in 4-second sound segments.

The implementation demonstrates the effective use of **Convolutional Neural Networks (CNN)** applied to a visual representation of the audio signal (**log-mel spectrograms**) for robust frequency feature extraction.

**Final Performance:** The model achieved a final validation accuracy of **97%** (Macro F1: 0.98) on a balanced subset of the NSynth Dataset.

## 🛠️ Technical Stack

* **Language:** Python
* **Deep Learning Libraries:** TensorFlow / Keras (for modeling), PyTorch (used in initial phases for data handling).
* **Audio Processing:** librosa
* **Data Source:** NSynth Dataset (Google Magenta)

## 📂 Project Structure and Evolution Phases

The project development was meticulously documented across four successive phases. Each phase is represented by a corresponding **Jupyter Notebook** (containing the code and primary results) and a **PDF report** (providing detailed analysis, charts, and conclusions for that phase).

| Phase | Jupyter Notebook (.ipynb) | Report (.pdf) | Brief Description |
| :--- | :--- | :--- | :--- |
| **Phase 1** | `1_Phase_Mini_Subset_SmallCNN.ipynb` | `1. first phase report.pdf` | Setting up the entire pipeline, initial testing on a small subset (~4k), and implementation of the baseline **SmallCNN** model. |
| **Phase 2** | `2_Phase_Validation_Subset_SmallCNN.ipynb` | `2. second phase report.pdf` | Scaling the model to a larger validation set (~12k) to check generalization and stability of the initial **SmallCNN** model. |
| **Phase 3** | `3_Phase_DeepCNN_Implementation.ipynb` | `3. third phase report.pdf` | Introduction of a deeper and more complex **DeepCNN** architecture (with BatchNorm and Dropout) leading to a significant performance boost (accuracy ~93%). |
| **Phase 4** | `4_Phase_Balancing_Final_Model.ipynb` | `4. fourth phase report.pdf` | Final optimization: Extended training (10 epochs) and application of **Weighted Random Sampling** to effectively resolve class imbalance, achieving the final accuracy of **97%**. |

## 🚀 Setup and Reproduction

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Install dependencies:** Install the required Python libraries:
    ```bash
    pip install tensorflow keras librosa numpy pandas
    ```
3.  **Data:** Download and unpack the NSynth dataset (or the corresponding split used in the notebooks) and place it in the appropriate project directory.
4.  **Run:** It is recommended to open the Jupyter Notebook files sequentially (from 1 to 4) to follow the implementation and model training process in each phase.
