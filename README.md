# RNN-model
 
Here’s a GitHub project description for your work:

---

# Time Series Prediction with RNN in PyTorch

This project focuses on developing a robust **Time Series Prediction Model** using **PyTorch** to forecast sequential patterns. The model achieved an impressive **accuracy of 88.69%** after training for 300 epochs, showcasing its efficiency and reliability in handling complex time-series data.

## Features
- **Custom RNN Architecture**: Built with 533 hidden neurons, 2 layers, and `tanh` activation for optimized performance.
- **Efficient Data Preprocessing**: Implemented sequence slicing, embedding remapping, and train-test splits for clean and structured data.
- **Batch Normalization**: Integrated for faster convergence and improved generalization.
- **High-Performance Training**: Leveraged **CUDA** for GPU acceleration and optimized training with **Adam optimizer** and a learning rate of `0.00024`.
- **Flexible Data Loaders**: Supported large datasets with 2,500-batch sizes for efficient processing.
- **Model Deployment**: Enabled saving and loading of both state dictionaries and full models for scalability.

## Technologies Used
- **PyTorch**: For deep learning model creation and training.
- **Python**: As the primary programming language.
- **CUDA**: For leveraging GPU computing power.
- **Pandas & NumPy**: For data manipulation and numerical operations.
- **PyTorch DataLoader**: For efficient batching and data handling.
- **Adam Optimizer**: For fine-tuned gradient-based learning.
- **CrossEntropy Loss**: For accurate loss computation during training.

## How It Works
1. **Data Preprocessing**: The dataset is cleaned, split, and prepared with sequence slicing and remapping to handle multi-class inputs.
2. **Model Training**: A custom RNN model is trained over 300 epochs with 2,500-batch sizes, utilizing embeddings, batch normalization, and tanh activation.
3. **Evaluation**: The model's accuracy is computed after each epoch to ensure consistent progress and performance validation.
4. **Deployment**: The trained model is saved for future use and can be loaded seamlessly for predictions.

## Results
- Achieved **88.69% accuracy** on the test dataset after 300 epochs of training.
- Demonstrated the model's ability to generalize well on unseen data with minimal overfitting.

## How to Run
1. Clone the repository.
2. Install dependencies listed in `requirements.txt`.
3. Prepare the dataset in the required format.
4. Execute the training script to train the model.
5. Load the trained model for predictions.

---

Let me know if you want any further refinements!