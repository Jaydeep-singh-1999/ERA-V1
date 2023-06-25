##  Assignment 08

1.Change the dataset to CIFAR10 

2.Make this network:

 C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
 
3.Keep the parameter count less than 50000

4.Try and add one layer to another

5.Max Epochs is 20

6.You are making 3 versions of the above code (in each case achieve above 70% accuracy):
<pre>
  1.Network with Group Normalization
  
  2.Network with Layer Normalization
  
  3.Network with Batch Normalization
</pre>
8.Share these details
<pre>
   1.Training accuracy for 3 models
   2.Test accuracy for 3 models
   3. Find 10 misclassified images for the BN model, and show them as a 5x2 image matrix in 3 separately annotated images.
</pre>
write an explanatory README file that explains:

what is your code all about,
your findings for normalization techniques,
add all your graphs
your collection-of-misclassified-images 


--- 



📚 **Please refer below for more information and guidelines!** 🚀

🔍 It's recommended to follow the information provided below to understand the implementation details and usage guidelines.

# Utils.py

This file contains the implementation of the `test` and `train` functions, which are crucial for the training and testing phases of the model.

### Functions

- `test(model, data_loader, device)`: Evaluates the model's performance on the test dataset. Pass the `model`, `data_loader` containing the test data, and specify the `device` (CPU or GPU) for computation. The function returns the average loss and accuracy achieved by the model on the test data.

- `train(model, data_loader, optimizer, device)`: Trains the model using the training dataset. Supply the `model`, `data_loader` with the training data, `optimizer` (SGD in this case) for updating model parameters, and specify the `device` (CPU or GPU) for computation. The function returns the average loss and accuracy achieved by the model on the training data.

# Model.py

This file contains the implementation of various models used in the project. It includes the following classes:

### Batch Norm (BN)

The `BN` class implements Batch Normalization, a technique that normalizes the inputs of each layer to improve training stability and speed for deep neural networks.

### Layer Norm (LN)

The `LN` class implements Layer Normalization, which normalizes the inputs across the features of each sample in a mini-batch. It offers an alternative to batch normalization, useful when batch size is small or batch normalization is not applicable.

### Group Norm (GN)

The `GN` class implements Group Normalization, which divides channels into groups and computes mean and variance for each group separately. It can be an alternative to batch normalization under specific circumstances.

### Add_layer
This class is implimentation of x+conv(x).

# S8.ipynb

The `s8.ipynb` file contains code related to data loading, transformations, and the training loop for multiple epochs.

### DataLoader

The DataLoader is responsible for loading the dataset and preparing it for training. It handles tasks such as batching the data, shuffling, and applying any necessary transformations.

### Transformations

Transformations are applied to the dataset to perform data augmentation and preprocessing. They can include operations such as random cropping, resizing, normalization, or any other modifications required to enhance the training data.

### Training Loop

The training loop is a crucial part of the model training process. It iterates over the dataset for a specified number of epochs, feeding the data to the model, computing loss, and updating the model's parameters using an optimizer.

### Image Augmentation Techniques

The model employs the following image augmentation techniques:

- Random Crop: Randomly crops a portion of the input image, introducing robustness by exposing the model to different image regions during training.

- Random Horizontal Flip: Randomly flips the input image horizontally, increasing the diversity of the training data and enhancing the model's ability to generalize to new data.

### Optimization Algorithm

The model uses Stochastic Gradient Descent (SGD) as the optimizer with a learning rate of 0.01 and momentum of 0.9. SGD is a popular optimization algorithm for training deep learning models. The learning rate controls the step size at each iteration, while momentum helps accelerate convergence by incorporating a fraction of the previous update.

### Number of Epochs

The model has been trained for 20 epochs. An epoch refers to one complete pass through the entire training dataset during the training process.

⚠️ Please make sure to follow the provided guidelines while exploring the code and using the functions and classes mentioned above.

Feel free to dive into the code for further details on implementation and usage instructions.

---

# Results: 

##  1.Batch Normalization

![Screenshot 2023-06-24 020201](https://github.com/Jaydeep-singh-1999/ERA-V1/assets/135359624/9a0ba711-cebc-4a42-a877-52d4e53853ff)

### Parameters:48,810
###  Training Accuracy = 70.53
###  Test Accuracy= 72 

### Observation: Most the time model is underfitting because i have used drop out and performed image random crop and random horizonatal flip.

## 2. Layer Normalization

![LN](https://github.com/Jaydeep-singh-1999/ERA-V1/assets/135359624/960e2e61-10b0-4d8f-a93c-20c51ff7a8ba)
### Parameters:48,810

### Training Accuracy = 68.59
### Test Accuracy= 71.13

### observation : Here also underfitting.


## 3.Group Normalisation

![GN](https://github.com/Jaydeep-singh-1999/ERA-V1/assets/135359624/7af7a276-1c14-4cf9-baf6-39673ae07c3e)


### Parameters:48,810
### Training Accuracy:68.17
### Test Accuracy: 71.02

### Observation: we can also see little bit of underfitting 

## 4. Addding Layers (x+conv(x)) in BN 

![Add_layers](https://github.com/Jaydeep-singh-1999/ERA-V1/assets/135359624/82ccd4d8-b782-4512-ba91-21752e8c3383)
### Parameters : 49370
### Training Accuracy:78.08
### Test Accuracy: 78.17

