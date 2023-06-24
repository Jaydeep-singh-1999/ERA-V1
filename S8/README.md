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
Upload your complete assignment on GitHub and share the link on LMS



##  1.Batch Normalization

![Screenshot 2023-06-24 020201](https://github.com/Jaydeep-singh-1999/ERA-V1/assets/135359624/9a0ba711-cebc-4a42-a877-52d4e53853ff)

### Parameters:48,810
###  Training Accuracy = 70.53
###  Test Accuracy= 72 

### Observation: Most the time model is underfitting because i have used drop out and performed image random crop and random horizonatal flip.

## 2. Layer Normalization


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
