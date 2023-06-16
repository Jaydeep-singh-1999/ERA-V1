# Assignment S7

## Target

Your new target is:
- 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
- Less than or equal to 15 Epochs
- Less than 8000 Parameters

Do this using your modular code. Every model that you make must be there in the `model.py` file as `Model_1`, `Model_2`, etc.

Do this in exactly 3 steps.

Each File must have a "target, result, analysis" TEXT block (either at the start or the end).

You must convince why have you decided that your target should be what you have decided it to be, and your analysis MUST be correct. 

Evaluation is highly subjective, and if you target anything out of the order, marks will be deducted. 

Explain your 3 steps using these targets, results, and analysis with links to your GitHub files (Colab files moved to GitHub). 

Keep Receptive field calculations handy for each of your models. 

If your GitHub folder structure or file_names are messy, -100. 

# SOLUTION: -

## Step 1:

### Model_1:

#### Target: 
  1. To acheive a skeleton which can give 99.4 % accuracy continuously
  2. No constraints on parameter
#### Results:
  1. Parameters: 1,968,874
  2. Best Training Accuracy: 99.85
  3. Best Test Accuracy: 99.18
#### Analysis:
  1. Model is not acheving 99.4 even and it is overfitting.
  2. Test Accuracy is stuck at 99.07 approx.
  3. No BatchNorm is Used
### Model_2:
#### Target:
  1. To acheive 99.4% val accuracy consistently
  2. Using BatchNorm
#### Results:
  1. Parameters: 1,968,874
  2. Best Training Accuracy: 100%
  3. Best Test Accuracy: 99.58%
#### Analysis:
  1. Model is achieving more then 99.4% becuase I just added batch norm layer.
  2. Still we can see overfitting in the model which can be reduce by regularization techniques liek drop out later.
  3. Model parameters are high now is the time to finalize structure and squeeze it.

### Model_3:

#### Target:
  1. To reduce parameter under 8K and achieve 99.4% consistent accuracy.
#### Result:
  1. Parameters: 7,836
  2. Best Training Accuracy: 99.69%
  3. Best Test Accuracy: 99.22%
#### Analysis:
  1. Overfitting in the model and Accuracy is stuck around 99%.

## Step 2:

### Model_4:

#### Target:
  1. To reduce overfitting and achieving 99.4%  val accuracy.

#### Result:
  1. Parameters : 7,836
  2. Best Training Accuracy: 99.28%
  3. Best Test Accuracy: 99.31%
#### Analysis:
  1. Model is underfitting due to drop out and training is hard through out the model.

### Model_5






 
