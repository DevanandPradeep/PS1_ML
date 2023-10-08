# Breast Cancer Diagnosis using Machine Learning 

# Problem Statement: 

Breast cancer is one of the most prevalent and life-threatening diseases among women globally.  Early detection is crucial for improving treatment outcomes and reducing mortality rates. In this  project, the goal is to develop a machine learning solution that can accurately classify breast tumors  as either malignant or benign based on a dataset of real-valued features extracted from cell nuclei.

# Data Exploration
To create a model capable to predict whether a tumor cell is or not malignant I used a labeled dataset:

The Wisconsin Diagnostic Breast Cancer was created in 1995 by:

Dr. William H. Wolberg (General Surgery Dept., University of Wisconsin Clinical Sciences Center),
W. Nick Street (Computer Sciences Dept., University of Wisconsin)
and Olvi L. Mangasarian (Computer Sciences Dept., University of Wisconsin).

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

The dataset have the following structure:

Number of instances: 569

Number of attributes: 32 (ID, diagnosis, 30 real-valued input features)

Diagnosis (M = malignant, B = benign)

Missing attribute values: none

Class distribution: 357 benign, 212 malignant

All feature values are recoded with four significant digits.

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)

b) texture (standard deviation of gray-scale values)

c) perimeter
d) area

e) smoothness (local variation in radius lengths)

f) compactness (perimeter^2 / area - 1.0)

g) concavity (severity of concave portions of the contour)

h) concave points (number of concave portions of the contour)

i) symmetry

j) fractal dimension ("coastline approximation" - 1)


The mean, standard error, and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

I used this physical characteristics as features to train a machine learning algorithms to create and evaluate models that classifies if a cell is benign or malignant based in this characteristics.

# Data Preprocessing
# Data structure


Drop columns:
X = data.drop(["id", "diagnosis", "Unnamed: 32"], axis=1)
"Unannamed: 32" feature has only NaN (not a number) values "Id" feature hasn't information to help us classify benignant/malignant tumors. I deleted "Unannamed: 32" and "Id]' features and save in a X dataframe.

Using Train Test Split I divided data in train and test data.


# Logistic Regression

It was developed by statistician David Cox in 1958. The binary logistic model is used to estimate the probability of a binary response based on one or more predictor (or independent) variables (features). It allows one to say that the presence of a risk factor increases the odds of a given outcome by a specific factor. The model itself simply models probability of output in terms of input, and does not perform statistical classification, though it can be used to make a classifier, for instance by choosing a cutoff value and classifying inputs with probability greater than the cutoff as one class, below the cutoff as the other.

# Implementation
Predict if a tumor is malignant or benignant is a classification problem, I choose some of the best classifier algorithms to perform this task. The initial process was very simple, using previously shuffled and divided data:

X_train (train features) and y_train (train labels)

I used Scikit Learn that is nativily instaled from Anaconda.
