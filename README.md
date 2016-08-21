
# Machine Learning Engineer Nanodegree

## Supervised Learning

### Description

A local school district has a goal to reach a 95% graduation rate by the end of the decade by identifying students who need intervention before they drop out of school. As a software engineer contacted by the school district, your task is to model the factors that predict how likely a student is to pass their high school final exam, by constructing an intervention system that leverages supervised learning techniques. The board of supervisors has asked that you find the most effective model that uses the least amount of computation costs to save on the budget. You will need to analyze the dataset on students' performance and develop a model that will predict the likelihood that a given student will pass, quantifying whether an intervention is necessary.


### Question 1 - Classification vs. Regression
*Your goal for this project is to identify students who might need early intervention before they fail to graduate. Which type of supervised learning problem is this, classification or regression? Why?*

**Answer: **

This is classification problem because the output that we are looking for will be used to determine whether or not an intervention is needed for a given student.

If this was a Regression problem, the question we would ask could be, "What grade will this student graduate with?".

## Exploring the Data
Run the code cell below to load necessary Python libraries and load the student data. Note that the last column from this dataset, `'passed'`, will be our target label (whether the student graduated or didn't graduate). All other columns are features about each student.


```python
# Import libraries
import numpy as np
import pandas as pd
from time import time
from __future__ import division
from sklearn.metrics import f1_score

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"
```

    Student data read successfully!


### Implementation: Data Exploration
Let's begin by investigating the dataset to determine how many students we have information on, and learn about the graduation rate among these students. In the code cell below, you will need to compute the following:
- The total number of students, `n_students`.
- The total number of features for each student, `n_features`.
- The number of those students who passed, `n_passed`.
- The number of those students who failed, `n_failed`.
- The graduation rate of the class, `grad_rate`, in percent (%).



```python
n_students, n_features = student_data.shape

n_passed, _ = student_data[student_data['passed'] == 'yes'].shape

n_failed, _ = student_data[student_data['passed'] == 'no'].shape

grad_rate = (n_passed / n_students) * 100

# Print the results
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

print "\n----------------------\n"

# Check for data imbalance, will deal with this by statifying the data when splitting
y_pred = ['yes']*n_students
y_true = ['yes']*n_passed + ['no']*n_failed
score = f1_score(y_true, y_pred, pos_label='yes', average='binary')

print "F1 score for all 'yes' on students: {:.4f}".format(score)
```

    Total number of students: 395
    Number of features: 31
    Number of students who passed: 265
    Number of students who failed: 130
    Graduation rate of the class: 67.09%

    ----------------------

    F1 score for all 'yes' on students: 0.8030


## Preparing the Data
In this section, we will prepare the data for modeling, training and testing.

### Identify feature and target columns
It is often the case that the data you obtain contains non-numeric features. This can be a problem, as most machine learning algorithms expect numeric data to perform computations with.

Run the code cell below to separate the student data into feature and target columns to see if any features are non-numeric.


```python
# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1]

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print "\nFeature values:"
print X_all.head()
```

    Feature columns:
    ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']

    Target column: passed

    Feature values:
      school sex  age address famsize Pstatus  Medu  Fedu     Mjob      Fjob  \
    0     GP   F   18       U     GT3       A     4     4  at_home   teacher   
    1     GP   F   17       U     GT3       T     1     1  at_home     other   
    2     GP   F   15       U     LE3       T     1     1  at_home     other   
    3     GP   F   15       U     GT3       T     4     2   health  services   
    4     GP   F   16       U     GT3       T     3     3    other     other   

        ...    higher internet  romantic  famrel  freetime goout Dalc Walc health  \
    0   ...       yes       no        no       4         3     4    1    1      3   
    1   ...       yes      yes        no       5         3     3    1    1      3   
    2   ...       yes      yes        no       4         3     2    2    3      3   
    3   ...       yes      yes       yes       3         2     2    1    1      5   
    4   ...       yes       no        no       4         3     2    1    2      5   

      absences  
    0        6  
    1        4  
    2       10  
    3        2  
    4        4  

    [5 rows x 30 columns]


### Preprocess Feature Columns

As you can see, there are several non-numeric columns that need to be converted! Many of them are simply `yes`/`no`, e.g. `internet`. These can be reasonably converted into `1`/`0` (binary) values.

Other columns, like `Mjob` and `Fjob`, have more than two values, and are known as _categorical variables_. The recommended way to handle such a column is to create as many columns as possible values (e.g. `Fjob_teacher`, `Fjob_other`, `Fjob_services`, etc.), and assign a `1` to one of them and `0` to all others.

These generated columns are sometimes called _dummy variables_, and we will use the [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) function to perform this transformation. Run the code cell below to perform the preprocessing routine discussed in this section.


```python
def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''

    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  

        # Collect the revised columns
        output = output.join(col_data)

    return output

X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))
```

    Processed feature columns (48 total features):
    ['school_GP', 'school_MS', 'sex_F', 'sex_M', 'age', 'address_R', 'address_U', 'famsize_GT3', 'famsize_LE3', 'Pstatus_A', 'Pstatus_T', 'Medu', 'Fedu', 'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home', 'reason_other', 'reason_reputation', 'guardian_father', 'guardian_mother', 'guardian_other', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']


### Implementation: Training and Testing Data Split
So far, we have converted all _categorical_ features into numeric values. For the next step, we split the data (both features and corresponding labels) into training and test sets. In the following code cell below, you will need to implement the following:
- Randomly shuffle and split the data (`X_all`, `y_all`) into training and testing subsets.
  - Use 300 training points (approximately 75%) and 95 testing points (approximately 25%).
  - Set a `random_state` for the function(s) you use, if provided.
  - Store the results in `X_train`, `X_test`, `y_train`, and `y_test`.


```python
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

# shuffle the data
X_all, y_all = shuffle(X_all, y_all, random_state=42)

# split the data into training and testing sets,
# use `stratify` to maintain balance between classifications
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, stratify=y_all,
                                                    test_size=0.24, random_state=42)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

# Check for imbalances
print "Grad rate of the training set: {:.2f}%".format(100 * (y_train == 'yes').mean())
print "Grad rate of the testing set: {:.2f}%".format(100 * (y_test == 'yes').mean())
```

    Training set has 300 samples.
    Testing set has 95 samples.
    Grad rate of the training set: 67.00%
    Grad rate of the testing set: 67.37%


## Training and Evaluating Models
In this section, you will choose 3 supervised learning models that are appropriate for this problem and available in `scikit-learn`. You will first discuss the reasoning behind choosing these three models by considering what you know about the data and each model's strengths and weaknesses. You will then fit the model to varying sizes of training data (100 data points, 200 data points, and 300 data points) and measure the F<sub>1</sub> score. You will need to produce three tables (one for each model) that shows the training set size, training time, prediction time, F<sub>1</sub> score on the training set, and F<sub>1</sub> score on the testing set.

### Question 2 - Model Application
*List three supervised learning models that are appropriate for this problem. What are the general applications of each model? What are their strengths and weaknesses? Given what you know about the data, why did you choose these models to be applied?*

**Answer: **

_To narrow down my choices from the variety of algorithims available in sklearn, here's the thought process followed:_

- The goal is to predict a category, we have to use a classification algorithim.
- The data available to us is relatively sparce with many features. This means we better stay away from neural networks since they might end up being hard to train.
- Our data is numeric and not text based which means we won't benefit from using Naive Bayes as much.

----

The algorithims chosen are then as follows:

##### Support Vector Classifier

- _General Applications_
    - text (and hypertext) categorization such as email classification and web searching
    - image classification
    - bioinformatics (Protein classification, Cancer classification)
    - hand-written character recognition


- _Strengths_
    - Avoids localization problems
    - Can use the kernel trick
    - Accurate with small & clean datasets
    - Effecient since it only uses a subset of the data

- _Weaknesses_
    - Sensitive to noise (may need to drop some features that are causing noise)
    - It only considers two classes at a time
    - Can be painful to train with large datasets

- _Reasons for Choosing_
    - SVMs can do well when with a small set as long as the number of data points is larger than the number of features being considered.


##### K-Nearest Neighbours

- _General Applications_
    - Gene Expression (sometimes with SVMs)
    - Predicting stock price change
    - Automatic text classification
- _Strengths_
    - Low training times
    - Memory based reasoning
- _Weaknesses_
    - Results may change over time as the algorithim is query based
- _Reasons for Choosing_
    - One could argue that in our use case, a student who is about to dropout is likely to be in a similar situation to a student who needs an intervention.


##### AdaBoost Classifier (with DecisionTrees estimator)

- _General Applications_
    - An AdaBoost classifier is to interpolot between many weak learners to create a more accurate model.
- _Strengths_
    - Improves classification accuracy
    - Can be used with many different classifiers
    - Not prone to overfitting which is helpful given our dataset's limited size
- _Weaknesses_
    - Effected by noisy data
    - Longer training time since boosting trains multiple instances of the estimator
- _Reasons for Choosing_
    - In the case that the above 2 classifiers don't work well, we can use the AdaBoost classifier with SVC or simply go with the default estimator (Decision Trees).

----

### Setup
Run the code cell below to initialize three helper functions which you can use for training and testing the three supervised learning models you've chosen above. The functions are as follows:
- `train_classifier` - takes as input a classifier and training data and fits the classifier to the data.
- `predict_labels` - takes as input a fit classifier, features, and a target labeling and makes predictions using the F<sub>1</sub> score.
- `train_predict` - takes as input a classifier, and the training and testing data, and performs `train_clasifier` and `predict_labels`.
 - This function will report the F<sub>1</sub> score for both the training and testing data separately.


```python
def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)


def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()

    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))

    # Train the classifier
    train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))
```

### Implementation: Model Performance Metrics
With the predefined functions above, you will now import the three supervised learning models of your choice and run the `train_predict` function for each one. Remember that you will need to train and predict on each classifier for three different training set sizes: 100, 200, and 300. Hence, you should expect to have 9 different outputs below — 3 for each model using the varying training set sizes. In the following code cell, you will need to implement the following:
- Import the three supervised learning models you've discussed in the previous section.
- Initialize the three models and store them in `clf_A`, `clf_B`, and `clf_C`.
 - Use a `random_state` for each model you use, if provided.
 - **Note:** Use the default settings for each model — you will tune one specific model in a later section.
- Create the different training set sizes to be used to train each model.
 - *Do not reshuffle and resplit the data! The new training points should be drawn from `X_train` and `y_train`.*
- Fit each model with each training set size and make predictions on the test set (9 in total).  
**Note:** Three tables are provided after the following code cell which can be used to store your results.


```python
# Import the three supervised learning models from sklearn
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

# Initialize the three models
clf_A = SVC()
clf_B = KNeighborsClassifier(n_neighbors=3)
clf_C = AdaBoostClassifier(n_estimators=10)

# Loop through each classifier
for clf in [clf_A, clf_B, clf_C]:
    print "=============================================================="
    print "\n{}: \n".format(clf.__class__.__name__)
    # for each classifier, train and predict with a different training set size ==> 100, 200, 300
    for i, n in enumerate([100, 200, 300]):
        train_predict(clf, X_train[:n], y_train[:n], X_test, y_test)
        if i < 2: print "----------------"


print "=============================================================="
print '''
We can benchmark the F1 scores of each of the classifiers to the score below:
'''
print "F1 score for predicting all \"yes\" on test set: {:.4f}".format(
    f1_score(y_test, ['yes']*len(y_test), pos_label='yes', average='binary'))
```

    ==============================================================

    SVC:

    Training a SVC using a training set size of 100. . .
    Trained model in 0.0019 seconds
    Made predictions in 0.0011 seconds.
    F1 score for training set: 0.8684.
    Made predictions in 0.0008 seconds.
    F1 score for test set: 0.8133.
    ----------------
    Training a SVC using a training set size of 200. . .
    Trained model in 0.0030 seconds
    Made predictions in 0.0022 seconds.
    F1 score for training set: 0.8750.
    Made predictions in 0.0013 seconds.
    F1 score for test set: 0.7714.
    ----------------
    Training a SVC using a training set size of 300. . .
    Trained model in 0.0093 seconds
    Made predictions in 0.0076 seconds.
    F1 score for training set: 0.8602.
    Made predictions in 0.0031 seconds.
    F1 score for test set: 0.8163.
    ==============================================================

    KNeighborsClassifier:

    Training a KNeighborsClassifier using a training set size of 100. . .
    Trained model in 0.0012 seconds
    Made predictions in 0.0020 seconds.
    F1 score for training set: 0.8889.
    Made predictions in 0.0019 seconds.
    F1 score for test set: 0.7344.
    ----------------
    Training a KNeighborsClassifier using a training set size of 200. . .
    Trained model in 0.0007 seconds
    Made predictions in 0.0033 seconds.
    F1 score for training set: 0.8773.
    Made predictions in 0.0024 seconds.
    F1 score for test set: 0.7840.
    ----------------
    Training a KNeighborsClassifier using a training set size of 300. . .
    Trained model in 0.0011 seconds
    Made predictions in 0.0057 seconds.
    F1 score for training set: 0.8879.
    Made predictions in 0.0022 seconds.
    F1 score for test set: 0.7752.
    ==============================================================

    AdaBoostClassifier:

    Training a AdaBoostClassifier using a training set size of 100. . .
    Trained model in 0.0369 seconds
    Made predictions in 0.0022 seconds.
    F1 score for training set: 0.8725.
    Made predictions in 0.0021 seconds.
    F1 score for test set: 0.7556.
    ----------------
    Training a AdaBoostClassifier using a training set size of 200. . .
    Trained model in 0.0351 seconds
    Made predictions in 0.0074 seconds.
    F1 score for training set: 0.8500.
    Made predictions in 0.0014 seconds.
    F1 score for test set: 0.7287.
    ----------------
    Training a AdaBoostClassifier using a training set size of 300. . .
    Trained model in 0.0351 seconds
    Made predictions in 0.0014 seconds.
    F1 score for training set: 0.8423.
    Made predictions in 0.0011 seconds.
    F1 score for test set: 0.7852.
    ==============================================================

    We can benchmark the F1 scores of each of the classifiers to the score below:

    F1 score for predicting all "yes" on test set: 0.8050


### Tabular Results
Edit the cell below to see how a table can be designed in [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#tables). You can record your results from above in the tables provided.

** Classifer 1 - SVC**  

| Training Set Size |      Training Time      | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
| :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
| 100               |        0.0020           |        0.0010          |     0.8684       |    0.8133       |
| 200               |        0.0031           |        0.0012          |     0.8750       |    0.7714       |
| 300               |        0.0071           |        0.0024          |     0.8602       |    0.8163       |

----


** Classifer 2 - KNN**  

| Training Set Size |      Training Time      | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
| :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
| 100               |        0.0009           |        0.0017          |     0.8889       |     0.7344      |
| 200               |        0.0012           |        0.0020          |     0.8773       |     0.7840      |
| 300               |        0.0008           |        0.0032          |     0.8879       |     0.7752      |

----

** Classifer 3 - AdaBoost - Decision Tree**  

| Training Set Size |      Training Time      | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
| :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
| 100               |        0.0353           |        0.0020          |     0.8725       |     0.7556      |
| 200               |        0.0347           |        0.0012          |     0.8500       |     0.7556      |
| 300               |        0.0257           |        0.0019          |     0.8423       |     0.7852      |

----

## Choosing the Best Model
In this final section, you will choose from the three supervised learning models the *best* model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F<sub>1</sub> score.

### Question 3 - Chosing the Best Model
*Based on the experiments you performed earlier, in one to two paragraphs, explain to the board of supervisors what single model you chose as the best model. Which model is generally the most appropriate based on the available data, limited resources, cost, and performance?*

**Answer: **

The chosen model for this project is one that implements a __Support Vector Machine__ (SVM) algorithim.

Support Vector Machines can be used for Classification (labeling) or Regression (numeric) predications. Since we are trying to find out if an intervention is necessary or not, we will make use of specific type of SVMs called __Support Vector Classifier__ (SVC) to get the results we are looking for.

The __SVM takes data about past students__ (age, gender, family, etc), and __uses them to create predictions about new student cases.__ These predictions are made by creating a function that draws a boundary between the students who graduated and those who did not. The boundary should be drawn so as to maximize the the space between itself and each of the classifications (graduation results), this space is called a "margin".

![](https://cl.ly/303V2O2H2L2S/Image%202016-08-13%20at%203.38.17%20PM.png "pic_1")

Often, though, it's not easy to draw a decision boundary in low dimensions (i.e. A line is not a good enough boundry), so the SVM separates the passing and failing students by turning the dimensions we're working with from a plane into a higher dimension such as a cube (see below). Once the SVM changes its way of looking at the data, it can then use a plane to seperate the data instead of just a line.

![](https://cl.ly/2x0l1f2r2R2W/Image%202016-08-13%20at%203.38.38%20PM.png "pic_2")


__By using the technique mentioned above, our SVC algorithim predicts if a student will graduate or not with an accuracy of 80% or more.__

----

### Question 4 - Model in Layman's Terms
*In one to two paragraphs, explain to the board of directors in layman's terms how the final model chosen is supposed to work. For example if you've chosen to use a decision tree or a support vector machine, how does the model go about making a prediction?*

**Answer: **

The chosen model is called __Support Vector Machine__ (SVM). This algorithim works by trying to split the data in the cleanest way. This means that SVMs try to draw a line between different outcomes (aka Classifications) from the training data (aka student records available) in order to learn to predict the outcomes of a new student case.

SVMs are a great choice when the resources are limited and the number of considerations (aka Features, ex: `absences`) in our data is relatively large. This is the case with in this project since the number of features is 40+ (after data processing) and the number of entries is 300+ which is relatively low. SVMs still however provide a quick training and prediction times compared to other algorithims that were considered.

### Implementation: Model Tuning
Fine tune the chosen model. Use grid search (`GridSearchCV`) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this. In the code cell below, you will need to implement the following:
- Import [`sklearn.grid_search.gridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html) and [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
- Create a dictionary of parameters you wish to tune for the chosen model.
 - Example: `parameters = {'parameter' : [list of values]}`.
- Initialize the classifier you've chosen and store it in `clf`.
- Create the F<sub>1</sub> scoring function using `make_scorer` and store it in `f1_scorer`.
 - Set the `pos_label` parameter to the correct value!
- Perform grid search on the classifier `clf` using `f1_scorer` as the scoring method, and store it in `grid_obj`.
- Fit the grid search object to the training data (`X_train`, `y_train`), and store it in `grid_obj`.


```python
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

# Create the parameters list you wish to tune
parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid')}

# Initialize the classifier
clf = SVC()

# Make an f1 scoring function using 'make_scorer'
f1_scorer = make_scorer(f1_score, pos_label = 'yes')

# Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf, parameters, f1_scorer)

# Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train)

# Get the estimator
clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))

print "Best params are: {}".format(grid_obj.best_params_)
```

    Made predictions in 0.0047 seconds.
    Tuned model has a training F1 score of 0.8602.
    Made predictions in 0.0017 seconds.
    Tuned model has a testing F1 score of 0.8163.
    Best params are: {'kernel': 'rbf'}


### Question 5 - Final F<sub>1</sub> Score
*What is the final model's F<sub>1</sub> score for training and testing? How does that score compare to the untuned model?*

**Answer: **

The final F1 score using the tuned SVC model is 0.8125 which is a 2 - 3% jump from the initial model.

The best parameters chosen by GridSearch are `{'kernel': 'rbf', 'C': 1, 'gamma': 1}`.

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
