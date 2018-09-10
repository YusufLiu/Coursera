import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
    import matplotlib.pyplot as plt
    %matplotlib notebook
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);


# NOTE: Uncomment the function below to visualize the data, but be sure
# to **re-comment it before submitting this assignment to the autograder**.
#part1_scatter()

Question 1
Write a function that fits a polynomial LinearRegression model on the training data X_train for degrees 1, 3, 6, and 9. (Use PolynomialFeatures in sklearn.preprocessing to create the polynomial features and then fit a linear regression model) For each model, find 100 predicted values over the interval x = 0 to 10 (e.g. np.linspace(0,10,100)) and store this in a numpy array. The first row of this array should correspond to the output from the model trained on degree 1, the second row degree 3, the third row degree 6, and the fourth row degree 9.

def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    res = []

    # Your code here
    for i in [1,3,6,9]:
        poly = PolynomialFeatures(degree =i)

        X_F1_poly = poly.fit_transform(X_train.reshape(11,1))
        linreg = LinearRegression().fit(X_F1_poly, y_train)
        x_Test = np.linspace(0,10,100).reshape(100,1)
        X_T1_poly = poly.fit_transform(x_Test)
        y_predict = linreg.predict(X_T1_poly)
        res.append(y_predict.flatten())


    result = np.array(res)
    return result

# feel free to use the function plot_one() to replicate the figure
# from the prompt once you have completed question one
def plot_one(degree_predictions):
    import matplotlib.pyplot as plt
    %matplotlib notebook
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)

#plot_one(answer_one())

Question 2
Write a function that fits a polynomial LinearRegression model on the training data X_train for degrees 0 through 9. For each model compute the  R2R2 (coefficient of determination) regression score on the training data as well as the the test data, and return both of these arrays in a tuple.

This function should return one tuple of numpy arrays (r2_train, r2_test). Both arrays should have shape (10,)

def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score

    r2_train = []
    r2_test = []


    # Your code here
    for i in range(10):
        poly = PolynomialFeatures(degree =i)

        X_F1_poly = poly.fit_transform(X_train.reshape(11,1))
        linreg = LinearRegression().fit(X_F1_poly, y_train)
        X_T1_poly = poly.fit_transform(X_test.reshape(4,1))
        r2_train.append(linreg.score(X_F1_poly, y_train))
        r2_test.append(linreg.score(X_T1_poly, y_test))


    result = (r2_train,r2_test)

    return result

Question 4
Training models on high degree polynomial features can result in overly complex models that overfit, so we often use regularized versions of the model to constrain model complexity, as we saw with Ridge and Lasso linear regression.

For this question, train two models: a non-regularized LinearRegression model (default parameters) and a regularized Lasso Regression model (with parameters alpha=0.01, max_iter=10000) both on polynomial features of degree 12. Return the  R2R2 score for both the LinearRegression and Lasso model's test sets.

This function should return one tuple (LinearRegression_R2_test_score, Lasso_R2_test_score)

def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score

    r2_train = []
    r2_test = []


    # Your code here

    poly = PolynomialFeatures(degree =12)

    X_F1_poly = poly.fit_transform(X_train.reshape(11,1))
    linreg = LinearRegression().fit(X_F1_poly, y_train)
    X_T1_poly = poly.fit_transform(X_test.reshape(4,1))
    linlasso = Lasso(alpha=0.01,max_iter=10000).fit(X_F1_poly, y_train)
    r2_lasso= linlasso.score(X_T1_poly, y_test)
    r2_lr = linreg.score(X_T1_poly, y_test)

    result = (r2_lr,r2_lasso)


    return result

Classification

The data in the mushrooms dataset is currently encoded with strings. These values will need to be encoded to numeric to work with sklearn. We'll use pd.get_dummies to convert the categorical variables into indicator variables.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2


def answer_five():
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(random_state = 0).fit(X_train2, y_train2)
    sortR = np.sort(clf.feature_importances_)
    sortvalue = [sortR[-1],sortR[-2],sortR[-3],sortR[-4],sortR[-5]]
    result = []
    for i in sortvalue:
        z = np.where(clf.feature_importances_ == i)
        result.append(X_train2.columns[z])
    # Your code here

    return result
