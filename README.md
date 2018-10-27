# 1. Linear Regression #

A restaurant franchise is considering different cities for opening a new outlet. The chain already has trucks in various cities and we have data for profits and populations from the cities. 

Lets use this data to determine which city to expand to next outlet. The first column is the population of a city and the second column is the profit of a food truck in that city. A negative value for profit indicates a loss.

Here, Linear Regression with one variable is Implemented to predict profits for a food truck in new outlet. 

### Plotting the Data

```matlab
function plotData(x, y)
%PLOTDATA Plots the data points x and y into a new figure 
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   population and profit.

figure; % open a new figure window

data = load('ex1data1.txt'); % read comma separated data
x = data(:, 1); y = data(:, 2);
m = length(y);

plot(x, y, 'rx', 'MarkerSize', 10); % Plot the data
ylabel('Profit in $10,000s'); % Set the y??axis label
xlabel('Population of City in 10,000s'); % Set the x??axis label
end
```
<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/1.%20Linear%20Regression/images/1.2.1.PNG">
</p>

### Gradient Descent ###

Here, we fit the Linear Regression parameters to our dataset using Gradient Descent.

<p>
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/1.%20Linear%20Regression/images/2.2.1.PNG">
</p>

**Lets perform gradient descent to minimize the cost function, it is helpful to monitor the convergence by computing the cost.**

```matlab
function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
```
> ` `**` % Compute the cost of a particular choice of theta and set to J `**` `<br/> 
> ` `**` J = sum((X*theta- y).^2)/(2*m); `**` `
```matlab
end
```

Next, we will implement gradient descent. Here, we minimize the value of the cost function by changing the values of the vector 'theta'. A good way to verify that gradient descent is working correctly is to look at the value of cost function and check that it is decreasing with each step.

```matlab
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    %  Perform a single gradient step on the parameter vector
    %               theta. 

    %%This is non vectorized implementation of gradient descent
%     a = theta(1) - X(:,1)'*(X*theta- y)*(alpha/m);
%     b = theta(2) - X(:,2)'*(X*theta- y)*(alpha/m);
%     theta(1) = a;
%     theta(2) = b;
    
    %%This is vectorized implementation of gradient descent
```

>  ` `**` theta = theta - X'*(X*theta- y)*(alpha/m); `**` `
    
>    ` `**` % Save the cost J in every iteration     `**` ` <br/>
>    ` `**` J_history(iter) = computeCost(X, y, theta); `**` `
```matlab
    %%fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J_history(iter));

end

end
```

<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/1.%20Linear%20Regression/images/2.2.4.PNG">
</p>

### Visualizing Cost Function ###

To understand the cost function better, we will now plot the cost over a 2-dimensional grid.

```matlab
% initialize J vals to a matrix of 0's
J vals = zeros(length(theta0 vals), length(theta1 vals));
% Fill out J vals
for i = 1:length(theta0 vals)
for j = 1:length(theta1 vals)
t = [theta0 vals(i); theta1 vals(j)];
J vals(i,j) = computeCost(x, y, t);
end
end
```
<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/1.%20Linear%20Regression/images/2.4.PNG">
</p>

# 2. Logistic Regression #

Here, we will build a logistic regression model to predict whether a student gets admitted into a university. Suppose the administrator of a university department wants to determine each applicant's chance of admission based on their results on two exams, we have historical data from previous applicants that can be used as a training set for logistic regression. For each training example, we have the applicant's scores on two exams and the admissions decision.

**Our task is to build a classification model that estimates an applicant's probability of admission based the scores from those two exams.**

### Visualizing the data

Code to load the data and display it on a 2-dimensional plot:

```matlab
function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% Find Indices of Positive and Negative Examples
pos = find(y==1); neg = find(y == 0);
% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, ...'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...'MarkerSize', 7);

hold off;

end
```
**Scatter plot of training data**
<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/2.%20Logistic%20Regression/images/1.1.PNG">
</p>

## Logistic Regression Hypothesis
<p>
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/2.%20Logistic%20Regression/images/1.2.1.PNG">
</p>

```matlab
function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

g = zeros(size(z));

dim = size(z);

for i=1:dim(1)
    for j=1:dim(2)
        g(i,j) = 1/(1+exp(-z(i,j)));
    end;
end;
end
```

### Cost function and gradient

<p>
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/2.%20Logistic%20Regression/images/1.2.2.1.PNG">
</p>
<p>
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/2.%20Logistic%20Regression/images/1.2.2.2.PNG">
</p>

```matlab
function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
```
>  ` `**`  grad = zeros(size(theta));  `**` ` <br/>

>  ` `**`  J = -sum((y.*log(sigmoid(X*theta))+(1-y).*log(1-sigmoid(X*theta))))/m;  `**` ` <br/>

>  ` `**`  grad = X'*(sigmoid(X*theta) - y)/m;  `**` ` <br/>
```matlab
end
```

### Evaluating logistic regression

We can use the model to predict whether a particular student will be admitted. For a student with an Exam 1 score of 45 and an Exam 2 score of 85, we should expect to see an admission probability of 0.776.
 
Another way to evaluate the quality of the parameters we have found is to see how well the learned model predicts on our training set. The predict function will produce "1" or "0" predictions given a dataset and a learned parameter vector theta.

```matlab
function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
```
>  ` `**`  m = size(X, 1); % Number of training examples  `**` ` <br/>

>  ` `**`  p = zeros(m, 1);  `**` ` <br/>

>  ` `**`  p = sigmoid(X*theta) >= 0.5;  `**` ` <br/>
```matlab
end
```

**Training data with decision boundary**
<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/2.%20Logistic%20Regression/images/1.2.4.PNG">
</p>

## Regularized logistic regression ##

Lets implement regularized logistic regression to predict whether microchips from a fabrication plant passes quality assurance (QA), to ensure it is functioning correctly. To determine whether the microchips should be accepted or rejected, we have a dataset of test results on past microchips, from which we can build a logistic regression model.

### Visualizing the data ###

```matlab
function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% Find Indices of Positive and Negative Examples
pos = find(y==1); neg = find(y == 0);
% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, ...'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...'MarkerSize', 7);

hold off;

end
```

**Plot of Training Data**
<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/2.%20Logistic%20Regression/images/2.1.PNG">
</p>
The plot shows that our dataset cannot be separated into positive and negative examples by a straight-line through the plot. Therefore, a straightforward application of logistic regression will not perform well on this dataset since logistic regression will only be able to find a linear decision boundary.

### Cost function and gradient ###

**regularized cost function in logistic regression**
<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/2.%20Logistic%20Regression/images/2.3.1.PNG">
</p>
<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/2.%20Logistic%20Regression/images/2.3.2.PNG">
</p>
<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/2.%20Logistic%20Regression/images/2.3.3.PNG">
</p>

```matlab
function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
```
>  ` `**` J = 0;  `**` ` <br/>
>  ` `**` grad = zeros(size(theta));  `**` ` <br/>

>  ` `**` J = -sum((y.*log(sigmoid(X*theta))+(1-y).*log(1-sigmoid(X*theta))))/m + (sum(theta.^2) - theta(1)^2)*lambda/(2*m);  `**` ` <br/>

>  ` `**` reg_param = theta*(lambda/m);  `**` ` <br/>
>  ` `**` reg_param(1) = 0;  `**` ` <br/>

>  ` `**` grad = X'*(sigmoid(X*theta) - y)/m + reg_param;  `**` ` <br/>
```matlab
end
```

# 3. Multiclass Classification and Neural Nets #

Implementation of one-vs-all logistic regression and neural networks to recognize hand-written digits (from 0 to 9). 

### Dataset Details

There are 5000 training examples in the dataset, where each training example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is represented by a oating point number indicating the grayscale intensity at that location. The 20 by 20 grid of pixels is "unrolled" into a 400-dimensional vector. Each of these training examples becomes a single row in our datamatrix X. This gives us a 5000 by 400 matrix X where every row is a training example for a handwritten digit image.

### Visualizing the data

The below function maps each row to a 20 pixel by 20 pixel grayscale image and displays the images together.

<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/3.%20Multiclass%20Classification%20and%20Neural%20Nets/images/1.2.PNG">
</p>
<br/>

### Vectorizing Logistic Regression ###

Lets use multiple one-vs-all logistic regression models to build a multi-class classifier. Since there are 10 classes, there is a need to train 10 separate logistic regression classifiers. To make this training efficient, it is important to ensure that our code is well vectorized. Here, we are implementing a vectorized version of logistic regression that does not employ any for loops. 
<br/>

**Vectorizing the cost function**

<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/3.%20Multiclass%20Classification%20and%20Neural%20Nets/images/1.3.1.1.PNG">
</p>
<br/>

**Vectorizing regularized logistic regression**

The gradient of the (unregularized) logistic regression cost is a vector where the jth element is defined as:

<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/3.%20Multiclass%20Classification%20and%20Neural%20Nets/images/1.3.2.1.PNG">
</p>

Lets add regularization to the cost function, which is defined as:

<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/3.%20Multiclass%20Classification%20and%20Neural%20Nets/images/1.3.3.PNG">
</p>
<br/>

**Code for the vectorizing the cost function, gradient and to account for regularization.**

```matlab
function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
```
> ` `**` grad = zeros(size(theta)); `**` ` <br/>

> ` `**` J = -sum((y.*log(sigmoid(X*theta))+(1-y).*log(1-sigmoid(X*theta))))/m + (sum(theta.^2) - theta(1)^2)*lambda/(2*m); `**` ` <br/>

> ` `**` reg_param = theta*(lambda/m);   `**` ` <br/>

> ` `**` reg_param(1) = 0;  `**` ` <br/>

> ` `**` grad = X'*(sigmoid(X*theta) - y)/m + reg_param;  `**` ` <br/>

> ` `**` grad = grad(:);  `**` ` <br/>
```matlab
end
```

### One-vs-all Classification

Lets implement one-vs-all classification by training multiple regularized logistic regression classifiers, one for each of the K classes in our dataset. In the handwritten digits dataset, K = 10, but our code will work for any value of K.

```matlab
function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

    % Set Initial theta
    initial_theta = zeros(n + 1, 1);
    
    % Set options for fminunc
    options = optimset('GradObj', 'on', 'MaxIter', 50);

    % Run fmincg to obtain the optimal theta
    % This function will return theta and the cost 
    for c = 1:num_labels
```
> ` `**`  theta] = ...
            fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                    initial_theta, options); `**` ` <br/>
               
> ` `**`  all_theta(c,:) = theta;        `**` ` <br/>
                
```matlab
    end;
end
```

## Neural Networks ##

The neural network will be able to represent complex models that form non-linear hypotheses unlike logistic regression. Hence, lets implement a neural network to recognize handwritten digits. Our neural network has 3 layers & Theta1 and Theta2 parameters have dimensions that are sized for a neural network with 25 units in the second layer and 10 output units (corresponding to the 10 digit classes).

<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/3.%20Multiclass%20Classification%20and%20Neural%20Nets/images/2.1.PNG">
</p>

### Feedforward Propagation and Prediction ###

```matlab
function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

X = [ones(size(X, 1), 1) X];
z2 = X*Theta1';
a2 = sigmoid(z2);

a2 = [ones(size(a2, 1), 1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);

[M p] = max(a3, [], 2);

end
```

