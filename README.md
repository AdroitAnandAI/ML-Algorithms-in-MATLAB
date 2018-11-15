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

# 4. Neural Networks Learning #

To implement the backpropagation algorithm for neural networks and apply it to the task of hand-written digit recognition.

### Feedforward and Cost Function ###

Lets implement the cost function and gradient for the neural network.

```matlab
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
capDelta1 = zeros(size(Theta1));
capDelta2 = zeros(size(Theta2));

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. 
%
% Part 2: backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. returns the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. Then run checkNNGradients

% Part 3: Implement regularization with the cost function and gradients.

X = [ones(size(X, 1), 1) X];
a1 = X; % to calculate gradient DELTA
z2 = X*Theta1';
a2 = sigmoid(z2);

a2 = [ones(size(a2, 1), 1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);  % a3 is H_theta(x)

decodedY = recode(y, num_labels);

for i=1:m
        J = J - sum((decodedY(i,:).*log(a3(i,:))) +...
                  (1-decodedY(i,:)).*log(1-a3(i,:)))/m;
end;

%Adding the Cost Regularization parameter
%Remove the 1st column to eliminate the bias
J = J + (sum(sum(Theta1(:,2:size(Theta1,2)).^2)) + ...
         sum(sum(Theta2(:,2:size(Theta2,2)).^2)))*lambda/(2*m);

for t = 1:m
    delta3 = a3(t,:) - decodedY(t,:);
    delta2 = (delta3*Theta2(:,2:size(Theta2,2))).*sigmoidGradient(z2(t,:));

    capDelta1 = capDelta1 + delta2'*a1(t,:);
    capDelta2 = capDelta2 + delta3'*a2(t,:);
    
end

Theta1_grad = capDelta1/m; 
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + Theta1(:,2:end)*lambda/m; %To regularize NN
Theta2_grad = capDelta2/m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + Theta2(:,2:end)*lambda/m; %To regularize NN

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

function m = recode(codedVector, k)
    
    t = size(codedVector,1);
%     temp = zeros(t, k+1);
    m = zeros(t, k);
    for i=1:t
%         if codedVector(i)==10
%             m(i,1) = 1;
%         else
            m(i, codedVector(i)) = 1;
%         end
    end

end
```

### Regularized cost function ###

The cost function for neural networks with regularization is given by

<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/4.%20Neural%20Networks%20Learning/images/1.4.PNG">
</p>

### Backpropagation ###

To implement the backpropagation algorithm to compute the gradient for the neural network cost function.

**Sigmoid gradient**

The gradient for the sigmoid function can be computed as

<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/4.%20Neural%20Networks%20Learning/images/2.1.PNG">
</p>

**Random initialization**

When training neural networks, it is important to randomly initialize the parameters for symmetry breaking.

```matlab
function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms

W = zeros(L_out, 1 + L_in); 

% Randomly initialize the weights to small values
epsilon_init = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;

end
```

**Backpropagation**

<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/4.%20Neural%20Networks%20Learning/images/2.3.PNG">
</p>

### Gradient checking ###

In our neural network, you are minimizing the cost function. To perform gradient checking on parameters, "unrolling"
the parameters theta(1) and theta(2) is done into a long vector theta. By doing so, we can think of the cost function being J(O) instead and use the following gradient checking procedure.

<p>
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/4.%20Neural%20Networks%20Learning/images/2.4.PNG">
</p>

```matlab
function checkNNGradients(lambda)
%CHECKNNGRADIENTS Creates a small neural network to check the
%backpropagation gradients
%   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
%   backpropagation gradients, it will output the analytical gradients
%   produced by your backprop code and the numerical gradients (computed
%   using computeNumericalGradient). These two gradient computations should
%   result in very similar values.
%

if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

input_layer_size = 3;
hidden_layer_size = 5;
num_labels = 3;
m = 5;

% We generate some 'random' test data
Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
% Reusing debugInitializeWeights to generate X
X  = debugInitializeWeights(m, input_layer_size - 1);
y  = 1 + mod(1:m, num_labels)';

% Unroll parameters
nn_params = [Theta1(:) ; Theta2(:)];

% Short hand for cost function
costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, ...
                               num_labels, X, y, lambda);

[cost, grad] = costFunc(nn_params);
numgrad = computeNumericalGradient(costFunc, nn_params);

% Visually examine the two gradient computations.  The two columns
% you get should be very similar. 
disp([numgrad grad]);
fprintf(['The above two columns you get should be very similar.\n' ...
         '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);

% Evaluate the norm of the difference between two solutions.  
% If you have a correct implementation, and assuming you used EPSILON = 0.0001 
% in computeNumericalGradient.m, then diff below should be less than 1e-9
diff = norm(numgrad-grad)/norm(numgrad+grad);

fprintf(['If your backpropagation implementation is correct, then \n' ...
         'the relative difference will be small (less than 1e-9). \n' ...
         '\nRelative Difference: %g\n'], diff);

end
```

# 5. Regularized Linear Regression #

Lets implement regularized linear regression and use it to study models with different bias-variance properties. One model here is to predict the amount of water owing out of a dam using the change of water level in a reservoir.

### Visualizing the dataset ###

To visualize the dataset containing historical records on the change in the water level, x, and the amount of water owing out of the dam, y. This dataset is divided into three parts:
• A training set that your model will learn on: X, y
• A cross validation set for determining the regularization parameter: Xval, yval
• A test set for evaluating performance.

```matlab
% Load from ex5data1: 
load ('ex5data1.mat');

% m = Number of examples
m = size(X, 1);

% Plot training data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
```

<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/5.%20Regularized%20Linear%20Regression/images/1.1.PNG">
</p>

### Regularized linear regression cost function ###

Regularized linear regression has the following cost function:

<p>
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/5.%20Regularized%20Linear%20Regression/images/1.2.PNG">
</p>

### Regularized linear regression gradient ###

<p>
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/5.%20Regularized%20Linear%20Regression/images/1.3.PNG">
</p>

```matlab
function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

%Regularization Parameter
regParam = (sum(theta.^2) - theta(1)^2)*lambda;

%Cost of linear regression
J = (sum((X*theta- y).^2) + regParam)/(2*m);

%Calculating Gradients
grad = (X'*(X*theta - y))/m;

regParam = theta*lambda/m;
regParam(1) = 0;

grad = grad + regParam;

grad = grad(:);

end
```
### Fitting linear regression ###

To compute the optimal values of theta, we set regularization parameter to zero. Because our current implementation of linear regression is trying to fit a 2-dimensional theta, regularization will not be incredibly helpful for a theta of such low dimension.

```matlab
%  Train linear regression with lambda = 0
lambda = 1;
[theta] = trainLinearReg([ones(m, 1) X], y, lambda);

%  Plot fit over the data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
```
<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/5.%20Regularized%20Linear%20Regression/images/1.4.PNG">
</p>

### Bias-variance ###

An important concept in machine learning is the bias-variance tradeoff. Models with high bias are not complex enough for the data and tend to underfit, while models with high variance overfit to the training data. We will plot training and test errors on a learning curve to diagnose bias-variance problems.

**Learning curves**

A learning curve plots training and cross validation error as a function of training set size. To plot the learning curve, we need a training and cross validation set error for different training set sizes. To obtain different training set sizes, you should use different subsets of the original training set X. Specifically, for a training set size of i, you should use the first i examples (i.e., X(1:i,:) and y(1:i)).

Training error for a dataset is defined as:
<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/5.%20Regularized%20Linear%20Regression/images/2.1.PNG">
</p>

```matlab
function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

      for i = 1:m
          % Compute train/cross validation errors using training examples 
          % X(1:i, :) and y(1:i), storing the result in 
          % error_train(i) and error_val(i)
          
          for j = 1:i
              theta = trainLinearReg(X(1:i, :), y(1:i), lambda);
              error_train(i) = linearRegCostFunction(X(1:i, :), y(1:i), theta, 0);
              error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
              
          end
          
      end
end
```

<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/5.%20Regularized%20Linear%20Regression/images/2.1.2.PNG">
</p>

We can observe that both the train error and cross validation error are high when the number of training examples is increased. This
reflects a high bias problem in the model - the linear regression model is too simple and is unable to fit our dataset well.

### Polynomial regression ###

The problem with our linear model was that it was too simple for the data and resulted in underfitting (high bias).

<p>
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/5.%20Regularized%20Linear%20Regression/images/3.PNG">
</p>

```matlab
function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%

X_poly = zeros(numel(X), p);

for i = 1:size(X,1)
    for j = 1:p
        X_poly(i, j) = X(i).^j;
    end
end
end
```

### Selecting Lambda using a cross validation set ###

Here, we will implement an automated method to select the lambda parameter. Concretely, we will use a cross validation set to evaluate
how good each lambda value is. After selecting the best lambda value using the cross validation set, we can then evaluate the model on the test set to estimate how well the model will perform on actual unseen data.

```matlab
function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

      for i = 1:length(lambda_vec)
          lambda = lambda_vec(i);
          % Compute train / val errors when training linear 
          % regression with regularization parameter lambda
          % You should store the result in error_train(i)
          % and error_val(i)
          
          theta = trainLinearReg(X, y, lambda);
          error_train(i) = linearRegCostFunction(X, y, theta, 0);
          error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
              
      end
end
```

<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/5.%20Regularized%20Linear%20Regression/images/3.3.PNG">
</p>

# 6. Support Vector Machines #

Lets use support vector machines (SVMs) to build a spam classifier. Experimenting with the given 2D datasets will help you gain an intuition of how SVMs work and how to use a Gaussian kernel with SVMs.

### Plotting 2D dataset 1

We will begin by with a 2D example dataset which can be separated by a linear boundary. In this dataset, the positions of the positive examples (indicated with +) and the negative examples (indicated with o) suggest a natural separation indicated by the gap. However, notice that there is an outlier positive example + on the far left at about (0:1; 4:1).

```matlab
fprintf('Loading and Visualizing Data ...\n')

% Load from ex6data1: 
% You will have X, y in your environment
load('ex6data1.mat');

% Plot training data
plotData(X, y);
```

<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/6.%20Support%20Vector%20Machines/images/1.1.PNG">
</p>

### SVM with Gaussian Kernels ###

We will be using SVMs to do non-linear classification. In particular, we will be using SVMs with Gaussian kernels on datasets that are not linearly separable.

To find non-linear decision boundaries with the SVM, we need to first implement a Gaussian kernel. You can think of the Gaussian kernel as a similarity function that measures the \distance" between a pair of examples, (x(i); x(j)). The Gaussian kernel is also parameterized by a bandwidth parameter, sigma, which determines how fast the similarity metric decreases (to 0) as the examples are further apart.

The Gaussian kernel function is defined as:

<p>
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/6.%20Support%20Vector%20Machines/images/1.2.1.PNG">
</p>

```matlab
function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

sim = exp(-sum((x1-x2).^2)/(2*sigma^2));
    
end
```

### Plotting 2D dataset 2

```matlab
=============== Part 4: Visualizing Dataset 2 ================
%  The following code will load the next dataset into your environment and 
%  plot the data. 
%

fprintf('Loading and Visualizing Data ...\n')

% Load from ex6data2: 
% You will have X, y in your environment
load('ex6data2.mat');

% Plot training data
plotData(X, y);

```

<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/6.%20Support%20Vector%20Machines/images/1.2.2.PNG">
</p>

From the figure, we can observe that there is no linear decision boundary that separates the positive and negative examples for this dataset. However, by using the Gaussian kernel with the SVM, we are able to learn a non-linear decision boundary that can perform reasonably well for the dataset.

### Plotting 2D dataset 3 ###

Lets use a SVM with a Gaussian kernel. Our task is to use the cross validation set Xval, yval to determine the best C and sigma parameter to use.

```matlab
function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

steps = [0.01 0.03 0.1 0.3 1 3 10 30];

minError = 1000;

for i = 1:numel(steps)
    testC = steps(i);
    for j=1:numel(steps)
        testSigma = steps(j);
        % Train the SVM
        model= svmTrain(X, y, testC, @(x1, x2) gaussianKernel(x1, x2, testSigma));
        predictions = svmPredict(model, Xval);
        meanError = mean(double(predictions ~= yval));
        
        if (meanError < minError)
            minError = meanError;
            C = testC;
            sigma = testSigma;
            fprintf('MeanError = %f', meanError);
        end
            
    end
end

end
```

**Below is the figure generated after plotting the dataset.**

<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/6.%20Support%20Vector%20Machines/images/1.2.3.1.PNG"> 
</p>

**SVM (Gaussian Kernel) Decision Boundary for the plotted dataset:**

<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/6.%20Support%20Vector%20Machines/images/1.2.3.2.PNG">
</p>

### Spam Classification ###

Many email services today provide spam lters that are able to classify emails into spam and non-spam email with high accuracy. In this part of the exercise, we will use SVMs to build your own spam filter.

Given the vocabulary list, we can now map each word in the preprocessed emails into a list of word indices that contains the index of the word in the vocabulary list. Specifically, in a sample email, the word "anyone" was first normalized to "anyon" and then mapped onto the index in the vocabulary list.

```matlab
function word_indices = processEmail(email_contents)
%PROCESSEMAIL preprocesses a the body of an email and
%returns a list of word_indices 
%   word_indices = PROCESSEMAIL(email_contents) preprocesses 
%   the body of an email and returns a list of indices of the 
%   words contained in the email. 
%

% Load Vocabulary
vocabList = getVocabList();

% Init return value
word_indices = [];

% ========================== Preprocess Email ===========================

% Find the Headers ( \n\n and remove )
% Uncomment the following lines if you are working with raw emails with the
% full headers

% hdrstart = strfind(email_contents, ([char(10) char(10)]));
% email_contents = email_contents(hdrstart(1):end);

% Lower case
email_contents = lower(email_contents);

% Strip all HTML
% Looks for any expression that starts with < and ends with > and replace
% and does not have any < or > in the tag it with a space
email_contents = regexprep(email_contents, '<[^<>]+>', ' ');

% Handle Numbers
% Look for one or more characters between 0-9
email_contents = regexprep(email_contents, '[0-9]+', 'number');

% Handle URLS
% Look for strings starting with http:// or https://
email_contents = regexprep(email_contents, ...
                           '(http|https)://[^\s]*', 'httpaddr');

% Handle Email Addresses
% Look for strings with @ in the middle
email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr');

% Handle $ sign
email_contents = regexprep(email_contents, '[$]+', 'dollar');


% ========================== Tokenize Email ===========================

% Output the email to screen as well
fprintf('\n==== Processed Email ====\n\n');

% Process file
l = 0;

while ~isempty(email_contents)

    % Tokenize and also get rid of any punctuation
    [str, email_contents] = ...
       strtok(email_contents, ...
              [' @$/#.-:&*+=[]?!(){},''">_<;%' char(10) char(13)]);
   
    % Remove any non alphanumeric characters
    str = regexprep(str, '[^a-zA-Z0-9]', '');

    % Stem the word 
    % (the porterStemmer sometimes has issues, so we use a try catch block)
    try str = porterStemmer(strtrim(str)); 
    catch str = ''; continue;
    end;

    % Skip the word if it is too short
    if length(str) < 1
       continue;
    end

    % Look up the word in the dictionary and add to word_indices if
    % found
   
    vocabLength = numel(vocabList);
    
    for i = 1:vocabLength
        if (strcmp(str, vocabList(i)) == 1)
            word_indices = [word_indices ; i];
        end
    end
   
     % Print to screen, ensuring that the output lines are not too long
    if (l + length(str) + 1) > 78
        fprintf('\n');
        l = 0;
    end
    fprintf('%s ', str);
    l = l + length(str) + 1;

end

end
```

### Extracting Features from Emails ###

We will now implement the feature extraction that converts each email into a vector. for a typical email, features would look like:

<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/6.%20Support%20Vector%20Machines/images/2.2.PNG">
</p>

```matlab
function x = emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector
%from the word indices
%   x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
%   produces a feature vector from the word indices. 

% Total number of words in the dictionary
n = 1899;

% You need to return the following variables correctly.
x = zeros(n, 1);

%               Concretely, if an email has the text:
%
%                  The quick brown fox jumped over the lazy dog.
%
%               Then, the word_indices vector for this text might look 
%               like:
%               
%                   60  100   33   44   10     53  60  58   5
%
%               where, we have mapped each word onto a number, for example:
%
%                   the   -- 60
%                   quick -- 100
%                   ...
%
%              (note: the above numbers are just an example and are not the
%               actual mappings).
%
%              Our task is take one such word_indices vector and construct
%              a binary feature vector that indicates whether a particular
%              word occurs in the email. That is, x(i) = 1 when word i
%              is present in the email. Concretely, if the word 'the' (say,
%              index 60) appears in the email, then x(60) = 1. The feature
%              vector should look like:
%
%              x = [ 0 0 0 0 1 0 0 0 ... 0 0 0 0 1 ... 0 0 0 1 0 ..];
%
%

emailLen = numel(word_indices);
for i = 1:emailLen
    x(word_indices(i)) = 1;
end

end
```

# 7. K Means and PCA #

Lets implement the K-means clustering algorithm and apply it to compress an image followed by principal component analysis to find a low-dimensional representation of face images.

### Finding closest centroids ###

<p>
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/7.%20K%20Means%20and%20PCA/images/1.1.1.PNG">
</p>

```matlab
function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);
m = size(X,1);

% You need to return the following variables correctly.
idx = zeros(m, 1);

for i = 1:m
    dist = 100000000;
    for j = 1:K
        if (sum((X(i,:) - centroids(j,:)).^2) < dist)
            dist = sum((X(i,:) - centroids(j,:)).^2);
            idx(i) = j;
        end
    end
end

end
```

### Computing centroid means ###

<p>
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/7.%20K%20Means%20and%20PCA/images/1.1.2.PNG">
</p>

```matlab
function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);

for i = 1:K
    centroids (i,:) = mean(X(idx == i, :));
end

end
```

**K-means on example dataset**

<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/7.%20K%20Means%20and%20PCA/images/1.2.PNG">
</p>

### Implementing PCA ###

PCA consists of two computational steps: 
**-compute the covariance matrix of the data 
**-compute the eigenvectors U1; U2; ; ; ;Un. 

These will correspond to the principal components of variation in the data.

Before using PCA, it is important to rst normalize the data by subtracting the mean value of each feature from the dataset, and scaling each dimension so that they are in the same range.

**To compute the covariance matrix of the data

<p>
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/7.%20K%20Means%20and%20PCA/images/2.2.PNG">
</p>

```matlab
function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

sigma = X'*X/m;

[U, S, V] = svd(sigma);

end
```

**Computed eigenvectors of the dataset**

<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/7.%20K%20Means%20and%20PCA/images/2.2.1.PNG">
</p>

### Dimensionality Reduction with PCA ###

After computing the principal components, we can use them to reduce the feature dimension of your dataset by projecting each example onto a lower dimensional space, x(i) ! z(i) (e.g., projecting the data from 2D to 1D). Here, we will use the eigenvectors returned by PCA and project the example dataset into a 1-dimensional space.

**Projecting the data onto the principal components**

Specifically, we are given a dataset X, the principal components U, and the desired number of dimensions to reduce to K. You should project each example in X onto the top K components in U. Note that the top K components in U are given by the first K columns of U, that is U reduce = U(:, 1:K).

```matlab
function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% need to return the following variables correctly.
Z = zeros(size(X, 1), K);

m = size(X, 1);

%     for i = 1:m
%         x = X(i, :)';
%         Z(i,:) = x' * U(:, K);
%     end
        

Z = X*U(:, 1:K);

end
```

**Reconstructing an approximation of the data**

After projecting the data onto the lower dimensional space, we can approximately recover the data by projecting them back onto the original high dimensional space.

```matlab
function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data
%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

% need to return the following variables correctly.
X_rec = zeros(size(Z, 1), size(U, 1));

m = size(Z, 1);

%               Notice that U(j, 1:K) is a row vector.
%               

%     for i = 1:m
%         v = Z(i, :)';
%         recovered_j = v' * U(:, 1:K)';
%     end
    
    X_rec = Z * U(:, 1:K)';

end
```

**Visualizing the projections**

```matlab
%  Visualization of Faces after PCA Dimension Reduction
%  Project images to the eigen space using the top K eigen vectors and 
%  visualize only using those K dimensions
%  Compare to the original input, which is also displayed

fprintf('\nVisualizing the projected (reduced dimension) faces.\n\n');

K = 100;
X_rec  = recoverData(Z, U, K);

% Display normalized data
subplot(1, 2, 1);
displayData(X_norm(1:100,:));
title('Original faces');
axis square;

% Display reconstructed data from only k eigenfaces
subplot(1, 2, 2);
displayData(X_rec(1:100,:));
title('Recovered faces');
axis square;

```

**The normalized and projected data after PCA.**

<p align="center">
    <img src="https://github.com/AdroitAnandAI/ML-Algorithms-in-MATLAB/blob/master/7.%20K%20Means%20and%20PCA/images/2.3.PNG">
</p>

