function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%Regularization Parameter
regParam = (sum(theta.^2) - theta(1)^2)*lambda;

%Cost of linear regression
J = (sum((X*theta- y).^2) + regParam)/(2*m);

%Calculating Gradients
grad = (X'*(X*theta - y))/m;

regParam = theta*lambda/m;
regParam(1) = 0;

grad = grad + regParam;


% =========================================================================

grad = grad(:);

end
