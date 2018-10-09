function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %%This is non vectorized implementation of gradient descent
%     a = theta(1) - X(:,1)'*(X*theta- y)*(alpha/m);
%     b = theta(2) - X(:,2)'*(X*theta- y)*(alpha/m);
%     theta(1) = a;
%     theta(2) = b;
    
    %%This is vectorized implementation of gradient descent
    theta = theta - X'*(X*theta- y)*(alpha/m);
    
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    %%fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J_history(iter));

end

end
