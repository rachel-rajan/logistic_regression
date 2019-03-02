function J = compute_cost_logistic_regression(theta, x_norm, y)
%computes the logistic regression error between the actual value of y and 
%estimated value of y based on the hypothesis
%%Inputs 
%theta :  (includes both theta_0 and theta_1) [Array vector theta (1) = theta_0 and theta (2) = theta_1]
% x_norm:normalized feature vector
% y :actual value
%%Outputs
% J :cost

% number of training examples
m = length(x_norm); 
%hypothesis
h = compute_sigmoid(x_norm*theta);
% Compute cost function
J  = (-1/m) * sum( y .* log(h) + (1 - y) .* log(1 - h));

end