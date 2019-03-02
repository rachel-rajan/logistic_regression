function [theta, J, h] = gradient_descent_logistic_regression(x_norm, y, theta, alpha, num_of_iter )
%%performs the gradient descent operation for a logistic regression problem for ‘n’ iterations
%% Inputs
% theta :includes both theta_0 and so on
% x_norm: normalized feature vector
% y: (actual value)
% alpha : learning rate
% num_of_iter : Number of iterations
%%Outpus
%J : cost at every iteration [Array variable]
%theta : updated weights

%%
% computing the length of our actual vlue
m = length(y);

% Creating a zero matrix to store cost value
J = zeros(num_of_iter, 1);

% Gradient descent
for i = 1:num_of_iter  
    
%hypothesis
h = compute_sigmoid(x_norm*theta);

% Calculating the delta
delta = (1/m)*(h - y)'*x_norm;  

% Updating theta
theta = theta - (delta*alpha)';

% Keeping track of the cost function
J(i) = compute_cost_logistic_regression(theta, x_norm, y);  

end 

end