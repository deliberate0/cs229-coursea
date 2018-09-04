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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% num_layers = size(nn_params,1)+1;

% add ones to all X
X = [ones(m,1),X];
output = zeros(num_labels,m);
% set up useful var
cost = zeros(m,1);
Delta1 = zeros(size(Theta1)); 
Delta2 = zeros(size(Theta2));

for t = 1:m
    % transform scale y to vector
    output(y(t),t)=1;
    
    % compute cost(t) without reg
    
    alpha1 = X(t,:)';
    z2 = Theta1*alpha1;
    alpha2 = sigmoid(z2);
    % add ones to layer 2, so as to facilite bp; all the layer of input
    % should be this so as to be a bias 
    alpha2 = [1;alpha2];
    z3 = Theta2*alpha2;
    alpha3 = sigmoid(z3);

    cost(t) = output(:,t)'*log(alpha3) + (1-output(:,t))'*log(1-alpha3);
    
    % compute gradient without reg;
    delta3 = alpha3 - output(:,t);
    % 1. note that delta2 , we should remove delta(1), for we don't expect 
    %    to change bias
    % 2. we add +1 to z2 just for matricalculation, we don't really want to
    %    change bias, so we remove the delta2(1) after.
    delta2 = (Theta2'*delta3) .* sigmoidGradient([1;z2]);
    delta2 = delta2(2:end);
%   Delta works as accumulator for all (x(t),y(t))
    Delta1 = Delta1 + delta2*alpha1';
    Delta2 = Delta2 + delta3*alpha2';
    
end

% J without regulation
J = -1/m * sum(cost);

% regulation term ; not penalise theta(:,0)

% reg = m/(2*lambda) * (Theta1(:)'*Theta1(:)+ Theta2(:)'*Theta2(:) ...
%     -Theta1(:,1)'*Theta1(:,1) - Theta2(:,1)'*Theta2(:,1));

% how to rectify this block??????????????
col_sum1 = sum(Theta1.^2,1);
col_sum2 = sum(Theta2.^2,1);
reg = lambda/(2*m) * sum([col_sum1(2:end),col_sum2(2:end)]);

% J with regulation
J = J + reg;

% Gradient with regulation
theta1_for_reg = Theta1;
theta1_for_reg(:,1) = 0;
Theta1_grad= 1/m * (Delta1 + lambda*theta1_for_reg);
theta2_for_reg = Theta2;
theta2_for_reg(:,1) = 0;
Theta2_grad= 1/m * (Delta2 + lambda*theta2_for_reg);
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
