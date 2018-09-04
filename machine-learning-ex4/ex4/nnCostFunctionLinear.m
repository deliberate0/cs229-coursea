function [J, grad]=nnCostFunctionLinear(nn_params,...
                                        input_layer_size,...
                                        hidden_layer_size,...
                                        X, y, lambda)
%NNCOSTFUNCTIONLINEAR implete nn with linear output
%just for example
%

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 1, (hidden_layer_size + 1));
             
 % Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


cost = zeros(m,1);
Delta1 = zeros(size(Theta1)); 
Delta2 = zeros(size(Theta2));

X = [ones(m,1),X];
for t = 1:m
    alpha1 = X(t,:)';
    z2 = Theta1*alpha1;
    alpha2 = tanh(z2);
    alpha2 = [1;alpha2];
    z3 = Theta2*alpha2;
    alpha3 = z3;
    % compute cost
    cost(t) = (y(t)-alpha3).^2;
    
    %compute grad
    delta3 = alpha3 - y(t);
    delta2 = (Theta2'*delta3) .* (1-tanh([1;z2]).^2);
    delta2 = delta2(2:end);
%   Delta works as accumulator for all (x(t),y(t))
    Delta1 = Delta1 + delta2*alpha1';
    Delta2 = Delta2 + delta3*alpha2';
    
end
J = 1/m * sum(cost);
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
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
