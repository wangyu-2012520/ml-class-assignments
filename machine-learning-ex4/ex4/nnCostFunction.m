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
y_matrix = eye(num_labels)(y,:);

% add the column of 1’s to the X matrix, to calculate a1
a1 = [ones(size(X,1),1) X];

% calculate a2; add the column of 1's to a2 matrix
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];

% calculate a3
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% calculate cost function
J = -y_matrix.*log(a3) - (1-y_matrix).*log(1-a3);
J = sum(sum(J)) / m;


% Let:
% 
% m = the number of training examples
% 
% n = the number of training features, including the initial bias unit.
% 
% h = the number of units in the hidden layer - NOT including the bias unit
% 
% r = the number of output classifications
% 
% -------------------------------
% 
% 1: Perform forward propagation, see the separate tutorial if necessary.
% 
% 2: Del3 or d3 is the difference between a3 and the y_matrix. The dimensions are the same as both, (m x r).
% 
% 3: z2 came from the forward propagation process - it's the product of a1 and Theta1, 
% prior to applying the sigmoid() function. Dimensions are (m x n)  (n x h) --> (m x h)
% 
% 4: Del2 or d2 is tricky. It uses the (:,2:end) columns of Theta2. d2 is the product of d3 and Theta2(no bias), 
% then element-wise scaled by sigmoid gradient of z2. The size is (m x r)  (r x h) --> (m x h). The size is the same as z2, as must be.
% 
% 5: Delta1 is the product of d2 and a1. The size is (h x m)  (m x n) --> (h x n)
% 
% 6: Delta2 is the product of d3 and a2. The size is (r x m)  (m x [h+1]) --> (r x [h+1])
% 
% 7: Theta1_grad and Theta2_grad are the same size as their respective Deltas, just scaled by 1/m.


Del2 = 0;
Del1 = 0;

temp = Theta2(:,2:end);

Theta1(:,[1]) = 0;
Theta2(:,[1]) = 0;

for i = 1:m,
	delta_3 = a3(i,:) - y_matrix(i,:);
	delta_2 = (delta_3 * temp).* sigmoidGradient(z2(i,:));

	Del2 = Del2 + delta_3' * a2(i,:);
	Del1 = Del1 + delta_2' * a1(i,:);

end;

	Theta2_grad = Del2 / m + lambda / m * Theta2;
	Theta1_grad = Del1 / m + lambda / m * Theta1; 

% get rid of the first column of Theta1 and Theta2, because it should not be regularizing 
% the terms that correspond to the bias
Theta1(:,[1]) = [];
Theta2(:,[1]) = [];


% calculate Regularized cost function, no considering the Theta1 and Theta2 with bias
J = J + lambda / 2 / m * (sum(sum(Theta1.*Theta1)) + sum(sum(Theta2.*Theta2)));




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
