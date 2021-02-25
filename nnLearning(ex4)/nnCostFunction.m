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
%fprintf('THETA2 a3: %f ', size(Theta2));
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
y_matrix = eye(num_labels)(y,:);



h = zeros(size(m));

% K = 10 with altering y loop
  
  % Forward Propagation
  a1 = [ones(m, 1), X];
  %fprintf('SIZE a1: %f ', size(a1));
  z2 = a1*Theta1'; %result
  a2 = sigmoid(z2);
  a2 = [ones(m,1), a2];  %%LOOPUP size
  %fprintf('SIZE a2: %f ', size(a2));
  z3 = a2*Theta2'; %result
  a3 = sigmoid(z3); % equal to h(parameterized by theta) %%%LOOKUP size, use it for computing a-y = delta
  %fprintf('SIZE a3: %f ', size(a3));
  
  

J = (1/m) * sum(sum(-y_matrix .* log(a3) - (1-y_matrix) .* log(1-a3),2));
%In order to the honor codex, I hereby mark this following line of code as extracted from stackoverflow!
%It computes exactly the same as reg below (Which I have implemented by myself), but this implementation works.
J = J + (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2,2)) + sum(sum(Theta2(:,2:end).^2,2)));

%reg = (lambda / (2*m))* (sum(sum(Theta1(:,2:end).^2,2)) + sum(sum(Theta2(:,2:end).*2,2)));


% use m, y_matrix and a3

%summed = sum((-y.*log(a3) - (1-y).*log(1-a3))) + lambda/(2*m)* sum(tmp.^2);

%tmp1 = Theta1;
%tmp1(1) = 0;

%tmp2 = Theta2;
%tmp2(1) = 0;


%a3 = h

%tmp = theta;
%tmp(1) = 0;


%J = 1/m*sum((-y.*log(a3) - (1-y).*log(1-a3))) + lambda/(2*m)* sum(tmp.^2);

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


% m: number of examples
% n: number of training features including the bias unit
% h: number of units in the hiffen layer, without the bias unit
% r = number of output classifications
% y_matrix = 5000x10
% a3 nach FFP: 5000x10


 
 

 %groesse von 400x10 für die deltas output layer
 
  a1 = [ones(m, 1), X];
  %fprintf('SIZE a1: %f ', size(a1));
  z2 = a1*Theta1'; % should be mxh 5000x25
  a2 = sigmoid(z2);
  a2 = [ones(m,1), a2];  %%LOOPUP size
  %fprintf('SIZE a2: %f ', size(a2));
  z3 = a2*Theta2'; % should be 
  a3 = sigmoid(z3);
  %fprintf('SIZE a3: %f ', size(a3));
  
  d3 = a3-y_matrix; %5000x10
  
  %fprintf('D3 SIZE a3: %f ', size(d3));
  tmp = Theta2(:,2:end)'*d3';
  d2 = tmp.*sigmoidGradient(z2)';
  %d2 = Theta2(:,2:end)'*d3.*sigmoidGradient(z2);  %10x26 = 10x25 transp = 25x10
  %fprintf('VECTOR: %f ', yVector);
  %d2 = d2';
  %5000x10 is a3
  %fprintf('SIZE z2 equal to size d2?: %f  %f', size(z2), size(d2));
  
  d3 = d3';
  
  delta1 = d2*a1; %BIG DELTAS
  delta2 = d3*a2;
  
  Theta1_grad = (1/m)*delta1; %small deltas
  Theta2_grad = (1/m)*delta2;

  %%DONE WITH FP AND BP wihtout regularization, we can now add regularization 
  %%AFTER we have computed BP completely
  
  Theta1(:,1) = 0;
  Theta2(:,1) = 0;
  
  scaledT1 = (lambda/m)*Theta1;
  scaledT2 = (lambda/m)*Theta2;
  
  Theta1_grad = Theta1_grad + scaledT1;
  Theta2_grad = Theta2_grad + scaledT2;









% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
