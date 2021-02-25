function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h = X*theta;
hNew = sigmoid(h);
cost = 0;

for ind = 1:m
    cost = cost + (-y(ind)*log(hNew(ind))-(1-y(ind))*log(1-(hNew(ind))));
end

secondTerm = 0;

#does not regularize the first theta
for ind2 = 2:size(theta, 1)
  
  secondTerm = secondTerm + theta(ind2)^2;
  
end

secondTerm = secondTerm * lambda /(2*m);

J = cost / m + secondTerm;

#Gradients
zeroSum = 0;

for ind0 = 1:m
  
  zeroSum += (hNew(ind0) - y(ind0)) * X(ind0, 1);
  
end

grad(1) = 1/m*zeroSum;


for ind3 = 2:size(theta, 1) 
  
  summe = 0;
  for ind4 = 1:m
    
    summe = summe + (hNew(ind4) - y(ind4)) * X(ind4, ind3);
  endfor
  
  grad(ind3) = 1/m * summe + lambda/m*theta(ind3);
  
end













% =============================================================

end
