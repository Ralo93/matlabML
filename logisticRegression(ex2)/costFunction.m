function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%
h = X*theta;
hNew = sigmoid(h);
cost = 0;

for ind = 1:m
    cost = cost + (-y(ind)*log(hNew(ind))-(1-y(ind))*log(1-(hNew(ind))));
end

J = cost / m;

#J = 1/m * sum(-y.*(log(hNew))-(1.-y)*log(1.-(hNew)));

for ind2 = 1:size(theta, 1) #3 mal
  
  summe = 0;
  for ind3 = 1:m
    
    summe = summe + (hNew(ind3) - y(ind3)) * X(ind3, ind2);
  endfor
  
  grad(ind2) = 1/m * summe;
end






% =============================================================

end
