function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

rows = size(z, 1);
coloums = size(z, 2);

for ind = 1:rows
  for ind2 = 1:coloums
    g(ind, ind2) = 1 / (1 + exp(-z(ind, ind2)));
  endfor
end



% =============================================================

end
