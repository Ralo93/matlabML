function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

a1 = [ones(m, 1), X];
%fprintf('X Size: %f ', size(X));
% a1 5000x401
% Theta1 25x401
% Theta2 10x26
z2 = a1*Theta1';

% now we have a 5000x401 times 501x25 = 5000x25


a2 = sigmoid(z2);


a2 = [ones(m,1), a2]; %now has 5000x26


z3 = a2*Theta2';

% now we have a 5000x26 times 26x10 = 5000x10

a3 = sigmoid(z3);


for ex = 1:m
  
  [tmpmax, indx] = max(a3(ex,:), [], 2);
  
    p(ex) = indx;
  
  %fprintf(' %f ', p(ex));
  
end


% =========================================================================


end
