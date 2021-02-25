function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y); 
J_history = zeros(num_iters, 1);

for iter = 1:num_iters


    Theta = theta;
    
    for feat = 1:length(theta)
      
      theta(feat,1) = Theta(feat,1) - alpha/m*sum((X*Theta-y).*X(:,feat));
      
    end
  
    J_history(iter) = computeCost(X, y, theta);

end

end
