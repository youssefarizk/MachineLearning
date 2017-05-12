function [w] = regresPercep(X,Y,n)

w = pinv(X)*Y'; %Weights vector initialization

for i = 1:n %setting a defined number of iterations
    
    for ii = 1 : size(X,1)        
        
        if sign(X(ii,:)*w) ~= Y(ii) %Check if the classification is right
            w = w + X(ii,:)' * Y(ii);   %then add (or subtract) this point to w
        end
        
end
  
end