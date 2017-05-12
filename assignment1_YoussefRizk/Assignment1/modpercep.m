function [w] = modpercep(X,Y,n)

w = zeros(1,size(X,2)); %Weights vector initialization
error = 1;

for i = 1:n %setting a defined number of iterations
    
    for ii = 1 : size(X,1)        
        
        if (sign(X(ii,:)*w') ~= Y(ii)) && (errorprob(sign((w + X(ii,:) * Y(ii))*X').*Y) < error) %Check if the classification is right
            w = w + X(ii,:) * Y(ii);   %then add (or subtract) this point to w
            error = errorprob(sign(w*X').*Y)
        end
        
end
  
end