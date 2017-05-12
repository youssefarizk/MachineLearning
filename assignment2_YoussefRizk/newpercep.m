function [w] = newpercep(X,Y,n)

w = zeros(1,size(X,2)); %Weights vector initialization

for i = 1:n    
    for ii = 1 : size(X,1)        
        
        if sign(X(ii,:)*w') ~= Y(ii) %Check if the classification is right
            w = w + X(ii,:) * Y(ii);
        end
        
    end
end
  
end