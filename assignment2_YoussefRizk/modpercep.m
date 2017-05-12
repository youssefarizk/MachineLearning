function [wtrue] = modpercep(X,Y,n)

w = zeros(1,size(X,2)); %Weights vector initialization
wtrue = zeros(1,size(X,2));
error = 1;

for i = 1:n %setting a defined number of iterations
    
    for ii = 1 : size(X,1)
        
        if (sign(X(ii,:)*w') ~= Y(ii)) %Check if the classification is right
            w = w + X(ii,:) * Y(ii);   %then add (or subtract) this point to w
            newerror = errorprob(sign(w*X').*Y);
            if newerror < error
                error = newerror;
                wtrue = w;
            end
        end
        
    end
    
end