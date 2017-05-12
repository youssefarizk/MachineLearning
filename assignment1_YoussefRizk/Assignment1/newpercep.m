function [w] = newpercep(X,Y)

w = zeros(1,size(X,2)); %Weights vector initialization
correct = false; % boolean variable will control loops

while ~correct
correct = true;
    
    for ii = 1 : size(X,1)        
        
        if ~correct 
           break %Exiting the loop if the classifciation has already failed
        end
        
        if sign(X(ii,:)*w') ~= Y(ii) %Check if the classification is right
            w = w + X(ii,:) * Y(ii);
            correct = false; %if not, we start again
        end
        
    end
end
  
end