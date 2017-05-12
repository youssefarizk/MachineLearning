function w = percep(X,Y)

w = zeros(1,size(X,2)); %Weights vector initialization
correct = false; % boolean variable will control loops

while ~correct
    
    class = sign(w*X');
    
    for ii = 1 : size(X,1)
        
         if class(ii) ~= Y(ii) %Check if the classification is right
            w = w + X(ii,:) * Y(ii);
            correct = errorprob(sign(w*X').*Y) == 0; %if not, we start again
        end
        
        
    end
end
