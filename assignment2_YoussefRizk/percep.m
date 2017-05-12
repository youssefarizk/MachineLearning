function w = percep(X,Y)

X = [ones(size(X,1),1) X];

sizeX = size(X,2);

w = zeros(1,sizeX); %Weights vector initialization



for i = 1:10000
    
    currentClassification = sign(w*X');
    
    for ii = 1 : size(X,1)
        
        if currentClassification(ii) ~= Y(ii) %Check if the classification is right
            w = w + X(ii,:) * Y(ii);
            break; %if not, we start again
        end
        
    end
end
