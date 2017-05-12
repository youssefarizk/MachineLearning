function [w,count] = randPercep(X,Y)

w = rand(1,size(X,2)); %Weights vector initialization
count = 0;
correct = false; % boolean variable will control loops

while ~correct    
    for ii = 1 : size(X,1)        
        
        if sign(X(ii,:)*w') ~= Y(ii) %Check if the classification is right
            w = w + X(ii,:) * Y(ii);
            count = count+1;
        end
        correct = errorprob(sign(w*X').*Y) == 0; %if not, we start again
        
    end
end

w= w/w(3);
  
end