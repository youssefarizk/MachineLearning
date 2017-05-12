function [w_true] = percepSRM(X,Y,n)

w_true = zeros(1,size(X,2));
globalError = 1;

pF0 = @(x1,x2,sizex) [ones(sizex, 1) x2];

pF1 = @(x1,x2,sizex) [ones(sizex, 1) x1 x2];

pF2 = @(x1,x2,sizex) [ones(sizex, 1) x1 x1.^2 x2];

pF3 = @(x1,x2,sizex) [ones(sizex, 1) x1 x1.^2 x1.^3 x2];

pF4 = @(x1,x2,sizex) [ones(sizex, 1) x1 x1.^2 x1.^3 x1.^4 x2];

size1 = size(X,1);

x_0 = pF0(X(:,2),X(:,3),size1);
x_1 = pF1(X(:,2),X(:,3),size1);
x_2 = pF2(X(:,2),X(:,3),size1);
x_3 = pF3(X(:,2),X(:,3),size1);
x_4 = pF4(X(:,2),X(:,3),size1);

w = zeros(1,size(x_0,2)); %Weights vector initialization
error = 1;

for i = 1:n %setting a defined number of iterations
    
    for ii = 1 : size(x_0,1)
        
        currentClassification = sign(w*x_0');
        
        if currentClassification(ii) ~= Y(ii) %Check if the classification is right
            w_old = w;
            w = w + x_0(ii,:) * Y(ii);   %then add (or subtract) this point to w
            if errorprob(sign(w*x_0').*Y) >= error
                w = w_old;
                error = errorprob(sign(w*x_0').*Y);
            end
        end
    end
    
end

if errorprob(sign(w*x_0').*Y) < globalError
    w_true = w;
    globalError = errorprob(sign(w*x_0').*Y);
end

w = zeros(1,size(x_1,2)); %Weights vector initialization
error = 1;

for i = 1:n %setting a defined number of iterations
    
    for ii = 1 : size(x_1,1)
        
        currentClassification = sign(w*x_1');
        
        if currentClassification(ii) ~= Y(ii) %Check if the classification is right
            w_old = w;
            w = w + x_1(ii,:) * Y(ii);   %then add (or subtract) this point to w
            if errorprob(sign(w*x_1').*Y) >= error
                w = w_old;
                error = errorprob(sign(w*x_1').*Y);
            end
        end
    end
    
end

if errorprob(sign(w*x_1').*Y) < globalError
    w_true = w;
    globalError = errorprob(sign(w*x_1').*Y);
end

w = zeros(1,size(x_2,2)); %Weights vector initialization
error = 1;

for i = 1:n %setting a defined number of iterations
    
    for ii = 1 : size(x_2,1)
        
        currentClassification = sign(w*x_2');
        
        if currentClassification(ii) ~= Y(ii) %Check if the classification is right
            w_old = w;
            w = w + x_2(ii,:) * Y(ii);   %then add (or subtract) this point to w
            if errorprob(sign(w*x_2').*Y) >= error
                w = w_old;
                error = errorprob(sign(w*x_2').*Y);
            end
        end
    end
    
end

if errorprob(sign(w*x_2').*Y) < globalError
    w_true = w;
    globalError = errorprob(sign(w*x_2').*Y);
end

w = zeros(1,size(x_3,2)); %Weights vector initialization
error = 1;

for i = 1:n %setting a defined number of iterations
    
    for ii = 1 : size(x_3,1)
        
        currentClassification = sign(w*x_3');
        
        if currentClassification(ii) ~= Y(ii) %Check if the classification is right
            w_old = w;
            w = w + x_3(ii,:) * Y(ii);   %then add (or subtract) this point to w
            if errorprob(sign(w*x_3').*Y) >= error
                w = w_old;
                error = errorprob(sign(w*x_3').*Y);
            end
        end
    end
    
end

if errorprob(sign(w*x_3').*Y) < globalError
    w_true = w;
    globalError = errorprob(sign(w*x_3').*Y);
end


w = zeros(1,size(x_4,2)); %Weights vector initialization
error = 1;

for i = 1:n %setting a defined number of iterations
    
    for ii = 1 : size(x_4,1)
        
        currentClassification = sign(w*x_4');
        
        if currentClassification(ii) ~= Y(ii) %Check if the classification is right
            w_old = w;
            w = w + x_4(ii,:) * Y(ii);   %then add (or subtract) this point to w
            if errorprob(sign(w*x_4').*Y) >= error
                w = w_old;
                error = errorprob(sign(w*x_4').*Y);
            end
        end
    end
    
end

if errorprob(sign(w*x_4').*Y) < globalError
    w_true = w;
    globalError = errorprob(sign(w*x_4').*Y);
end