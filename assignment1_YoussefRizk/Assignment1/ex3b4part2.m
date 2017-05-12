%exercise 3b4 part 2
n = 100;
boundVector = [];

for margin = [0:0.01:0.3]
    Test = filterMargin(100000, margin);
    testClassify = classify(Test);

    bound = zeros(1,n);
    
    for ii = 1:n 
        X = filterMargin(100,margin);
        Y = classify(X);
        [A,count] = theoreticalPercep(X,Y);
        
        for i = 1:n
            normvector(i) = norm(X(i,:));
            minvector(i) = A*X(i,:)'*Y(i);
        end

        maxNorm = max(normvector);
        minRho = min(minvector);

        upperBound = (norm(A)^2*maxNorm)/minRho^2;
        bound(ii) = count/upperBound;
    end
    sort(bound);
    boundVector = horzcat(boundVector,[mean(bound(6:95))]);
    hold on;
end

plot([0:0.01:0.3],boundVector)

grid on;
title('Number of iterations to upper bound ratio for various margin sizes');
xlabel('Margin size');
ylabel('Ratio');