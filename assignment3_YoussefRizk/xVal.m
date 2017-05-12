function y = xVal(training1,k, movieFeatures,lambda)

minVec = zeros(1,k);

xInd = crossvalind('Kfold', training1(:,1),k);

for iii = 1:k
    regresWeights = zeros(671,size(movieFeatures,2));
    testSum = 0;
    count = 0;
    for i = 1:671
        
        elements = [];
        
        while (size(training1,1) >count) && training1(count+1,1)==i
            if xInd(count+1) ~= iii
                elements = vertcat(elements, [movieFeatures(training1(count+1,2),2:end) training1(count+1,3)]);
            end
            count = count + 1;
        end
        
        if size(elements,1) > 0
            inv2 = ridge(elements(:,end),elements(:,1:end-1),lambda,0);
            regresWeights(i,:) = inv2;
        else
            regresWeights(i,:) = zeros(1,size(movieFeatures,2));
        end
        
    end
    
    
    %Testing now
    
    for i = 1:size(training1,1)
        if xInd(i) == iii
            testSum = testSum + (regresWeights(training1(i,1),:)*[1 movieFeatures(training1(i,2),2:end)]' - training1(i,3))^2;
        end
    end
    
    testSum = testSum/sum(xInd(:) == iii);
    
    minVec(iii) = testSum;
end

y = mean(minVec);
end