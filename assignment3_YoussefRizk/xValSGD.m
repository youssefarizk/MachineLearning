function y = xValSGD(rawTrainingData,k,lambda)

minVec = zeros(1,10);
eta = 0.000145;
xInd = crossvalind('Kfold', rawTrainingData(:,1),10);
iterations = 4000000;
for iii = 1:10
    
    userWeights = rand(671,k);
    movieWeights = rand(9066,k);
    
    testSum = 0;
    for T = 1:iterations
        
    index = round((size(rawTrainingData,1)-1)*rand)+1;
    i = rawTrainingData(index,1);
    j = rawTrainingData(index,2);
    
    % compute loss function
    loss_ij = (rawTrainingData(index,3) - userWeights(i,:)*movieWeights(j,:)')^2;
    
    % compute estimates of loss function gradient
    gradu = -2*(rawTrainingData(index,3) - userWeights(i,:)*movieWeights(j,:)')*movieWeights(j,:) - 2*lambda*userWeights(i,:);
    gradv = -2*(rawTrainingData(index,3) - userWeights(i,:)*movieWeights(j,:)')*userWeights(i,:) - 2*lambda*movieWeights(i,:);
    
    % update rule
    if xInd(index) ~= iii
    userWeights(i,:) = (1)*userWeights(i,:) - eta*gradu;
    movieWeights(j,:) = (1)*movieWeights(j,:) - eta*gradv;
    end
        
    end
    
    
    %Testing now
    calculatedRatings = userWeights*movieWeights';
    for i = 1:size(rawTrainingData,1)
        if xInd(i) == iii
            desired_rating = rawTrainingData(i,3);
            exp_rating = calculatedRatings(rawTrainingData(i,1),rawTrainingData(i,2));
            testSum = testSum + (desired_rating - exp_rating)^2;
        end
    end
    
    testSum = testSum/sum(xInd(:) == iii);
    
    minVec(iii) = testSum;
end

y = mean(minVec);
end