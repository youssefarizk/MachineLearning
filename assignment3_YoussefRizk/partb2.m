%% Part B: Regression Baseline Predictor

clear all;
rawTrainingFile = importdata('ratings-train.csv');
rawTrainingData = rawTrainingFile.data;
rawTestFile = importdata('ratings-test.csv');
rawTestData = rawTestFile.data;
rawFeaturesFile = importdata('movie-features.csv');
movieFeatures = rawFeaturesFile.data;
regresWeights = zeros(671,size(movieFeatures,2));
xValVec = [];
lambdaList = [0:200:2000];

for lambda = lambdaList
    xValVec = horzcat(xValVec,xVal(rawTrainingData, 10, movieFeatures,lambda))
end

bestLambda = lambdaList(find(xValVec == min(xValVec),1))

rawTrainingData = rawTrainingFile.data;
for i = 1:671
    
    elements = [];
    
    while (size(rawTrainingData,1) >0) && rawTrainingData(1,1) == i
        
        elements = vertcat(elements, [movieFeatures(rawTrainingData(1,2),2:end) rawTrainingData(1,3)]);
        rawTrainingData(1,:) = []; %Clearing the first row of the matrix
        
    end
    
    if size(elements,1) > 0
        inv2 = ridge(elements(:,end),elements(:,1:end-1),100000,0);
        regresWeights(i,:) = inv2;
    else
        regresWeights(i,:) = zeros(1,size(movieFeatures,2));
    end 
    
end

testSum = 0;

for i = 1:size(rawTestData,1)
    
    testSum = testSum + (regresWeights(rawTestData(i,1),:)*[1 movieFeatures(rawTestData(i,2),2:end)]' - rawTestData(i,3))^2;
    
end

testError = testSum/size(rawTestData,1);

