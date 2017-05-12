%% Part C: Regression Baseline Predictor for Transformed Inputs

clear all;
rawTrainingFile = importdata('ratings-train.csv');
rawTrainingData = rawTrainingFile.data;
rawTestFile = importdata('ratings-test.csv');
rawTestData = rawTestFile.data;
rawFeaturesFile = importdata('movie-features.csv');
movieFeatures = rawFeaturesFile.data;
k = 18;
kList = 1:k;
justFeatures = movieFeatures(:, 2:19);
xValkList = [];

%X-Validation on K

for i = kList
    transformFeatures_2 = movieFeatures;
    %Grouping using k-means clustering 
    kmeansVec = kmeans(movieFeatures(:,2:end)', i, 'Distance', 'correlation');
    
    for ii = 1:i
        indx = find(kmeansVec == ii);
        transformFeatures_2 = horzcat(transformFeatures_2, prod(justFeatures(:,indx),2));
    end
    
    xValkList = horzcat(xValkList,xVal(rawTrainingData, 10, transformFeatures_2,200))
    
end

bestK = kList(find(xValkList == min(xValkList),1))

xValVec = [];
lambdaList = [0:10:200];

transformFeatures_2 = movieFeatures;
%Grouping using k-means clustering
kmeansVec = kmeans(movieFeatures(:,2:end)', bestK, 'Distance', 'correlation');

for ii = 1:bestK
    indx = find(kmeansVec == ii);
    transformFeatures_2 = horzcat(transformFeatures_2, prod(justFeatures(:,indx),2));
end


regresWeights = zeros(671,size(transformFeatures_2,2));
%X-Validation on Lambda

for lambda = lambdaList
    xValVec = horzcat(xValVec,xVal(rawTrainingData, 10, transformFeatures_2,lambda))
end

bestLambda = lambdaList(find(xValVec == min(xValVec),1))


for i = 1:671
    
    elements = [];
    
    while (size(rawTrainingData,1) > 0) && rawTrainingData(1,1) == i
        
        elements = vertcat(elements, [transformFeatures_2(rawTrainingData(1,2),2:end) rawTrainingData(1,3)]);
        rawTrainingData(1,:) = []; %Clearing the first row of the matrix
        
    end
    
    if size(elements,1) > 0
        inv2 = ridge(elements(:,end),elements(:,1:end-1),bestLambda,0);
        regresWeights(i,:) = inv2';
    else
        regresWeights(i,:) = zeros(1,size(transformFeatures_2,2));
    end
    
end

testSum = 0;

for i = 1:size(rawTestData,1)
    
    testSum = testSum + (regresWeights(rawTestData(i,1),:)*[1 transformFeatures_2(rawTestData(i,2),2:end)]' - rawTestData(i,3))^2;
    
end

testError = testSum/size(rawTestData,1)