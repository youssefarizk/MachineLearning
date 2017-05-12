%% Part A: Averaging by userID and movieID
clear all;
%Averaging by userID
rawTrainingFile = importdata('ratings-train.csv');
rawTrainingData = rawTrainingFile.data;
rawTestFile = importdata('ratings-test.csv');
rawTestData = rawTestFile.data;

aveUser = zeros(1, 671);

for i = 1:671
    
    elements = [];
    
    while (size(rawTrainingData,1) >0) && rawTrainingData(1,1) == i
        
        elements = horzcat(elements, [rawTrainingData(1,3)]);
        rawTrainingData(1,:) = []; %Clearing the first row of the matrix
        
    end
    
    if length(elements) > 0
        aveUser(i) = mean(elements);
    else
        aveUser(i) = 0;
    end
    
end

%Training performance of Average UserID

rawTrainData = rawTrainingFile.data;

trainSum = 0;

for i = 1:size(rawTrainData,1)
    
    trainSum = trainSum + (aveUser(rawTrainData(i,1)) - rawTrainData(i,3))^2;
    
end

trainError = trainSum/size(rawTrainData,1)

%Test performance of Average UserID

testSum = 0;

for i = 1:size(rawTestData,1)
    
    testSum = testSum + (aveUser(rawTestData(i,1)) - rawTestData(i,3))^2;
    
end

testError = testSum/size(rawTestData,1)

% Averaging by movieID

rawTrainingData = rawTrainingFile.data;
aveMovie = zeros(1, 9066);

rawTrainingData = sortrows([rawTrainingData(:,2) rawTrainingData(:,1) rawTrainingData(:,3)]);

for i = 1:9066
    
    elements = [];
    
    while (size(rawTrainingData,1) > 0) && rawTrainingData(1,1) == i
        
        elements = horzcat(elements, [rawTrainingData(1,3)]);
        rawTrainingData(1,:) = []; %Clearing the first row of the matrix
        
    end
    
    if length(elements) > 0
        aveMovie(i) = mean(elements);
    else
        aveMovie(i) = 0;
    end
    
end

%Training performance of Average MovieID

rawTrainingData = rawTrainingFile.data;

trainSum = 0;

expectedMov = zeros(size(rawTestData,1),1);

for i = 1:size(rawTrainingData,1)
    
    expectedMov(i) = aveMovie(rawTrainingData(i,2));
    
end

trainError = immse(rawTrainingData(:,3),expectedMov)

%Test performance of Average MovieID

testSum = 0;

expectedMov = zeros(size(rawTestData,1),1);

for i = 1:size(rawTestData,1)
    
    expectedMov(i) = aveMovie(rawTestData(i,2));
    
end

testError = immse(rawTestData(:,3),expectedMov)


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

%% ------------------------------- PART D ---------------------------------

clear all;
rawTrainingFile = importdata('ratings-train.csv');
rawTrainingData = rawTrainingFile.data;
rawTestFile = importdata('ratings-test.csv');
rawTestData = rawTestFile.data;
rawFeaturesFile = importdata('movie-features.csv');
movieFeatures = rawFeaturesFile.data;
errorVec = [];
lambdaList = 0:0.0005:0.01;
kList = 1:15;
xValL = [];
xValK = [];


for k = kList
    xValK = horzcat(xValK,xValSGD(rawTrainingData, k, 0))
end
bestK = kList(find(xValK == min(xValK),1));

plot(kList,xValK);
xlabel('K');
ylabel('Validation Error');
title('Variation of Validation Error with K');
grid on;

for l = lambdaList
    xValL = horzcat(xValL,xValSGD(rawTrainingData, bestK, l))
end
bestL = lambdaList(find(xValK == min(xValL),1));
plot(lambdaList,xValL);
xlabel('Lambda');
ylabel('Validation Error');
title('Variation of Validation Error with Lambda');
grid on;

userWeights = rand(671,bestK);
movieWeights = rand(9066,bestK);



iterations = 10000000;
eta_0 = 0.000145;


count = 0;
itTrain = [];
itTest = [];
for T = 1:iterations
    eta = eta_0;
    % pick a random user and movie from training data
    index = round((size(rawTrainingData,1)-1)*rand)+1;
    i = rawTrainingData(index,1);
    j = rawTrainingData(index,2);
    
    % compute loss function
    loss_ij = (rawTrainingData(index,3) - userWeights(i,:)*movieWeights(j,:)')^2;
    
    % compute estimates of loss function gradient
    gradu = -2*(rawTrainingData(index,3) - userWeights(i,:)*movieWeights(j,:)')*movieWeights(j,:) - 2*bestL*userWeights(i,:);
    gradv = -2*(rawTrainingData(index,3) - userWeights(i,:)*movieWeights(j,:)')*userWeights(i,:) - 2*bestL*movieWeights(i,:);
    
    % update rule
    userWeights(i,:) = (1)*userWeights(i,:) - eta*gradu;
    movieWeights(j,:) = (1)*movieWeights(j,:) - eta*gradv;
    count = count + 1;
    
    if count == 1000
        calculatedRatings = userWeights*movieWeights';
        error = 0;
        tests = size(rawTestData,1);
        for r = 1:tests
            desired_rating = rawTestData(r,3);
            exp_rating = calculatedRatings(rawTestData(r,1),rawTestData(r,2));
            error = error + (desired_rating - exp_rating)^2;
        end
        error = error/tests;
        itTest = horzcat(itTest,error);
        error = 0;
        train = size(rawTrainingData,1);
        for r = 1:train
            desired_rating = rawTrainingData(r,3);
            exp_rating = calculatedRatings(rawTrainingData(r,1),rawTrainingData(r,2));
            error = error + (desired_rating - exp_rating)^2;
        end
        error = error/train;
        itTrain = horzcat(itTrain,error);
        count = 1;
    end
    
end
it = [1:1000:10010000];
plot(it, itTest);
hold on;
plot(it, itTrain);
xlabel('Iterations');
ylabel('Average Means Squared Error');
title('Variation of Error with Iterations');
axis([0 10000000 0.6 2.5]);
legend('Test Error','Training Error');

calculatedRatings = userWeights*movieWeights';
error = 0;
tests = size(rawTestData,1);
for r = 1:tests
    actRating = rawTestData(r,3);
    calcRating = calculatedRatings(rawTestData(r,1),rawTestData(r,2));
    error = error + (actRating-calcRating)^2;
end
error = error/tests;
errorVec = horzcat(errorVec,error)
