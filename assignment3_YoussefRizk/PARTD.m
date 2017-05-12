% ------------------------------- PART D ---------------------------------

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
