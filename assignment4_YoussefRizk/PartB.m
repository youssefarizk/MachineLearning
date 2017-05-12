clear all;

raw = importdata('zip.train');
rawTrain = filter1(raw);
trainLabels = classify2(rawTrain);
rawTest = filter1(importdata('zip.test'));
testLabels = classify2(rawTest);

pcaVal = pca(rawTrain(:,2:end));
kmax = size(rawTrain(:,2:end),2);
klist = 1:5:kmax;

trainingError = [];
testingError = [];
validationError = [];

gamma = 0.1;
C = 1000;

for k = klist
    transformedTrainFeat = rawTrain(:,2:end)*pcaVal(:,1:k);
    transformedTestFeat = rawTest(:,2:end)*pcaVal(:,1:k);

    
%     SVMModel = fitcsvm(transformedTrainFeat, trainLabels, 'Standardize',...
%         false, 'KernelFunction', 'rbf','KernelScale', 1/gamma,...
%         'BoxConstraint',C);
    SVMModel = fitcsvm(transformedTrainFeat, trainLabels, 'Standardize', false, 'KernelFunction', 'rbf',...
        'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions',...
        struct('SaveIntermediateResults', true, 'ShowPlots', false));
    
    predTrainLabel = predict(SVMModel,transformedTrainFeat);
    
    trainError = sum((trainLabels.*predTrainLabel') == -1)/ length(trainLabels);
    
    trainingError = horzcat(trainingError,trainError);
        
    predTestLabel = predict(SVMModel,transformedTestFeat);
    
    testError = sum((testLabels.*predTestLabel') == -1)/ length(testLabels);
    
    testingError = horzcat(testingError, testError);
    
    validationError = horzcat( validationError, xVal([rawTrain(:,1) transformedTrainFeat],trainLabels,10,gamma,C));
    
end

trainplot = figure;
plot(klist, trainingError);
hold on;
plot(klist, testingError);
plot(klist, validationError);
title('Different Error variation with K');
xlabel('K')
ylabel('Error')
legend('Training Error', 'Testing Error','Validation Error')
grid on;
