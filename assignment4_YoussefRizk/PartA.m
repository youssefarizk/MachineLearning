clear all;

raw = importdata('zip.train');
rawTrain = filter1(raw);
trainLabels = classify2(rawTrain);
rawTest = filter1(importdata('zip.test'));
testLabels = classify2(rawTest);

SVMModel = fitcsvm(rawTrain(:,2:end), trainLabels, 'Standardize', false,...,
    'KernelFunction', 'rbf','OptimizeHyperparameters', 'auto',...,
    'HyperparameterOptimizationOptions',...,
    struct('SaveIntermediateResults', true,...
    'Kfold',10));

predTrainLabel = predict(SVMModel,rawTrain(:,2:end));

trainError = sum((trainLabels.*predTrainLabel') == -1)/ length(trainLabels)

predTestLabel = predict(SVMModel,rawTest(:,2:end));

testError = sum((testLabels.*predTestLabel') == -1)/ length(testLabels)