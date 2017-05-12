function y = xVal(training, labels, k, gamma, C)

minVec = zeros(1,k);

xInd = crossvalind('Kfold', training(:,1),k);

for i = 1:k

trainSet = training(xInd ~= i,:);
trainingLabels = labels(xInd ~= i);

validationSet = training(xInd == i,:);
validationLabels = labels(xInd == i);

SVMModel = fitcsvm(trainSet(:,2:end), trainingLabels, 'Standardize', false, 'KernelFunction', 'rbf', 'KernelScale', 1/gamma, 'BoxConstraint', C);

% SVMModel = fitcsvm(trainSet(:,2:end), trainingLabels,'Standardize', false, 'KernelFunction', 'rbf',...
%         'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions',...
%         struct('SaveIntermediateResults', true, 'ShowPlots', false));
    
predictedValLabel = predict(SVMModel,validationSet(:,2:end));

minVec(i) = sum((validationLabels.*predictedValLabel') == -1)/length(validationLabels);

end

y = mean(minVec);