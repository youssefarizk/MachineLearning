clear all;

rawTrain = importdata('features.train');
trainLabels = classifyFor1(rawTrain);
rawTest = importdata('features.test');
testLabels = classifyFor1(rawTest);
rawTrain256 = importdata('zip.train');
rawTest256 = importdata('zip.test');
pcaVal = pca(rawTrain256(:,2:end));
pcaVal = pcaVal(:, 1:2);
transformed = [rawTrain256(:,1) rawTrain256(:,2:end)*pcaVal];
transformedTest = rawTest256(:,2:end)*pcaVal;

[onesO, notOnesO] = splitUp(rawTrain);
[onesT, notOnesT] = splitUp(transformed);

origplot = subplot(2,1,1)
hold on;
scatter(onesO(:,2),onesO(:,3),'rx');
scatter(notOnesO(:,2), notOnesO(:,3),'bx');
grid on;
title('Original Data');

SVMModel1 = fitcsvm(rawTrain(:,2:end), trainLabels, 'Standardize', false, 'KernelFunction', 'rbf',...
        'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions',...
        struct('SaveIntermediateResults', true, 'ShowPlots', false));
[x1grid, x2grid] = meshgrid(min(rawTrain(:,2)):0.05:max(rawTrain(:,2)), meshgrid(min(rawTrain(:,3)):0.05:max(rawTrain(:,3))));
xgrid = [x1grid(:) x2grid(:)];  
[~,scores] = predict(SVMModel1, xgrid);
rawTrain = rawTrain(:,2:end);
plot(rawTrain(SVMModel1.IsSupportVector,1),rawTrain(SVMModel1.IsSupportVector,2), 'ko')
[h,c] = contour(x1grid, x2grid, reshape(scores(:,2),size(x1grid)), [0 0], 'k')
legend('1','Non-1', 'Support Vectors', 'SVM Separator')


transplot = subplot(2,1,2)
hold on;
scatter(onesT(:,2),onesT(:,3),'rx');
scatter(notOnesT(:,2), notOnesT(:,3),'bx');
grid on;
title('PCA Generated Data');

SVMModel2 = fitcsvm(transformed(:,2:end), trainLabels, 'Standardize', false, 'KernelFunction', 'rbf',...
        'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions',...
        struct('SaveIntermediateResults', true, 'ShowPlots', false));
[x1grid, x2grid] = meshgrid(min(transformed(:,2)):0.05:max(transformed(:,2)), meshgrid(min(transformed(:,3)):0.05:max(transformed(:,3))));
xgrid = [x1grid(:) x2grid(:)];  
[~,scores] = predict(SVMModel2, xgrid);
transformed = transformed(:,2:end);
plot(transformed(SVMModel2.IsSupportVector,1),transformed(SVMModel2.IsSupportVector,2), 'ko')
[h,c] = contour(x1grid, x2grid, reshape(scores(:,2),size(x1grid)), [0 0], 'k')
legend('1','Non-1', 'Support Vectors', 'SVM Separator')

predTrainLabel1 = predict(SVMModel1,rawTrain);
trainError1 = sum((trainLabels.*predTrainLabel1') == -1)/ length(trainLabels)

predTrainLabel2 = predict(SVMModel2,transformed);
trainError2 = sum((trainLabels.*predTrainLabel2') == -1)/ length(trainLabels)

%Ideally add the test code as well
predTestLabel1 = predict(SVMModel1,rawTest(:,2:end));
testError1 = sum((testLabels.*predTestLabel1') == -1)/ length(testLabels)

predTestLabel2 = predict(SVMModel2,transformedTest);
testError2 = sum((testLabels.*predTestLabel2') == -1)/ length(testLabels)