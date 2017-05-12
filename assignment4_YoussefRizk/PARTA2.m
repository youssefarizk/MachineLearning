clear all;

raw = importdata('zip.train');
rawTrain = filter1(raw);
trainLabels = classify2(rawTrain);
rawTest = filter1(importdata('zip.test'));
testLabels = classify2(rawTest);
minGamma = [];
%gammaList = 0:0.005:0.4;
gammaList = 0:0.1:0.1;
cList = 0.01:50:1000.01;
minC = [];
xValError = [];

for gamma = gammaList
    Clis = [];
    for C = cList
        Clis = horzcat(Clis, xVal(rawTrain,trainLabels,10,gamma,C));
    end
    figure;
    plot(cList,Clis)
    minC = horzcat(minC, cList(find(Clis == min(Clis),1)))
end

plot(gammaList, minGamma);
grid on;
title('Validation Error variation with Gamma');
xlabel('Gamma');
ylabel('Error');

bestGamma = gammaList(find(minGamma == min(minGamma),1))

figure;
plot(cList, minC)
grid on;
title('Validation Error variation with C');
xlabel('C');
ylabel('Error');

bestC = cList(find(minC == min(minC),1))

SVMModel = fitcsvm(rawTrain(:,2:end), trainLabels, 'Standardize', false, 'KernelFunction', 'rbf', 'KernelScale', 1/bestGamma, 'BoxConstraint', bestC);

predTrainLabel = predict(SVMModel,rawTrain(:,2:end));

trainError = sum((trainLabels.*predTrainLabel') == -1)/ length(trainLabels)

predTestLabel = predict(SVMModel,rawTest(:,2:end));

testError = sum((testLabels.*predTestLabel') == -1)/ length(testLabels)