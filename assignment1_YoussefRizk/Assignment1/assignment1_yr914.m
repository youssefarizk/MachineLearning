%Testing script for problem 3a
n = 2;
X = [ones(n,1) rand(n,2)];
Y = simpleClassify(X);
subplot(2,2,1)
scatter(X(:,2),X(:,3))
hold on
X1 = 0.5*ones(n,1);
plot(X(:,2),X1);
hold on

w = percep(X,Y);
w = w/w(3)
Y1 = -w(2)*X(:,2) - w(1);
plot(X(:,2),Y1);
legend('Data Points','Actual Separator','Calculated Separator');
title('Testing the Perceptron Algorithm with 2 Points');
xlabel('x1')
ylabel('x2')

n = 4;
X = [ones(n,1) rand(n,2)];
Y = simpleClassify(X);
subplot(2,2,2)
scatter(X(:,2),X(:,3))
hold on
X1 = 0.5*ones(n,1);
plot(X(:,2),X1);
hold on

w = percep(X,Y);
w = w/w(3)
Y1 = -w(2)*X(:,2) - w(1);
plot(X(:,2),Y1);
legend('Data Points','Actual Separator','Calculated Separator');
title('Testing the Perceptron Algorithm with 4 Points');
xlabel('x1')
ylabel('x2')

n = 10;
X = [ones(n,1) rand(n,2)];
Y = simpleClassify(X);
subplot(2,2,3)
scatter(X(:,2),X(:,3))
hold on
X1 = 0.5*ones(n,1);
plot(X(:,2),X1);
hold on

w = percep(X,Y);
w = w/w(3)
Y1 = -w(2)*X(:,2) - w(1);
plot(X(:,2),Y1);
legend('Data Points','Actual Separator','Calculated Separator');
title('Testing the Perceptron Algorithm with 10 Points');
xlabel('x1')
ylabel('x2')

n = 100;
X = [ones(n,1) rand(n,2)];
Y = simpleClassify(X);
subplot(2,2,4)
scatter(X(:,2),X(:,3))
hold on
X1 = 0.5*ones(n,1);
plot(X(:,2),X1);
hold on

w = percep(X,Y);
w = w/w(3)
Y1 = -w(2)*X(:,2) - w(1);
plot(X(:,2),Y1);
legend('Data Points','Actual Separator','Calculated Separator');
title('Testing the Perceptron Algorithm with 100 Points');
xlabel('x1')
ylabel('x2')

clear all;

%Exercise 3.b.2

Test = [ones(100000,1) rand(100000,2)];
testClassify = classify(Test);
p = zeros(1,5);
figure
for i = 1:5
    X = zeros(i*100,3);
    X(:,1:3) = [ones(i*100,1) rand(i*100,2)];
    Y = classify(X);
    A = percep(X,Y);
    A = A/A(3);
    p(i) = errorprob(sign(A*Test').*testClassify);
end
plot(1:5,p)
title('Test error for various data sizes');
xlabel('Number of data points (00s)');
ylabel('Error');
grid on;

clear all;
percent95 = zeros(1,5);
percent5 = zeros(1,5);

Test = [ones(100000,1) rand(100000,2)];
testClassify = classify(Test);

figure
n = 100;
prob1 = zeros(n,5);
for ii = 1:n
    p = zeros(1,5);
    for i = 1:5
        X = zeros(i*100,3);
        X(:,1:3) = [ones(i*100,1) rand(i*100,2)];
        Y = classify(X);
        A = percep(X,Y);
        A = A/A(3);
        p(i) = errorprob(sign(A*Test').*testClassify);
        prob1(ii,i) = p(i)
    end
end
for i = 1:5
    percent95(i) = prctile(prob1(:,i),95);
    percent5(i) = prctile(prob1(:,i),5)
end

sort(prob1);
plot(1:5, mean(prob1(6:95,:)));
hold on;
plot(1:5,mean(prob1));


title('Average test error for various data sizes')
xlabel('Number of data points (00s)')
ylabel('Error')
grid on
legend('90% Confidence','Average')

hold off;

figure
plot(1:5,mean(prob1))
hold on;
plot(1:5,percent95)
hold on;
plot(1:5,percent5)
k = fill([[1:5], fliplr([1:5])], [percent95, fliplr(percent5)], 'r');
set(k,'facealpha',.3);
grid on;
title('Average test error for various data sizes')
xlabel('Number of data points (00s)')
ylabel('Error')
legend('Average','5th to 95th Percentile')

clear percent95;
clear percent5;

%exercise 3b4
n = 100;
figure;
percent95 = [];
percent5 = [];
meanVec = [];

for margin = [0.3 0.1 0.01 0.001]
    Test = filterMargin(100000, margin);
    testClassify = classify(Test);
    
    prob = zeros(n,5);
    
    for ii = 1:n
        for i = 1:5
            X = filterMargin(i*100,margin);
            Y = classify(X);
            A = percep(X,Y);
            A = A/A(3);
            prob(ii,i) = errorprob(sign(A*Test').*testClassify)
        end
    end
    sort(prob);
    plot(1:5,mean(prob(6:95,1:5)))
    meanVec = vertcat(meanVec,mean(prob));
    percent95 = vertcat(percent95,prctile(prob,95));
    percent5 = vertcat(percent5,prctile(prob,5));
    hold on;
end

plot(1:5,mean(prob1(6:95,1:5)))

grid on;
title('Test error vs. number of data points for various margin sizes');
xlabel('Number of data points (00s)');
ylabel('Error');
legend('Margin = 0.3','Margin = 0.1','Margin = 0.01','Margin = 0.001','Margin = 0');
hold off;

figure
subplot(2,2,1)
plot(1:5,meanVec(1,:))
hold on;
plot(1:5,percent95(1,:))
hold on;
plot(1:5,percent5(1,:))
k = fill([[1:5], fliplr([1:5])], [percent95(1,:), fliplr(percent5(1,:))], 'r');
set(k,'facealpha',.3);
grid on;
title('Average test error for various data sizes for 0.3 margin')
xlabel('Number of data points (00s)')
ylabel('Error')
legend('Average','5th to 95th Percentile')

subplot(2,2,2)
plot(1:5,meanVec(2,:))
hold on;
plot(1:5,percent95(2,:))
hold on;
plot(1:5,percent5(2,:))
k = fill([[1:5], fliplr([1:5])], [percent95(2,:), fliplr(percent5(2,:))], 'r');
set(k,'facealpha',.3);
grid on;
title('Average test error for various data sizes for 0.1 margin')
xlabel('Number of data points (00s)')
ylabel('Error')
legend('Average','5th to 95th Percentile')

subplot(2,2,3)
plot(1:5,meanVec(3,:))
hold on;
plot(1:5,percent95(3,:))
hold on;
plot(1:5,percent5(3,:))
k = fill([[1:5], fliplr([1:5])], [percent95(3,:), fliplr(percent5(3,:))], 'r');
set(k,'facealpha',.3);
grid on;
title('Average test error for various data sizes for 0.01 margin')
xlabel('Number of data points (00s)')
ylabel('Error')
legend('Average','5th to 95th Percentile')

subplot(2,2,4)
plot(1:5,meanVec(4,:))
hold on;
plot(1:5,percent95(4,:))
hold on;
plot(1:5,percent5(4,:))
k = fill([[1:5], fliplr([1:5])], [percent95(4,:), fliplr(percent5(4,:))], 'r');
set(k,'facealpha',.3);
grid on;
title('Average test error for various data sizes for 0.001 margin')
xlabel('Number of data points (00s)')
ylabel('Error')
legend('Average','5th to 95th Percentile')

clear all;


%% exercise 3b4 part 2
figure;
n = 100;
boundVector = [];

for margin = [0:0.01:0.3]
    Test = filterMargin(100000, margin);
    testClassify = classify(Test);
    bound = zeros(1,n);
    
    for ii = 1:n
        X = filterMargin(100,margin);
        Y = classify(X);
        [A,count] = theoreticalPercep(X,Y);
        
        for i = 1:n
            normvector(i) = norm(X(i,:));
            minvector(i) = A*X(i,:)'*Y(i);
        end
        
        maxNorm = max(normvector);
        minRho = min(minvector);
        
        upperBound = (norm(A)^2*maxNorm)/minRho^2;
        bound(ii) = count/upperBound;
    end
    sort(bound);
    boundVector = horzcat(boundVector,[mean(bound(6:95))]);
    hold on;
end

plot([0:0.01:0.3],boundVector)

grid on;
title('Number of iterations to upper bound ratio for various margin sizes');
xlabel('Margin size');
ylabel('Ratio');
hold off;

%exercise 3b4 part 2: finding 2 weight vectors
X = [ones(100,1) rand(100,2)];
Y = classify(X);
[w1,count1] = randPercep(X,Y)
[w2,count2] = randPercep(X,Y)

%Exercise 3.c.1
figure;
unfilteredTraining256 = importdata('../data/zip.train');
filteredTraining256 = filter1(unfilteredTraining256);
unfilteredTest256 = importdata('../data/zip.test');
filteredTest256 = filter1(unfilteredTest256);
trainOut256 = classify2(filteredTraining256);
testOut256 = classify2(filteredTest256);
trainIn256 = [ones(1273,1) filteredTraining256(:,2:257)];
testIn256 = [ones(364,1) filteredTest256(:,2:257)];
weights256 = modpercep(trainIn256,trainOut256,1000);
errorprob(sign(weights256*trainIn256').*trainOut256)
errorprob(sign(weights256*testIn256').*testOut256)

unfilteredTraining2 = importdata('../data/features.train');
filteredTraining2 = filter1(unfilteredTraining2);

train2 = filter2(filteredTraining2);
train8 = filter8(filteredTraining2);

subplot(2,1,1)
plot1 = scatter(train2(:,2),train2(:,3),'bo');
hold on;
scatter(train8(:,2),train8(:,3),'gx');
hold on;

unfilteredTest2 = importdata('../data/features.test');
filteredTest2 = filter1(unfilteredTest2);
trainOut2 = classify2(filteredTraining2);
testOut2 = classify2(filteredTest2);
trainIn2 = [ones(1273,1) filteredTraining2(:,2:3)];
testIn2 = [ones(364,1) filteredTest2(:,2:3)];
weights2 = modpercep(trainIn2,trainOut2,1000);
weights2_norm = weights2/weights2(3);

plot(0:0.1:1,(-weights2_norm(2)*[0:0.1:1]' - weights2_norm(1)))
grid on;
title('Training Data Classification');
xlabel('Intensity');
ylabel('Symmetry');
legend('Data -> 2','Data -> 8','Classifier Line')

hold off;


test2 = filter2(filteredTest2);
test8 = filter8(filteredTest2);

subplot(2,1,2)
plot2 = scatter(test2(:,2),test2(:,3),'bo');
hold on;
scatter(test8(:,2),test8(:,3),'gx');
hold on;

plot(0:0.1:1,(-weights2_norm(2)*[0:0.1:1]' - weights2_norm(1)))
grid on;
title('Test Data Classification');
xlabel('Intensity');
ylabel('Symmetry');
legend('Data -> 2','Data -> 8','Classifier Line')

subplot()


errorprob(sign(weights2*trainIn2').*trainOut2)
errorprob(sign(weights2*testIn2').*testOut2)

clear all;

%Exercise 3.c.2
figure;
p_training256 = zeros(1,35);
p_test256 = zeros(1,35);

unfilteredTraining256 = importdata('../data/zip.train');
filteredTraining256 = filter1(unfilteredTraining256);
unfilteredTest256 = importdata('../data/zip.test');
filteredTest256 = filter1(unfilteredTest256);
trainOut256 = classify2(filteredTraining256);
testOut256 = classify2(filteredTest256);
trainIn256 = [ones(1273,1) filteredTraining256(:,2:257)];
testIn256 = [ones(364,1) filteredTest256(:,2:257)];

for i = 1:35
    weights256 = modpercep(trainIn256,trainOut256,i);
    p_training256(i) = errorprob(sign(weights256*trainIn256').*trainOut256)
    p_test256(i) = errorprob(sign(weights256*testIn256').*testOut256)
end

plot(1:35,p_training256)
hold on;
plot(1:35, p_test256)
title('Training and Test errors for 256-D data')
grid on;
xlabel('Number of iterations');
ylabel('Error');
legend('Training Error','Test Error');
hold off;

clear all;

figure;

n = size([10:10:1000],2);
p_training2 = zeros(1,n);
p_test2 = zeros(1,n);

unfilteredTraining2 = importdata('../data/features.train');
filteredTraining2 = filter1(unfilteredTraining2);
unfilteredTest2 = importdata('../data/features.test');
filteredTest2 = filter1(unfilteredTest2);
trainOut2 = classify2(filteredTraining2);
testOut2 = classify2(filteredTest2);
trainIn2 = [ones(1273,1) filteredTraining2(:,2:3)];
testIn2 = [ones(364,1) filteredTest2(:,2:3)];

for i = 10:10:1000
    weights2 = modpercep(trainIn2,trainOut2,i);
    p_training2(i/10) = errorprob(sign(weights2*trainIn2').*trainOut2)
    p_test2(i/10) = errorprob(sign(weights2*testIn2').*testOut2)
end

plot(10:10:1000,p_training2)
hold on;
plot(10:10:1000, p_test2)
hold on;
title('Training and Test errors for 2-D data')
grid on;
xlabel('Number of iterations');
ylabel('Error');
legend('Training Error','Test Error');
hold off;

clear all;

%Exercise 3.c.3
figure;
unfilteredTraining2 = importdata('../data/features.train');
filteredTraining2 = filter1(unfilteredTraining2);
unfilteredTest2 = importdata('../data/features.test');
filteredTest2 = filter1(unfilteredTest2);
trainOut2 = classify2(filteredTraining2);
testOut2 = classify2(filteredTest2);
trainIn2 = [ones(1273,1) filteredTraining2(:,2:3)];
testIn2 = [ones(364,1) filteredTest2(:,2:3)];
weights2 = modpercep(trainIn2,trainOut2,1000);
weights2_norm = weights2/weights2(3);

test2 = filter2(filteredTest2);
test8 = filter8(filteredTest2);

scatter(test2(:,2),test2(:,3),'bo')
hold on;
scatter(test8(:,2),test8(:,3),'gx')
hold on;
plot(0:0.1:1,(-weights2_norm(2)*[0:0.1:1]' - weights2_norm(1)),'r')
hold on;

weights2 = regresPercep(trainIn2,trainOut2,1000);
weights2_norm = weights2/weights2(3);

plot(0:0.05:1,(-weights2_norm(2)*[0:0.05:1]' - weights2_norm(1)),'k')

title('Comparing Regression vs. Zero Initializations');
grid on;
xlabel('Intensity');
ylabel('Symmetry');
legend('Data -> 2','Data -> 8','Zero-Initialization','Regression Line')
hold off;

clear all;

%exercise3c3 part 2
figure;
n = size([10:10:1000],2);
p_training2 = zeros(1,n);
p_test2 = zeros(1,n);
p_training2r = zeros(1,n);
p_test2r = zeros(1,n);

unfilteredTraining2 = importdata('../data/features.train');
filteredTraining2 = filter1(unfilteredTraining2);
unfilteredTest2 = importdata('../data/features.test');
filteredTest2 = filter1(unfilteredTest2);
trainOut2 = classify2(filteredTraining2);
testOut2 = classify2(filteredTest2);
trainIn2 = [ones(1273,1) filteredTraining2(:,2:3)];
testIn2 = [ones(364,1) filteredTest2(:,2:3)];

for i = 10:10:1000
    weights2 = modpercep(trainIn2,trainOut2,i);
    weights2r = regresPercep(trainIn2,trainOut2,i);
    p_training2(i/10) = errorprob(sign(weights2*trainIn2').*trainOut2)
    p_test2(i/10) = errorprob(sign(weights2*testIn2').*testOut2)
    p_training2r(i/10) = errorprob(sign(weights2r'*trainIn2').*trainOut2)
    p_test2r(i/10) = errorprob(sign(weights2r'*testIn2').*testOut2)
end

plot(10:10:1000,p_training2, 'r--')
hold on;
plot(10:10:1000, p_test2,'r')
hold on;
plot(10:10:1000,p_training2r,'b--')
hold on;
plot(10:10:1000, p_test2r, 'b')
title('Training and Test errors for 2-D data')
grid on;
xlabel('Number of iterations');
ylabel('Error');
legend('Training Error (Zero)','Test Error (Zero)','Training Error (Regression)','Test Error (Regression)');


