n = size([10:10:1000],2);
p_training2 = zeros(1,n);
p_test2 = zeros(1,n);

unfilteredTraining2 = importdata('features.train');
filteredTraining2 = filter1(unfilteredTraining2);
unfilteredTest2 = importdata('features.test');
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

