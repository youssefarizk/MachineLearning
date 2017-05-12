%Exercise 3.c.2

p_training256 = zeros(1,35);
p_test256 = zeros(1,35);

unfilteredTraining256 = importdata('zip.train');
filteredTraining256 = filter1(unfilteredTraining256);
unfilteredTest256 = importdata('zip.test');
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
