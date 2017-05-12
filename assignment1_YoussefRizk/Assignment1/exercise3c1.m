%Exercise 3.c.1

unfilteredTraining256 = importdata('zip.train');
filteredTraining256 = filter1(unfilteredTraining256);
unfilteredTest256 = importdata('zip.test');
filteredTest256 = filter1(unfilteredTest256);
trainOut256 = classify2(filteredTraining256);
testOut256 = classify2(filteredTest256); 
trainIn256 = [ones(1273,1) filteredTraining256(:,2:257)];
testIn256 = [ones(364,1) filteredTest256(:,2:257)];
weights256 = modpercep(trainIn256,trainOut256,1000);
errorprob(sign(weights256*trainIn256').*trainOut256)
errorprob(sign(weights256*testIn256').*testOut256)

unfilteredTraining2 = importdata('features.train');
filteredTraining2 = filter1(unfilteredTraining2);

train2 = filter2(filteredTraining2);
train8 = filter8(filteredTraining2);

subplot(2,1,1)
plot1 = scatter(train2(:,2),train2(:,3),'bo');
hold on;
scatter(train8(:,2),train8(:,3),'gx');
hold on;

unfilteredTest2 = importdata('features.test');
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