%Exercise 3.c.3

unfilteredTraining2 = importdata('features.train');
filteredTraining2 = filter1(unfilteredTraining2);
unfilteredTest2 = importdata('features.test');
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