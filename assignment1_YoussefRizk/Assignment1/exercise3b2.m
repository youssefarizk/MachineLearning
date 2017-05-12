%Exercise 3.b.3

Test = [ones(100000,1) rand(100000,2)];
testClassify = classify(Test);


figure
n = 100;
prob = zeros(n,5);
for ii = 1:n
p = zeros(1,5);
for i = 1:5   
    X = zeros(i*100,3);
    X(:,1:3) = [ones(i*100,1) rand(i*100,2)];
    Y = classify(X);
    A = percep(X,Y);
    A = A/A(3);
    p(i) = errorprob(sign(A*Test').*testClassify);
    prob(ii,i) = p(i)
end
end
sort(prob);
plot(1:5,mean(prob(6:95,1:5)))
hold on;
plot(1:5,mean(prob))

title('Average test error for various data sizes')
xlabel('Number of data points (00s)')
ylabel('Error')
grid on
legend('90% Confidence','Average')