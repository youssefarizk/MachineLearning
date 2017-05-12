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