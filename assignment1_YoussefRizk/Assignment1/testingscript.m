X = [ones(100,1) rand(100,2)];
Y = simpleClassify(X);

scatter(X(:,2),X(:,3))
hold on
X1 = 0.5*ones(100,1);
plot(X(:,2),X1);
hold on

w = percep(X,Y);
w = w/w(3)
Y1 = -w(2)*X(:,2) - w(1);
plot(X(:,2),Y1);
legend('Data Points','Actual Separator','Calculated Separator');
title('Testing the Perceptron Algorithm');
xlabel('x1')
ylabel('x2')