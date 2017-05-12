n = 1000;
x10 = generateRandPoints(n);
y10 = classify(x10);

size1 = size(x10,1);

%transformed inputs
x10_0 = pF0(x10(:,1),x10(:,2),size1);
x10_1 = pF1(x10(:,1),x10(:,2),size1);
x10_2 = pF2(x10(:,1),x10(:,2),size1);
x10_3 = pF3(x10(:,1),x10(:,2),size1);
x10_4 = pF4(x10(:,1),x10(:,2),size1);


%weights of transformed inputs
px10_0 = modpercep(x10_0,y10,100);
px10_1 = modpercep(x10_1,y10,100);
px10_2 = modpercep(x10_2,y10,100);
px10_3 = modpercep(x10_3,y10,10000);
px10_4 = modpercep(x10_4,y10,10000);

testsize = size(test,1);

%test transformation
test_0 = pF0(test(:,1),test(:,2),testsize);
test_1 = pF1(test(:,1),test(:,2),testsize);
test_2 = pF2(test(:,1),test(:,2),testsize);
test_3 = pF3(test(:,1),test(:,2),testsize);
test_4 = pF4(test(:,1),test(:,2),testsize);

p0_train = errorprob(sign(px10_0*x10_0').*y10);
p1_train = errorprob(sign(px10_1*x10_1').*y10);
p2_train = errorprob(sign(px10_2*x10_2').*y10);
p3_train = errorprob(sign(px10_3*x10_3').*y10);
p4_train = errorprob(sign(px10_4*x10_4').*y10);

p0_test = errorprob(sign(px10_0*test_0').*testClass);
p1_test = errorprob(sign(px10_1*test_1').*testClass);
p2_test = errorprob(sign(px10_2*test_2').*testClass);
p3_test = errorprob(sign(px10_3*test_3').*testClass);
p4_test = errorprob(sign(px10_4*test_4').*testClass);
%normalizing the weights
px10_1 = px10_1/px10_1(3);
px10_2 = px10_2/px10_2(4);
px10_3 = px10_3/px10_3(5);
px10_4 = px10_4/px10_4(6);

figure;
N = [x10 y10'];

[z1, z2] = filter1(N);

scatter(z1(:,1),z1(:,2),'rx')
hold on;
scatter(z2(:,1),z2(:,2),'bx')
%plotting the line x2 = ... in terms of x1
plot([0:0.05:2.5],(-px10_1(1) - [px10_1(2)]*[0:0.05:2.5]),'LineWidth',3)
plot([0:0.05:2.5],(-px10_2(1) - [px10_2(2)]*[0:0.05:2.5] - [px10_2(3)]*[0:0.05:2.5].^2),'LineWidth',3)
plot([0:0.05:2.5],(-px10_3(1) - [px10_3(2)]*[0:0.05:2.5] - [px10_3(3)]*[0:0.05:2.5].^2 - [px10_3(4)]*[0:0.05:2.5].^3),'LineWidth',3)
plot([0:0.05:2.5],(-px10_4(1) - [px10_4(2)]*[0:0.05:2.5] - [px10_4(3)]*[0:0.05:2.5].^2 - [px10_4(4)]*[0:0.05:2.5].^3 - [px10_4(5)]*[0:0.05:2.5].^4),'LineWidth',3)
axis([0,2.5,-1,2])
legend('data','data','linear','second','third','fourth')

figure;


plot([0:4],[mean(p0_train) mean(p1_train) mean(p2_train) mean(p3_train) mean(p4_train)])
hold on;
plot([0:4],[mean(p0_test) mean(p1_test) mean(p2_test) mean(p3_test) mean(p4_test)])

