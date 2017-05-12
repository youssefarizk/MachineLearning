clear all;
test = generateRandPoints(100000);
testClass = classify(test);
%
% n = 10;
%
% x10 = generateRandPoints(n);
% y10 = classify(x10);
% N = [x10 y10'];
%
% [z1, z2] = filter1(N);
%
% scatter(z1(:,1),z1(:,2),'rx')
% hold on;
% scatter(z2(:,1),z2(:,2),'bx')
% title('10 training points')
%
% % figure;
% n = 100;
%
% x100 = generateRandPoints(n);
% y100 = classify(x100);
% N = [x100 y100'];
%
% [z1, z2] = filter1(N);
%
% scatter(z1(:,1),z1(:,2),'rx')
% hold on;
% scatter(z2(:,1),z2(:,2),'bx')
% title('100 training points')
%
% % figure;
% n = 10000;
%
% x10000 = generateRandPoints(n);
% y10000 = classify(x10000);
% N = [x10000 y10000'];
%
% [z1, z2] = filter1(N);
%
% scatter(z1(:,1),z1(:,2),'rx')
% hold on;
% scatter(z2(:,1),z2(:,2),'bx')
% title('10000 training points')

%%Declaring the features

pF0 = @(x1,x2,sizex) [ones(sizex, 1) x2];

pF1 = @(x1,x2,sizex) [ones(sizex, 1) x1 x2];

pF2 = @(x1,x2,sizex) [ones(sizex, 1) x1 x1.^2 x2];

pF3 = @(x1,x2,sizex) [ones(sizex, 1) x1 x1.^2 x1.^3 x2];

pF4 = @(x1,x2,sizex) [ones(sizex, 1) x1 x1.^2 x1.^3 x1.^4 x2];

m = 100; %controls the number of iterations over which average is taken


for n = [10 100 1000 10000];
    
    
    
    ptrainSRM = zeros(1,m);
    ptestSRM = zeros(1,m);
    
    p0_train = zeros(1,m);
    p1_train = zeros(1,m);
    p2_train = zeros(1,m);
    p3_train = zeros(1,m);
    p4_train = zeros(1,m);
    
    p0_test = zeros(1,m);
    p1_test = zeros(1,m);
    p2_test = zeros(1,m);
    p3_test = zeros(1,m);
    p4_test = zeros(1,m);
    
    %terms consisting of training error and complexity
    pSRM0 = zeros(1,m);
    pSRM1 = zeros(1,m);
    pSRM2 = zeros(1,m);
    pSRM3 = zeros(1,m);
    pSRM4 = zeros(1,m);
    
    weight0 = zeros(m,2);
    weight1 = zeros(m,3);
    weight2 = zeros(m,4);
    weight3 = zeros(m,5);
    weight4 = zeros(m,6);
    %% 100 iterations to achieve average results
    for i = 1:m
        globalError = 100;
        x10 = generateRandPoints(n);
        y10 = classify(x10);
        
        size1 = size(x10,1);
        
        %transformed inputs
        x10_0 = pF0(x10(:,1),x10(:,2),size1);
        x10_1 = pF1(x10(:,1),x10(:,2),size1);
        x10_2 = pF2(x10(:,1),x10(:,2),size1);
        x10_3 = pF3(x10(:,1),x10(:,2),size1);
        x10_4 = pF4(x10(:,1),x10(:,2),size1);
        
        
        %ERM weights of transformed inputs
        px10_0 = modpercep(x10_0,y10,100);
        px10_1 = modpercep(x10_1,y10,100);
        px10_2 = modpercep(x10_2,y10,100);
        px10_3 = modpercep(x10_3,y10,100);
        px10_4 = modpercep(x10_4,y10,100);
        
        
        testsize = size(test,1);
        
        %test transformation
        test_0 = pF0(test(:,1),test(:,2),testsize);
        test_1 = pF1(test(:,1),test(:,2),testsize);
        test_2 = pF2(test(:,1),test(:,2),testsize);
        test_3 = pF3(test(:,1),test(:,2),testsize);
        test_4 = pF4(test(:,1),test(:,2),testsize);
        
        
        %training error
        p0_train(i) = errorprob(sign(px10_0*x10_0').*y10);
        p1_train(i) = errorprob(sign(px10_1*x10_1').*y10);
        p2_train(i) = errorprob(sign(px10_2*x10_2').*y10);
        p3_train(i) = errorprob(sign(px10_3*x10_3').*y10);
        p4_train(i) = errorprob(sign(px10_4*x10_4').*y10);
        
        %test error
        p0_test(i) = errorprob(sign(px10_0*test_0').*testClass);
        p1_test(i) = errorprob(sign(px10_1*test_1').*testClass);
        p2_test(i) = errorprob(sign(px10_2*test_2').*testClass);
        p3_test(i) = errorprob(sign(px10_3*test_3').*testClass);
        p4_test(i) = errorprob(sign(px10_4*test_4').*testClass);
        
        %normalizing the weights and determining SRM solution
        px10_0 = px10_0/px10_0(2);
        pSRM0(i) = errorprob(sign(px10_0*x10_0').*y10) + complexity(n,1,0.1);
        if pSRM0(i) < globalError
            w_SRM = px10_0;
            globalError = pSRM0(i);
            ptrainSRM(i) = errorprob(sign(px10_0*x10_0').*y10);
            ptestSRM(i) = errorprob(sign(px10_0*test_0').*testClass);
        end
        px10_1 = px10_1/px10_1(3);
        pSRM1(i) = errorprob(sign(px10_1*x10_1').*y10) + complexity(n,2,0.1);
        if pSRM1(i) < globalError
            w_SRM = px10_1;
            globalError = pSRM1(i);
            ptrainSRM(i) = errorprob(sign(px10_1*x10_1').*y10);
            ptestSRM(i) = errorprob(sign(px10_1*test_1').*testClass);
        end
        px10_2 = px10_2/px10_2(4);
        pSRM2(i) = errorprob(sign(px10_2*x10_2').*y10) + complexity(n,3,0.1);
        if pSRM2(i) < globalError
            w_SRM = px10_2;
            globalError = pSRM2(i);
            ptrainSRM(i) = errorprob(sign(px10_2*x10_2').*y10);
            ptestSRM(i) = errorprob(sign(px10_2*test_2').*testClass);
        end
        px10_3 = px10_3/px10_3(5);
        pSRM3(i) = errorprob(sign(px10_3*x10_3').*y10) + complexity(n,4,0.1);
        if pSRM3(i) < globalError
            w_SRM = px10_3;
            globalError = pSRM3(i);
            ptrainSRM(i) = errorprob(sign(px10_3*x10_3').*y10);
            ptestSRM(i) = errorprob(sign(px10_3*test_3').*testClass);
        end
        px10_4 = px10_4/px10_4(6);
        pSRM4(i) = errorprob(sign(px10_4*x10_4').*y10) + complexity(n,5,0.1);
        if pSRM4(i) < globalError
            w_SRM = px10_4;
            globalError = pSRM4(i);
            ptrainSRM(i) = errorprob(sign(px10_4*x10_4').*y10);
            ptestSRM(i) = errorprob(sign(px10_4*test_4').*testClass);
        end
        
        %w_SRM is the SRM
        disp(w_SRM);
        
        weight0(i,:) = px10_0;
        weight1(i,:) = px10_1;
        weight2(i,:) = px10_2;
        weight3(i,:) = px10_3;
        weight4(i,:) = px10_4;
        
    end
    
    %% averaging and normalizing the weights to plot them
    weight0 = mean(weight0);
    weight1 = mean(weight1);
    weight2 = mean(weight2);
    weight3 = mean(weight3);
    weight4 = mean(weight4);
    
    
    weight0 = weight0/weight0(2);
    weight1 = weight1/weight1(3);
    weight2 = weight2/weight2(4);
    weight3 = weight3/weight3(5);
    weight4 = weight4/weight4(6);
    
    %% plotting the last iteration as an example of what the sample looks
    %%like
    figure;
    N = [x10 y10'];
    
    [z1, z2] = filter1(N);
    
    scatter(z1(:,1),z1(:,2),'rx')
    hold on;
    scatter(z2(:,1),z2(:,2),'bx')
    %plotting the line x2 = ... in terms of x1
    
    plot([0:0.05:2.5],-weight0(1)*ones(1,length([0:0.05:2.5])),'LineWidth',3)
    plot([0:0.05:2.5],(-weight1(1) - [weight1(2)]*[0:0.05:2.5]),'LineWidth',3)
    plot([0:0.05:2.5],(-weight2(1) - [weight2(2)]*[0:0.05:2.5] - [weight2(3)]*[0:0.05:2.5].^2),'LineWidth',3)
    plot([0:0.05:2.5],(-weight3(1) - [weight3(2)]*[0:0.05:2.5] - [weight3(3)]*[0:0.05:2.5].^2 - [weight3(4)]*[0:0.05:2.5].^3),'LineWidth',3)
    plot([0:0.05:2.5],(-weight4(1) - [weight4(2)]*[0:0.05:2.5] - [weight4(3)]*[0:0.05:2.5].^2 - [weight4(4)]*[0:0.05:2.5].^3 - [weight4(5)]*[0:0.05:2.5].^4),'LineWidth',3)
    plot([0:0.05:2.5],(2*[0:0.05:2.5] - 3*[0:0.05:2.5].^2 + [0:0.05:2.5].^3),'LineWidth',3)
    axis([0,2.5,-1,2])
    legend('data','data','constant','linear','second','third','fourth', 'actual')
    grid on;
    title(sprintf('Different degree classifiers for %d points', n))
    xlabel('X1')
    ylabel('X2')
    
    figure;
    
    scatter(z1(:,1),z1(:,2),'rx')
    hold on;
    scatter(z2(:,1),z2(:,2),'bx')   
    plot([0:0.05:2.5],(-weight3(1) - [weight3(2)]*[0:0.05:2.5] - [weight3(3)]*[0:0.05:2.5].^2 - [weight3(4)]*[0:0.05:2.5].^3),'LineWidth',3)
    plot([0:0.05:2.5],(-weight4(1) - [weight4(2)]*[0:0.05:2.5] - [weight4(3)]*[0:0.05:2.5].^2 - [weight4(4)]*[0:0.05:2.5].^3 - [weight4(5)]*[0:0.05:2.5].^4),'LineWidth',3)
    plot([0:0.05:2.5],(2*[0:0.05:2.5] - 3*[0:0.05:2.5].^2 + [0:0.05:2.5].^3),'LineWidth',3)
    axis([0,2.5,-1,2])
    legend('data','data','third','fourth', 'actual')
    grid on;
    title(sprintf('Different degree classifiers for %d points', n))
    xlabel('X1')
    ylabel('X2')
    
    %% plotting training and test errors
    figure;
    
    
    plot([0:4],[mean(p0_train) mean(p1_train) mean(p2_train) mean(p3_train) mean(p4_train)])
    hold on;
    plot([0:4],[mean(p0_test) mean(p1_test) mean(p2_test) mean(p3_test) mean(p4_test)])
    plot([0:4],[mean(pSRM0) mean(pSRM1) mean(pSRM2) mean(pSRM3) mean(pSRM4)])
    plot([0:4],[complexity(n,1,0.1) complexity(n,2,0.1) complexity(n,3,0.1) complexity(n,4,0.1) complexity(n,5,0.1)])
    hold off;
    title(sprintf('Training and testing error for ERM of each Hypothesis Class for %d points', n))
    grid on;
    legend('Training error','Test error','VC bound','Complexity term')
    xlabel('Hypothesis Class (q)')
    ylabel('Error')
    
    %% determining average training and test error of SRM solution
    disp(mean(ptrainSRM));
    disp(mean(ptestSRM));
end


