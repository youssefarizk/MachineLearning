%exercise 3b4
n = 100;

for margin = [0.3 0.1 0.01 0.001 0]
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
    hold on;
end

grid on;
title('Test error vs. number of data points for various margin sizes');
xlabel('Number of data points (00s)');
ylabel('Error');
legend('Margin = 0.3','Margin = 0.1','Margin = 0.01','Margin = 0.001','Margin = 0');