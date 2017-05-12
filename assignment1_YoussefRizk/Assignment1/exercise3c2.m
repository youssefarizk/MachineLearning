function exercise3c2(filteredTraining, filteredTest, empiricalClassification, trueClassification)

    p_train = zeros(1, size([1:1:100],2));
    p_test = zeros(1, size([1:1:100],2));

    for i = 1:1:100
       weights = modpercep(filteredTraining,empiricalClassification,i);
       p_train(i) = errorprob(sign(empiricalClassification.*(weights*filteredTraining')));
       p_test(i) = errorprob(sign(trueClassification.*(weights*filteredTest')));
    end   
    
    plot(i,p_train);
    hold on;
    plot(i, p_test);

end