function p = errorprob(y)

    p = 0;
    for i = 1:size(y,2)
        if y(i) == -1
            p = p+1;
        end
    end
    p = p/(size(y,2));
    
end