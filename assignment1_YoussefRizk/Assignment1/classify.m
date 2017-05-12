function [y] = classify(x)

    for i = 1:size(x,1)
        if x(i,3) >= x(i,2) + 0.1
            y(i) = 1;
        else
            y(i) = -1;
        
        end
    end
end
