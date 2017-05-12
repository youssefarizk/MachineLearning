function [y] = classify2(x)

    for i = 1:size(x,1)
        if x(i,1) == 1
            y(i) = 1;
        else
            y(i) = -1;
        
        end
    end
end
