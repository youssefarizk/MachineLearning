function [y] = classify2(x)

    for i = 1:size(x,1)
        if x(i,2) >= x(i,1)^2 + x(i,1) + 0.1
            y(i) = 1;
        else
            y(i) = -1;
        
        end
    end
end
