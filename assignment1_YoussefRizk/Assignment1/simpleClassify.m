function [y] = simpleClassify(x)

    for i = 1:size(x,1)
       
        if x(i,3) > 0.5
            y(i) = 1;
        else 
            y(i) = -1;
        end
        
    end    

end