%This is a function that takes the whole unfiltered trainind data set and
%return only those elements that characterize either 2 or 8

function [y1, ym1] = filter1(x)

    y1 = [];
    ym1 = [];
    
    for i = 1:size(x,1)
        
        a = x(i,:); 
        if (a(3) == 1)
            y1 = vertcat(y1,a);
        else
            ym1 = vertcat(ym1,a);
        end
        
    end 
    
end
