%This is a function that takes the whole unfiltered trainind data set and
%return only those elements that characterize either 2 or 8

function [y] = filterFor1(x)

    y = [];
    
    for i = 1:size(x,1)
        
        a = x(i,:); 
        if (a(1) == 1)
            y = vertcat(y,a);
        end
        
    end 
    
end
