%This is a function that takes the whole unfiltered trainind data set and
%return only those elements that characterize either 2 or 8

function [out_2, out_8] = filter2(x)

    out_2 = [];
    out_8 = [];
    
    for i = 1:size(x,1)
        
        a = x(i,:); 
        if (a(1) == 2)
            out_2 = vertcat(out_2,a);
        elseif (a(1) == 8)
            out_8 = vertcat(out_8,a);
        end
        
    end 
    
end
