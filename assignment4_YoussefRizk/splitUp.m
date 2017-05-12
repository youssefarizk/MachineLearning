function [z1, z2] = splitUp(x)

    z1 = [];
    z2 = [];
    
    for i = 1:size(x,1)
       if x(i,1) == 1
           z1 = vertcat(z1,x(i,:));
       else
           z2 = vertcat(z2,x(i,:));
       end
    end

end