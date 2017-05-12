%creates a nx3 matrix of random numbers separated by a margin gamma
function y = filterMargin(n,gamma)
    

    y = zeros(n,3);
    
    for i = 1:n
    found = false;
    while ~found
        
        a = [ones(1,1) rand(1,2)];
        if abs(a(3) - a(2) - 0.1) > gamma
            y(i,:) = a;
            found = true;
        end
        
    end 
    end

end