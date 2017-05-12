function y = actClassify(x)

y = zeros(1,size(x,1));

func = @(x) x*(x-1)*(x-2);

for i = 1:size(x,1)
    if x(i,2) >= feval(func,x(i,1))
        y(i) = 1;
    else
        y(i) = -1;
        
        
    end
end
end
