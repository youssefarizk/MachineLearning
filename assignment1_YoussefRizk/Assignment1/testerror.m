function p = testerror(a,b)

   t1 = (a - 1);
   t2 = (b - 0.1);
   x1 = -t1/t2;
   fun = @(x) t1*x + t2;
   
   if (x1 < 1) && (x1 > 0)
      p = abs(integral(fun,0,x1)) + abs(integral(fun,x1,1));
       
   else
      p = abs(integral(fun,0,1))
   end

end