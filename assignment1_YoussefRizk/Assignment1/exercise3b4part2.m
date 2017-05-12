%exercise 3.b.4

n = 100;

X = filterMargin(n,0.2);
Y = classify(X);

[A, count] = theoreticalPercep(X,Y)

normvector = zeros(1,n);
minvector = zeros(1,n);

for i = 1:n
   normvector(i) = norm(X(i,:));
   minvector(i) = A*X(i,:)'*Y(i);
end

maxNorm = max(normvector);
minRho = min(minvector);

upperBound = (norm(A)^2*maxNorm)/minRho^2
