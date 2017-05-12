%modperceptron test

X = [ones(1000,1) rand(1000,2)];
Y = classify1(X(:,2:3));
p = zeros(1,100);

for n = 1:100
    w = modpercep(X,Y,n);
    p(n) = errorprob(sign(w*X').*Y);
end
disp(p)