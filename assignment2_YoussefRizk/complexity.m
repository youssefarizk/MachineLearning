%calculates the complexity term

function c = complexity(n, H, delta)
    
    c = 8*H/n * log(2*n * exp(1)/H);
    c = c + 8/n * log(4 / (0.2 * delta));
    c = sqrt(c);