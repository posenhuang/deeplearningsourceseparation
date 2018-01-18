function x = vec(X)

[m n]= size(X);
x= reshape(X, m*n, 1);