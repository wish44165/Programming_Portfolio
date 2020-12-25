function [e0, e1, e2] = hw4_3_cde_BVP(uexact, myf, a, b, n_interval);
%{
a = 0;    %x in [a,b]
b = 1;
n_interval = 10;
uexact = @(x) x.^4+x;
myf = @(x) 12*x.^2;
%}

m = n_interval-1;
h = (b-a)/(m+1);

alpha = uexact(a);
beta = uexact(b);

xj = (a+h:h:b-h)';

A = (-2)*eye(m);
for ii = 2:m
    A(ii, ii-1) = 1;
    A(ii-1, ii) = 1;
end

F = myf(xj);    %column vector
F(1) = F(1) - alpha/(h^2);
F(m) = F(m) - beta/(h^2);

F((m+1)/2) = 0;    %mistake term

A = A/(h^2);
U = A\F;    %numerical solution
Uex = uexact(xj);    %exact solution
E = U - Uex;    %error
e0 = max(abs(E));    %sup-norm
e1 = h*sum(abs(E));    %1-norm
e2 = sqrt(h*sum(E.^2));    %2-norm
end