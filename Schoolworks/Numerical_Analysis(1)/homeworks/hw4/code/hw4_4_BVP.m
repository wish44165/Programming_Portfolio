function [e0, e1, e2] = hw4_4_BVP(uexact, du, myf, a, b, n_interval);
%{
a = 0;    %x in [a,b]
b = 1;
n_interval = 10;
uexact = @(x) x.^4+x;
myf = @(x) 12*x.^2;
%}
m = n_interval - 1;
h = (b-a)/(m+1);

sigma = du(a);
beta = uexact(b);

xj = (a:h:b)';

A = -2*eye(m+2);
A(1,1) = A(1,1)*h/2;
A(1,2) = h;
A(m+2,m+2) = h^2;
for i = 2:m+1
    A(i,i-1)=1;
end
for ii = 2:m+1
    A(ii, ii+1) = 1;
end
A = A/(h^2);

F = zeros(m+2, 1);
F(2:m+1) = myf(xj(2:m+1));
F(1) = sigma;
F(m+2) = beta;

U = A\F;    %numerical solution
Uex = uexact(xj);    %exact solution
E = U - Uex;    %error
e0 = max(abs(E));    %sup-norm
e1 = h*sum(abs(E));    %1-norm
e2 = sqrt(h*sum(E.^2));    %2-norm
end