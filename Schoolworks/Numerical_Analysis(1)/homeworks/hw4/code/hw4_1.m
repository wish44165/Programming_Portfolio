%%
clear; clc;
close all;

%% interval [a, b]
a = 0;
b = 1;

%% functions
%uexact = @(x) x.^4+c1*x.^3+c2*x.^2+c3*x+c4;
%f_x = @(x) 12*x.^2;
f_x = @(x) x;

%% Dirichlet condition
%alpha = uexact(a);
%beta = uexact(b);
alpha = 0;
beta = 0;

%% intervals
n_interval = 6;
m = n_interval - 1;
h = (b-a)/(m-1);

%% xj
x0 = 0;
xm = 1;
x_i = (a+h:h:b-h)';

%% matrix A
A = (-2)*eye(m);
A(1,1) = A(1,1)*(h^2)/(-2);
A(m,m) = A(m,m)*(h^2)/(-2);

for i = 2:m-1
    A(i, i-1) = 1;
    A(i, i+1) = 1;
end
A = A/(h^2);

%% matrix B=A^-1
B = zeros(m);

B(1, 1) = 1-x0;
B(m, m) = xm;

for i = 2:m-1
    B(i, 1) = 1-x_i(i-1);
    B(i, m) = x_i(i-1);
end

for i = 2:m-1
    for j = 2:m-1
        if i<=j
            B(i, j) = h*(x_i(j-1)-1)*x_i(i-1);
        else
            B(i, j) = h*(x_i(i-1)-1)*x_i(j-1);
        end
    end
end

%% vector F
F = zeros(m, 1);
F(1, 1) = alpha;
F(m, 1) = beta;
for i = 2:m-1
    F(i, 1) = f_x(x_i(i-1));
end

%% vector U
U = zeros(m, 1);
U = alpha*B(:, 1) + beta*B(:, m);
for i = 2:m-1
    U = U + F(i)*B(:, i);
end

%% AU = F
Uexact = A\F;

%% plot
% {
hold on
for i=1:m
    if (i==1)|(i==m)
        plot([x0; x_i; xm], B(:,i))
    else
        plot([x0; x_i; xm], B(:,i)/h)
    end
end

plot([x0; x_i; xm], U, 'o-')
legend('G0', 'G_{B(:,2)}', 'G_{B(:,3)}', 'G_{B(:,4)}', 'G1', 'sol = G0+G1+h*G_{B(:,i)}')
xlabel('x')
ylabel('u')
title('Green''s functions and solution')
set(gca,'FontSize', 18)
%}