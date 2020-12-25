%%
clear; clc;
close all;

%% interval [a, b]
a = 0;
b = 1;

%% Neumann & Dirichlett conditions
sigma = 0;
beta = 1;

%% intervals
n_interval = 4;
m = n_interval - 1;
h = (b-a)/(m+1);

%% xj
x_i = (a:h:b)';

%% matrix A
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

%% matrix B = A^-1
B = zeros(m+2);

for i = 1:m+2
    B(i, 1) = x_i(i)-1;
    B(i, m+2) = 1;
end

for i = 1:m+1
    for j = 1:m+1
        if i<=j
            B(i, j) = h*(x_i(j)-1);
        else
            B(i, j) = h*(x_i(i)-1);
        end
    end
end
