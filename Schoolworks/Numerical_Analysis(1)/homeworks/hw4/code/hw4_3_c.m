%%
clear; clc;
close all;

%% interval [a, b]
a = -1;
b = 1;

%% functions
uexact = @(x) exp(x);
myf = @(x) exp(x);    %uexact''

%%
ni = 4;
h_his = zeros(ni, 1);
err_0 = zeros(ni, 1);
err_1 = zeros(ni, 1);
err_2 = zeros(ni, 1);
ord_0 = zeros(ni, 1);
ord_1 = zeros(ni, 1);
ord_2 = zeros(ni, 1);

for ii = 1:ni
    n = 10^ii;
    h_his(ii) = (b-a)/n;
    [err_0(ii), err_1(ii), err_2(ii)] = hw4_3_cde_BVP(uexact, myf, a, b, n);
end

fprintf('   h         max-norm       ratio       1-norm        ratio       2-norm        ratio\n');
fprintf('%7.5f %15.6e %9.3f % 15.6e %9.3f %15.6e %9.3f\n', h_his(1), err_0(1), ord_0(1), err_1(1), ord_1(1), err_2(1), ord_2(1));
for ii = 1:ni-1
    ratio_h = h_his(ii)/h_his(ii+1);
    
    ratio_0 = err_0(ii)/err_0(ii+1);
    ord_0(ii+1) = log(ratio_0)/log(ratio_h);
    
    ratio_1 = err_1(ii)/err_1(ii+1);
    ord_1(ii+1) = log(ratio_1)/log(ratio_h);
    
    ratio_2 = err_2(ii)/err_2(ii+1);
    ord_2(ii+1) = log(ratio_2)/log(ratio_h);
    fprintf('%7.5f %15.6e %9.3f %15.6e %9.3f %15.6e %9.3f\n',h_his(ii+1), err_0(ii+1), ord_0(ii+1), err_1(ii+1), ord_1(ii+1), err_2(ii+1), ord_2(ii+1));
end
loglog(h_his, err_0, 'ks:', h_his, err_1, 'ro--', h_his, err_2, 'bx-.')    %loglog
xlabel('h')
ylabel('Error')
legend('Max-norm', '1-norm', '2-norm')
title('f(x) = e^x, a = -1, \alpha = e^{-1}, b = 1, \beta = e^1, F((m+1)/2)=0')
set(gca,'FontSize', 18)