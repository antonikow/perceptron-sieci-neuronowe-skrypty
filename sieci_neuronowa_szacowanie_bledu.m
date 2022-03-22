clc 
close all 
clear all

%parametry nastawialne
m = 1000;
kmax = 10^6;
eta = 10^-1;
N = 50;

%przygotowanie danych uczacych
% rng(2);
x1 = pi*rand(m,1);
x2 = pi*rand(m,1);
y = cos(x1.*x2).*cos(2.*x1);
X = [x1 x2];
subplot(2,1,1)
scatter3(x1,x2,y,'.');
title('zbior probek')

%UCZENIE
%inicalizacja wag
V = randi([-10000,10000],N,size(X,2)+1)/10000000;
W = randi([-10000,10000],N+1, 1)/10000000; 
%petla 
k=0;
while k ~= kmax  
%losuje xi,yi 
rand_idx= randi([1 size(X,1)]);
xi = [1 X(rand_idx,:)];
yi = y(rand_idx);
%obliczenie ymlp
S = V*xi'; 
PHI = 1./(1+exp(-S)); 
ymlp = W(1) + sum(W(2:size(W,1),:).*PHI); 
%poprawa wag
Wnew = W - eta*((ymlp - yi)*[1; PHI]); %ok
Vnew = V - eta*((ymlp - yi)*W(2:size(W,1),:).*(PHI.*(1-PHI))*xi);%ok
W=Wnew;
V=Vnew;
k=k+1;
end
%koniec petli

%przygotowanie danych testowych
m_test = 10000;
x1_test = pi*rand(m_test,1);
x2_test = pi*rand(m_test,1);
y_test = cos(x1_test.*x2_test).*cos(2.*x1_test);
X_test = [x1_test x2_test];

%ODPOWIEDZ
y_pred_test = [];
for i = 1:m_test
    xi = [1 X_test(i,:)];
    yi = y_test(i);
    %obliczenie ymlp
    S = V*xi'; 
    PHI = 1./(1+exp(-S)); 
    ymlp = W(1) + sum(W(2:size(W,1),:).*PHI); 
    y_pred_test = [y_pred_test; ymlp];
end
subplot(2,1,2)
scatter3(x1_test,x2_test,y_pred_test, '.')
title('funkcja aproksymujaca')
fprintf('sredni blad bezwzgledny to %d\n',mean(abs(y_test-y_pred_test)))


