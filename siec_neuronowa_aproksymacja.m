clc 
close all 
clear all

%parametry nastawialne
m = 1000;
kmax = 10^6;
eta = 10^-1;
N = 50;

%przygotowanie danych
% rng(2);
x1 = pi*rand(m,1);
x2 = pi*rand(m,1);
y = cos(x1.*x2).*cos(2.*x1);
X = [x1 x2];
D = [X y];
subplot(2,2,1)
scatter3(x1,x2,y,'.');
title('zbior probek')

xx1=linspace(0,pi,100);
xx2=linspace(0,pi,100);
[xx1m,xx2m]=meshgrid(xx1,xx2);
yy2m=cos(xx1m.*xx2m).*cos(2.*xx1m);
subplot(2,2,2)
surf(xx1m,xx2m,yy2m);
title('funkcja aproksymowana')

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

%ODPOWIEDZ
y_pred = [];
for i = 1:m
    xi = [1 X(i,:)];
    yi = y(i);
    %obliczenie ymlp
    S = V*xi'; 
    PHI = 1./(1+exp(-S)); 
    ymlp = W(1) + sum(W(2:size(W,1),:).*PHI); 
    y_pred = [y_pred; ymlp];
end
subplot(2,2,3)
scatter3(x1,x2,y_pred, '.')
title('funkcja aproksymujaca')


subplot(2,2,4)
scatter3(x1,x2,y,'.')
hold on
scatter3(x1,x2,y_pred, '.')
legend('f_aproksymowana', 'f_aproksymujaca', 'Location','southwest')
title('nalozenie wykresow na siebie')


