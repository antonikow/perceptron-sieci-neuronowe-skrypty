clc 
close all 
clear all

%parametry nastawialne
m = 200;
kmax = 10^6;
eta = 10^-1;
%przygotowanie danych
% rng(2);
x1 = pi*rand(m,1);
x2 = pi*rand(m,1);
srednia = 0;
odchylenie = 0.2;
blad_losowy = odchylenie*randn(m,1) + srednia; 
y = cos(x1.*x2).*cos(2.*x1) + blad_losowy;
X = [x1 x2];
%czesc uczaca i testowa
P = 0.7;
idx = randperm(m);
X_train = X(idx(1:round(P*m)),:); 
y_train = y(idx(1:round(P*m)),:);
X_test = X(idx(round(P*m)+1:end),:) ;
y_test = y(idx(round(P*m)+1:end),:) ;


ERR = zeros(10,2);
for it = 1:10
N = 10*it;
%UCZENIE
%inicalizacja wag
V = randi([-10000,10000],N,size(X_train,2)+1)/10000000;
W = randi([-10000,10000],N+1, 1)/10000000; 
%petla 
k=0;
while k ~= kmax  
%losuje xi,yi 
rand_idx= randi([1 size(X_train,1)]);
xi = [1 X_train(rand_idx,:)];
yi = y_train(rand_idx);
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
%KONIEC UCZENIA

%ODPOWIEDZ zb uczacy
y_train_pred = [];
for i = 1:size(X_train,1)
    xi = [1 X_train(i,:)];
    yi = y_train(i);
    %obliczenie ymlp
    S = V*xi'; 
    PHI = 1./(1+exp(-S)); 
    ymlp = W(1) + sum(W(2:size(W,1),:).*PHI); 
    y_train_pred = [y_train_pred; ymlp];
end
 ERR(it,1)=mean(abs(y_train - y_train_pred));
%ODPOWIEDZ zb testowy
y_test_pred = [];
for i = 1:size(X_test,1)
    xi = [1 X_test(i,:)];
    yi = y_test(i);
    %obliczenie ymlp
    S = V*xi'; 
    PHI = 1./(1+exp(-S)); 
    ymlp = W(1) + sum(W(2:size(W,1),:).*PHI); 
    y_test_pred = [y_test_pred; ymlp];
end
 ERR(it,2)=mean(abs(y_test-y_test_pred));
end
plot(10:10:100,ERR(:,1))
hold on
plot(10:10:100,ERR(:,2))
legend('blad uczacy', 'blad testowy')
ERR
[min_err, opt_liczba_neuronow_idx]=min(ERR(:,2));  %%podmieniec na min!!!!!
fprintf('optymalna liczba neuronow to %d\n',opt_liczba_neuronow_idx*10)


%NA OPTYMALNEJ LICZBIE NEURONOW
%UCZENIE
%inicalizacja wag
N=opt_liczba_neuronow_idx*10;
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
figure
scatter3(x1,x2,y_pred, '.')
title('funkcja aproksymujaca')


