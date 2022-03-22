clc 
close all 
clear all

%parametry nastawialne
m = 121;
eta = 0.5;
gamma = 0.2;
%przygotowanie danych
%  rng(1);
X = [randn(m, 2)];
Y = ones(m,1);
%przesuniecie
X(:,1) = X(:,1) + randi([-900, 900])/100;
X(:,2) = X(:,2) + randi([-900, 900])/100;
%wyznaczenie klas
srodek = mean(X(:,1));
for i = 1 : m
    if srodek < X(i,1) 
        Y(i) = 1;
        X(i,1) = X(i,1) + gamma/2;
    else
        Y(i) = -1;
         X(i,1) = X(i,1) - gamma/2;
    end
end
%obrot
theta = randi(359);
R = [cosd(theta) -sind(theta);
    sind(theta) cosd(theta)];
X=X*R;
D = [X Y];

%wykres
x1 = X(:,1);
x2 = X(:,2);
scatter(x1(Y==-1), x2(Y==-1), 200, 'b', '.')
hold on
scatter(x1(Y==1), x2(Y==1), 200, 'r', '.')
[w, k] = PerceptronLearningRule(D, eta);
%narysowanie prostej separacji
x = [min(x1):0.1:max(x1)];
f = @(w, x) (-w(1,1) - w(1,2)*x)/w(1,3);
f(w,x);
hold on
scatter(x,f(w,x))

function [wFinal, k] = PerceptronLearningRule(D, ni)
    X = [ones(size(D,1),1) D(:, 1:size(D,2)-1)];
    Y = D(:, size(D,2));
    W = zeros(1,size(X,2));
    k = 1;
   while 1
       S = sum(W(k,:).*X, 2);
       Iloczyn = Y.*S;
       zleIdx = find(Iloczyn <= 0);
       E = X(zleIdx,:);
       Ey = Y(zleIdx,:); %klasy zle sklasyfikowanych probek

       if size(E,1) == 0 
           wFinal = W(k,:);
           break
       end
       
       idxRand = randi([1, size(E,1)]);
       W(k+1,:) = W(k,:) + ni*Ey(idxRand,1)*E(idxRand,:);
       k=k+1;   
   end
   k
end