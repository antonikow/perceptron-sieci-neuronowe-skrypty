clc 
close all 
clear all


I = 200; %liczba punktow danych
m = 100; %liczba centrow
eta = 0.5;
sigma = 0.3;

rx = 2*pi* rand(I,1);
ry = randi([-10000,10000],I,1)/10000;
X = [rx ry];
%ustalenie klas
Y = zeros(I,1);
for i = 1:I
    if (X(i,2) <= sin(X(i,1)) &&  (X(i,2) >= -sin(X(i,1)))) || (X(i,2) <= -sin(X(i,1)) &&  (X(i,2) >= sin(X(i,1))))
        Y(i,1) = 1;
    else
        Y(i,1) = -1;
    end
end
%transformacja do przedzialu od -1 do 1
X(:,1) = 2*((X(:,1))-min(X(:,1)))/(max(X(:,1))-min(X(:,1)))-1;



%generowanie centrow
C = [];
for i = 1: m
    C = [C; randi([-10000,10000])/10000 randi([-10000,10000])/10000];
end

%wykres danych
x1 = X(:,1);
x2 = X(:,2);
subplot(2,1,1)
scatter(x1(Y==1),x2(Y==1) , 200, 'b', '.')
hold on
scatter(x1(Y==-1),x2(Y==-1) , 200, 'r', '.')
hold on
scatter(C(:,1), C(:,2), 500, 'g', '.')


Z = zeros(I, m);

for i = 1 : I
    for j = 1 : m
        Z(i,j) = exp(-sum((X(i,:) -  C(j,:)).^2)/(2*sigma^2));
        
    end
end

D = [Z Y];
warStopu = 5000;
[w, k] = PerceptronLearningRule(D, eta, warStopu);
w;

subplot(2,1,2)
xx = linspace(-1,1, 100);
yy = linspace(-1,1, 100);
[XX, YY] = meshgrid(xx,yy);
Xgr = [XX(:) YY(:)];

Zgr = zeros(size(Xgr,1), m);
for i = 1 : size(Zgr,1)
    for j = 1 : m
        Zgr(i,j) = exp(-( (sum((Xgr(i,:) -  C(j,:)).^2)) )/(2*sigma^2));
    end
end
Zgr = [ones(size(Zgr,1), 1) Zgr];
size(Xgr);
size(Zgr);
size(w);
size(sum(Zgr.*w, 2));

suma = sum(Zgr.*w, 2);
suma(suma > 0) = 1;
suma(suma <= 0) = -1;
ZZ = reshape(suma, size(XX,1), size(XX,2)); 

surf(XX,YY,ZZ)

function [wFinal, k] = PerceptronLearningRule(D, ni, warStopu)
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

       if (size(E,1) == 0) || (k == warStopu)
           wFinal = W(k,:);
           break
       end
       
       idxRand = randi([1, size(E,1)]);
       W(k+1,:) = W(k,:) + ni*Ey(idxRand,1)*E(idxRand,:);
       k=k+1;   
   end
   k
end
