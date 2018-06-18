clear all;
di=10;
tt=2^di;
DATA=zeros(tt,di);
for i=1:tt
    a=dec2bin(i-1,di)-48;
    DATA(i,:)=a;
end
% x=0.05:0.05:2;
% y=0.05:0.05:2;
% Dset=zeros(length(x),length(y),2);
A=DATA;
Y=sum(DATA,2);
L=10;
j=0;

% for i=1:200*200
%         Dset(i,j,1)=x(i);
% end
x=0:0.01:2;
y=0:0.01:2;
norm_loss=zeros(length(x),length(y));
for j=1:length(y)
    for i=1:length(x)
        WZ(1).Wre=y(i);
        WZ(1).Win=x(j);
        WZ_tmp = compute_z_rnn(A,WZ,L);
        Z = WZ_tmp(L).Z;
        %norm_loss(i,j) = 1e-6+norm((Y'-Z),'fro')^2/norm(Y','fro')^2;
        norm_loss(i,j) = 1e-6+norm((Y'-Z),'fro')^2/norm(Y','fro')^2;
    end
end

s=surf(x,y,norm_loss);
set(gca,'Zscale','log')
caxis([-0,1])
s.EdgeColor = 'none';

lossort=sort(norm_loss(:));
lossort(1:10)