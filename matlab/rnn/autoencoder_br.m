function W1 = autoencoder_br(i,A,Asvd,param)

[n,m] = size(A);
k = param.hidden(i);

iter  = param.aeitr(2);
delta = param.delta(2);
aeitr = param.aeitr;
lambda = param.lambda;

U = rand(n,k+1); %??????????????
V = rand(k,m);
a = ones(size(A,2),1);
%U = sqrt(6/(n+k))*2*(rand(n,k)-0.5);
%W = sqrt(6/(n+k))*2*(rand(k,n)-0.5);
%U = appl_f(U);
%V = appl_f(W*A);

W1 = zeros(k,size(A,1)+1);
for k = 1:aeitr(1)
  [U,V] = nmf_br('N_F',A,U,V,lambda(1),lambda(2),iter,0,0,delta);
  %[U,V] = nmf_br('N_F',A,U,V,0,0,iter,0,0,delta);
  W1 = non_linear_lsq_br('XA',A,Asvd,V,W1,0,aeitr(3));
  V = appl_f(W1*[A;a']);
end

end
% function W1 = autoencoder_br(i,A,Asvd,param)
% 
% [n,m] = size(A);
% k = param.hidden(i);
% 
% iter  = param.aeitr(2);
% delta = param.delta(2);
% aeitr = param.aeitr;
% lambda = param.lambda;
% 
% U = rand(n,k+1); %?o?C?A?X?x?N?g???????????s??
% V = rand(k,m);
% a = ones(size(A,2),1);
% %U = sqrt(6/(n+k))*2*(rand(n,k)-0.5);
% %W = sqrt(6/(n+k))*2*(rand(k,n)-0.5);
% %U = appl_f(U);
% %V = appl_f(W*A);
% 
% W1 = zeros(k,size(A,1)+1);
% for k = 1:aeitr(1)
%   [U,V] = nmf_br('N_1',A,U,V,lambda(1),lambda(2),iter,0,0,delta);
%   W1 = non_linear_lsq_br('XA',A,Asvd,V,W1,0,aeitr(3));
%   V = appl_f(W1*[A;a']);
% end
% 
% end
