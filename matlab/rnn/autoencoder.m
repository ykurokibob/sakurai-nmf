function W1 = autoencoder(i,A,Asvd,param)

[n,m] = size(A);
k = param.hidden(i);

iter  = param.aeitr(2);
delta = param.delta(2);
aeitr = param.aeitr;

U = rand(n,k);
V = rand(k,m);

%U = sqrt(6/(n+k))*2*(rand(n,k)-0.5);
%W = sqrt(6/(n+k))*2*(rand(k,n)-0.5);
%U = appl_f(U);
%V = appl_f(W*A);
k = param.hidden(i);
W1 = zeros(k,size(A,1));
for k = 1:aeitr(1)
[U,V] = nmf('N',A,U,V,iter,0,0,delta);
W1 = non_linear_lsq('XA',A,Asvd,V,W1,aeitr(3));
V = appl_f(W1*A);
end 

end
