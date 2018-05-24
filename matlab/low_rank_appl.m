function Asvd = low_rank_appl(A,delta)

[U,S,V] = svd(A,'econ');
s = diag(S);
s = s/s(1);
k = sum(s>delta);

Asvd.U = U(:,1:k);
Asvd.S = S(1:k,1:k);
Asvd.V = V(:,1:k);

end
