function X = non_linear_lsq_br(AXorXA,A,Asvd,B,X,lambda,itermax)

U = Asvd.U;
S = Asvd.S;
V = Asvd.V;

omega = 1.0;
if strcmp(AXorXA,'AX') % min_X || B - f(A X) || frobenius norm regularizer
  a = ones(size(X,2),1);
  Xa = [X;a'];
  usv = low_rank_appl(A(:,1:end-1)'*A(:,1:end-1),1e-14);
  for k = 1:itermax
    R = B - appl_f(A*Xa);
    X = X + omega * (V * (S \ (U' * R)));
    I = eye(size(A,2)-1);
    X = (I+lambda*(usv.V * (usv.S \usv.U')))\X;
    X = appl_f(X); % non-negative constraint;
    Xa = [X;a'];
  end
elseif strcmp(AXorXA,'XA') % min_X || B - f(X A) || frobenius norm regularizer
  a = ones(size(A,2),1);
  Aa = [A;a'];
  for k = 1:itermax
    R = B - appl_f(X*Aa);
    X = X + omega * (((R * V) / S) * U');
    ss = diag(S);
    ss = ss.^2./(lambda+ss.^2);
    X = X*(U*diag(ss)*U');
  end
end
