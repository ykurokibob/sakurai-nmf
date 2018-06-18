function X = non_linear_lsq_r(AXorXA,A,Asvd,B,X,lambda,itermax)

U = Asvd.U;
S = Asvd.S;
V = Asvd.V;

omega = 1.0;
if strcmp(AXorXA,'AX') % min_X || B - f(A X) ||
  AtA = A'*A;
  usv = low_rank_appl(AtA,1e-15);
    for k = 1:itermax
        R = B - appl_f(A*X);
        X = X + omega * (V * (S \ (U' * R)));
        
        [m,n] = size(AtA);
        I = eye(m,n);
        %X = (I+lambda*inv(AtA))\X;
        X = (I+lambda*(usv.V*(usv.S\usv.U')))\X;
        
        %ss = diag(S);
        %ss = ss.^2./(lambda+ss.^2);
        %[m,n] = size(V);
        %I = eye(m);
        %VVt_norm = norm(I-V*V',2);
        %X = (V*diag(ss)*V')*X;
        
        X = appl_f(X); % non-negative constraint
    end
    
elseif strcmp(AXorXA,'XA') % min_X || B - f(X A) ||
    
    for k = 1:itermax
        R = B - appl_f(X*A);
        X = X + omega * (((R * V) / S) * U');
        
        %AAt = A*A';
        %[m,n] = size(AAt);
        %I = eye(m,n);
        %X = X/(I+lambda*inv(AAt));
        ss = diag(S);
        ss = ss.^2./(lambda+ss.^2);
        [m,n] = size(U);
        I = eye(m);
        X = X*(U*diag(ss)*U');
        %UUt_norm = norm(I-U*U',2)

    end
    
end
