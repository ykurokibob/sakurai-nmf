function X = non_linear_lsq(AXorXA,A,Asvd,B,X,itermax)

U = Asvd.U;
S = Asvd.S;
V = Asvd.V;

omega = 1.0;
if strcmp(AXorXA,'AX') % min_X || B - f(A X) ||
    
    for k = 1:itermax
        R = B - appl_f(A*X);
        X = X + omega * (V * (S \ (U' * R)));
        X = appl_f(X); % non-negative constraint
    end
    
elseif strcmp(AXorXA,'XA') % min_X || B - f(X A) ||
    
    for k = 1:itermax
        R = B - appl_f(X*A);
        X = X + omega * (((R * V) / S) * U');
    end
    
end
