function X = non_linear_lsq_br_scale(AXorXA,A,Asvd,B,X,itermax)

U = Asvd.U;
S = Asvd.S;
V = Asvd.V;

omega = 1.0;
if strcmp(AXorXA,'AX') % min_X || B - f(A X) || フロベニウスノルム
  a = ones(size(X,2),1);
  Xa = [X;a'];
  usv = low_rank_appl(A(:,1:end-1)'*A(:,1:end-1),1e-14);
  for k = 1:itermax
    R = B - appl_f(A*Xa);
    X = X + omega * (V * (S \ (U' * R)));
    X = appl_f(X); % non-negative constraint;
    Xa = [X;a'];
  end
elseif strcmp(AXorXA,'XA') % min_X || B - f(X A) || フロベニウスノ
                           % ルム
 %計算の工夫あり
 % a = ones(size(A,2),1);
 % Aa = [A;a'];
 % for k = 1:itermax
 %   R = B - appl_f(X*Aa);
 %   X = X + omega * (((R * V) / S) * U');
 %   ss = diag(S);
 %   ss = ss.^2./(lambda+ss.^2);
 %   X = X*(U*diag(ss)*U');
 % end
  
  %計算の工夫なし
  a = ones(size(A,2),1);
  Aa = [A;a'];
  for k = 1:itermax
    R = B - appl_f(X*Aa);
    X = X + omega * (((R * V) / S) * U');
   end
  
elseif strcmp(AXorXA,'AX2') % min_X || B - f(A X) || 1ノルム
  a = ones(size(X,2),1);
  Xa = [X;a'];
  usv = low_rank_appl(A(:,1:end-1)'*A(:,1:end-1),1e-14);
  for k = 1:itermax
    R = B - appl_f(A*Xa);
    X = X + omega * (V * (S \ (U' * R)));
    I = eye(size(A,2)-1);
    IE = ones(size(I,1));
    %X = (I+lambda*(usv.V * (usv.S \usv.U')))\X;
    X = (I+lambda*(usv.V * (usv.S \usv.U'))*IE)\X;
    X = appl_f(X); % non-negative constraint;
    Xa = [X;a'];
  end
elseif strcmp(AXorXA,'XA2') % min_X || B - f(X A) || フロベニウスノ
                            % ルム バイアスが0
  a = ones(size(A,2),1);
  Aa = [A;a'];
  usv = low_rank_appl(Aa*Aa',1e-14);
  I = eye(size(Aa,1));
  II = I;
  II(end,end) = 0;
  for k = 1:itermax
    R = B - appl_f(X*Aa);
    X = X + omega * (((R * V) / S) * U');
    X = X/(I+lambda*I*(usv.V*(usv.S \ usv.U')));
  end
end
