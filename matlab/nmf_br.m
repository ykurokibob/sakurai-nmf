function [U,V] = nmf_br(Type,A,U,V,alpha,beta,iter1,iter2,iter3,delta)

a = ones(size(V,2),1);
n = size(V,2);
Va = [V;a'];
IE = ones(n,n);

if strcmp(Type,'N_1') %L1-norm regularizer
  % Normal NMF: min_U,V ||A - ([U b] [V 1])|| s.t. U, V >= 0
  for i = 1:iter1
    U_org = U(:,1:end-1);
    V = V .* (U_org'*A) ./ ((U_org'*U)*Va + beta*V*IE + eps);
    %V = V .* (U_org'*A) ./ ((U_org'*U)*Va + beta*V + eps);
    Va = [V;a'];
    U = U .* (A*Va') ./ (U*(Va*Va') + alpha*U + eps);
  end
elseif strcmp(Type,'N_F') %frobenius norm regularizer

  for i = 1:iter1
      U_org = U(:,1:end-1);
      V = V .* (U_org'*A) ./ ((U_org'*U)*Va + beta*V + eps);
      Va = [V;a'];
      U = U .* (A*Va') ./ (U*(Va*Va') + alpha*U + eps);
  end

elseif strcmp(Type,'N_F2') %frobenius norm regularizer without bias

  for i = 1:iter1
    U_org = U(:,1:end-1);
    %V = V .* (U_org'*A) ./ ((U_org'*U)*Va + beta*V*IE + eps);
    V = V .* (U_org'*A) ./ ((U_org'*U)*Va + beta*V*I + eps);
    Va = [V;a'];
    UU = U;
    UU(:,end) = 0; %バイアス部分が0の行列
    U = U .* (A*Va') ./ (U*(Va*Va') + alpha*UU + eps);
  end

elseif strcmp(Type,'S_F') %正則化項がフロベニウスノルム
  % Semi-NMF: min_U,V ||A - U V|| s.t. V >= 0

    for i = 1:iter1
      Vsvd = low_rank_appl(Va,delta);
      R = A - U*Va;
      %VVt = Va * Va';
      %[m,n] = size(VVt);
      %I = eye(m,n);
      U = U + (((R * Vsvd.V) / Vsvd.S) * Vsvd.U');
      ss = diag(Vsvd.S);
      ss = ss.^2 ./ (alpha+ss.^2);
      U = U*(Vsvd.U*diag(ss)*Vsvd.U');
      U_org = U(:,1:end-1);
      UtA = U_org'*A;
      UtAp = (abs(UtA) + UtA) / 2;
      UtAm = (abs(UtA) - UtA) / 2;
      UtU = U_org'*U;
      UtUp = (abs(UtU) + UtU) / 2;
      UtUm = (abs(UtU) - UtU) / 2;

      V = V .* sqrt( (UtAp + UtUm*Va + beta*V) ./ ...
      (UtAm + UtUp*Va + beta*V + eps) );
      Va = [V;a'];
    end

elseif strcmp(Type,'S_1') %L1-norm regularizer
  % Semi-NMF: min_U,V ||A - U V|| s.t. V >= 0
    %＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    for i = 1:iter1
      Vsvd = low_rank_appl(Va,delta);
      R = A - U*Va;
      %VVt = Va * Va';
      %[m,n] = size(VVt);
      %I = eye(m,n);
      U = U + (((R * Vsvd.V) / Vsvd.S) * Vsvd.U');
      ss = diag(Vsvd.S);
      ss = ss.^2 ./ (alpha+ss.^2);
      U = U*(Vsvd.U*diag(ss)*Vsvd.U');
      U_org = U(:,1:end-1);
      UtA = U_org'*A;
      UtAp = (abs(UtA) + UtA) / 2;
      UtAm = (abs(UtA) - UtA) / 2;
      UtU = U_org'*U;
      UtUp = (abs(UtU) + UtU) / 2;
      UtUm = (abs(UtU) - UtU) / 2;

      V = V .* sqrt( (UtAp + UtUm*Va + beta*V*IE) ./ ...
      (UtAm + UtUp*Va + beta*V*IE + eps) );
      Va = [V;a'];
      %＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    end

elseif strcmp(Type,'NS_F') %frobenius norm regularizer
  % Nonlinear Semi-NMF: min_U,V ||A - f(U V)|| s.t. V >= 0
    for i = 1:iter1
        usv = low_rank_appl([V;a'],delta);
        U = non_linear_lsq_br('XA',V,usv,A,U,alpha,iter2);

        usv = low_rank_appl(U(:,1:end-1),delta);
        V = non_linear_lsq_br('AX',U,usv,A,V,beta,iter3);
    end

elseif strcmp(Type,'KL') % NMF on KL divergence: min_U,V d(A,UV) s.t. U, V >= 0
    
    for i = 1:iter1
        D = diag(1./(sum(U)+eps));
        V = V .* ((D * U') * (A ./ (U*V+eps)));

        D = diag(1./(sum(V,2)+eps));
        U = U .* ((A ./ (U*V+eps)) * (V' * D));
    end

end

end
