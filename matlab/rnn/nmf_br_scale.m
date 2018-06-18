function [U,V] = nmf_br_scale(Type,A,U,V,lambda,iter1,iter2,iter3,delta)

a = ones(size(V,2),1); %行列Uの制約
n = size(V,2); %次元サイズ;
Va = [V;a'];
IE = ones(n,n); %全ての要素が1の行列;
k = size(U,2);
flag1 = 'scale_1'; %semi-NMFのscale
flag2 = 'scale_1'; %Nonlinear semi-NMFのscale;
flag3 = 'scale';%NMFのscale

if strcmp(Type,'N_1') %正則化項が1ノルム
  % Normal NMF: min_U,V ||A - ([U b] [V 1])|| s.t. U, V >= 0
  %実装済み
  %＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
  for i = 1:iter1
    U_org = U(:,1:end-1);
    V = V .* (U_org'*A) ./ ((U_org'*U)*Va + beta*V*IE + eps);
    %V = V .* (U_org'*A) ./ ((U_org'*U)*Va + beta*V + eps);
    Va = [V;a'];
    U = U .* (A*Va') ./ (U*(Va*Va') + alpha*U + eps);
  end
  %＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

elseif strcmp(Type,'N_F') %正則化項がFノルム
    for i = 1:iter1
      U_org = U(:,1:end-1);
      V = V .* (U_org'*A) ./ ((U_org'*U)*Va + eps);

      U = U .* (A*Va') ./ (U*(Va*Va') + eps);
      Va = [V;a'];
    end
    
    if strcmp(flag3,'scale_1')
      for j = 1:k-1
        d(j) = sqrt(norm(V(j,:),2)/norm(U(:,j),2));
        if d(j) == 0 || norm(U(:,j),2) == 0
          d(j) = 1;
        end
      end
      d(k) = 1;
      D = diag(d);
    d = d * lambda;
    U = U * D;
    V = D^(-1)*[V;a'];
    V = V(1:end-1,:);

  elseif strcmp(flag3,'scale_2')
    d = sqrt(norm(V,'fro') / norm(U(:,1:end-1),'fro'));
    U(:,1:end-1) = d * U(:,1:end-1);
    V = V/d;
  elseif strcmp(flag3,'scale_3')
    d = sum(sum(abs(U(:,1:end-1))));
    U(:,1:end-1) = d * U(:,1:end-1);
    V = V/d;
  end

    
elseif strcmp(Type,'N_F2') %正則化項がフロベニウスノルムでバイアス
                           %部分が0
  for i = 1:iter1
    U_org = U(:,1:end-1);
    %V = V .* (U_org'*A) ./ ((U_org'*U)*Va + beta*V*IE + eps);
    V = V .* (U_org'*A) ./ ((U_org'*U)*Va + eps);
    Va = [V;a'];
    UU = U;
    UU(:,end) = 0; %バイアス部分が0の行列
    U = U .* (A*Va') ./ (U*(Va*Va') + eps);
  end

elseif strcmp(Type,'S_F') %正則化項がフロベニウスノルム
  % Semi-NMF: min_U,V ||A - U V|| s.t. V >= 0
    %＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
  for i = 1:iter1
    Vsvd = low_rank_appl(Va,delta);
    U = ((A * Vsvd.V) / Vsvd.S) * Vsvd.U';
    %       U = A / V;

    U_org = U(:,1:end-1);
    UtA = U_org'*A;
    UtAp = (abs(UtA) + UtA) / 2;
    UtAm = (abs(UtA) - UtA) / 2;
    UtU = U_org'*U;
    UtUp = (abs(UtU) + UtU) / 2;
    UtUm = (abs(UtU) - UtU) / 2;

    V = V .* sqrt( (UtAp + UtUm*Va) ./ ...
                   (UtAm + UtUp*Va + eps) );
    Va = [V;a'];
    %＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
  end
  
  if strcmp(flag1,'scale_1')
    for j = 1:k-1
      d(j) = sqrt(norm(V(j,:),2)/norm(U(:,j),2));
      if d(j) == 0 || norm(U(:,j),2) == 0
        d(j) = 1;
      end
    end
    d(k) = 1;
    D = diag(d);
    d = d * lambda;
    U = U * D;
    V = D^(-1)*[V;a'];
    V = V(1:end-1,:);

  elseif strcmp(flag1,'scale_2')
    d = sqrt(norm(V,'fro') / norm(U(:,1:end-1),'fro'));
    U(:,1:end-1) = d * U(:,1:end-1);
    V = V/d;
  elseif strcmp(flag1,'scale_3')
    d = sum(sum(abs(U(:,1:end-1))));
    U(:,1:end-1) = d * U(:,1:end-1);
    V = V/d;
  end

elseif strcmp(Type,'S_1') %正則化項が1ノルム
  % Semi-NMF: min_U,V ||A - U V|| s.t. V >= 0
    %＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    for i = 1:iter1
      Vsvd = low_rank_appl(Va,delta);
      R = A - U*Va;
      VVt = Va * Va';
      [m,n] = size(VVt);
      I = eye(m,n);
      %中間層が3層になると警告がでる
      U = U + (((R * Vsvd.V) / Vsvd.S) * Vsvd.U');
      ss = diag(Vsvd.S);
      ss = ss.^2 ./ (alpha+ss.^2);
      U = U*(Vsvd.U*diag(ss)*Vsvd.U');
      %       U = A / V;
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

elseif strcmp(Type,'S_12') %正則化項が1ノルムでバイアス部分が0
  %＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    for i = 1:iter1
      %Vsvd = low_rank_appl(Va,delta);
      R = A - U*Va;
      VVt = Va * Va';
      [m,n] = size(VVt);
      I = eye(m,n);
      %II:単位行列の右下が0の行列
      II = I;
      II(end,:) = 0;
      II(:,end) = 0;
      %中間層が3層になると警告がでる
      %U = U + (((R * Vsvd.V) / Vsvd.S) * Vsvd.U');
      %ss = diag(Vsvd.S);
      %ss = ss.^2 ./ (alpha+ss.^2);
      %U = U*(Vsvd.U*diag(ss)*Vsvd.U');
      %       U = A / V;
      temp = VVt+alpha*II;
      t_svd = low_rank_appl(temp,1e-14);
      %U = (A*Va')/temp;
      U =(((A*Va')*t_svd.V) / t_svd.S) * t_svd.U';

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

elseif strcmp(Type,'NS_F')
  % Nonlinear Semi-NMF: min_U,V ||A - f(U V)|| s.t. V >= 0
  d = zeros(k,1);
  n = size(V,2);
    for i = 1:iter1
        usv = low_rank_appl([V;a'],delta);
        U = non_linear_lsq_br_scale('XA',V,usv,A,U,iter2);
        %U = non_linear_lsq_br_ver2('XA',V,usv,A,U,alpha,iter2);

        usv = low_rank_appl(U(:,1:end-1),delta);
        V = non_linear_lsq_br_scale('AX',U,usv,A,V,iter3);

        Va = [V;a'];
        %V = non_linear_lsq_br_ver2('AX',U,usv,A,V,beta,iter3);
    end
    
    if strcmp(flag2,'scale_1')
      for j = 1:k-1
        d(j) = sqrt(norm(V(j,:),2)/norm(U(:,j),2));
        if d(j) == 0 || norm(U(:,j),2) == 0
          d(j) = 1;
        end
      end
      d(k) = 1;
      D = diag(d);
    d = d * lambda;
    U = U * D;
    V = D^(-1)*[V;a'];
    V = V(1:end-1,:);

  elseif strcmp(flag2,'scale_2')
    d = sqrt(norm(V,'fro') / norm(U(:,1:end-1),'fro'));
    U(:,1:end-1) = d * U(:,1:end-1);
    V = V/d;
  elseif strcmp(flag1,'scale_3')
    d = sum(sum(abs(U(:,1:end-1))));
    U(:,1:end-1) = d * U(:,1:end-1);
    V = V/d;
  end
      
elseif strcmp(Type,'NS2')

  % Nonlinear Semi-NMF: min_U,V ||A - f(U V)|| s.t. V >= 0
    for i = 1:iter1
        usv = low_rank_appl([V;a'],delta);
        U = non_linear_lsq_br('XA2',V,usv,A,U,alpha,iter2);

        usv = low_rank_appl(U(:,1:end-1),delta);
        V = non_linear_lsq_br('AX2',U,usv,A,V,beta,iter3);
    end
    
elseif strcmp(Type,'NS_r')
    lambda = 0;
    for i = 1:iter1
        usv = low_rank_appl(V,delta);
        U = non_linear_lsq_r('XA',V,usv,A,U,lambda,iter2);
        
        usv = low_rank_appl(U,delta);
        V = non_linear_lsq_r('AX',U,usv,A,V,lambda,iter3);
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
