%function [U,V] = nmf_rnn(Type,A,U,V,alpha,beta,iter1,iter2,iter3,delta)
function [Uall,V] = nmf_rnn(Type,A,Uall,V,alpha,beta,iter1,iter2,iter3,delta)
%fprintf('daaa')
%a = ones(size(V,2),1); %?s??U??????
n = size(V,2); %?????T?C?Y;
Va = V;
sizeV=size(V);
IE = ones(n,n); %?S?????v?f??1???s??;
%size(U)
%size(V)
%size(A)
%A1=A(1,2:10);
%sizeA=size(A);
%A=A(1,1);
if strcmp(Type,'N_1') %???????????t???x?j?E?X?m????
    % Normal NMF: min_U,V ||A - ([U b] [V 1])|| s.t. U, V >= 0
    %????????
    %????????????????????????????????????????????
    for i = 1:iter1
        U_org = U(:,1:end-1);
        V = V .* (U_org'*A) ./ ((U_org'*U)*Va + beta*V*IE + eps);
        %V = V .* (U_org'*A) ./ ((U_org'*U)*Va + beta*V + eps);
        Va = [V;a'];
        U = U .* (A*Va') ./ (U*(Va*Va') + alpha*U + eps);
    end
    %????????????????????????????????????????????
    
elseif strcmp(Type,'N_F2') %???????????t???x?j?E?X?m???????o?C?A?X
    %??????0
    for i = 1:iter1
        U_org = U(:,1:end-1);
        %V = V .* (U_org'*A) ./ ((U_org'*U)*Va + beta*V*IE + eps);
        V = V .* (U_org'*A) ./ ((U_org'*U)*Va + beta*V*I + eps);
        Va = [V;a'];
        UU = U;
        UU(:,end) = 0; %?o?C?A?X??????0???s??
        U = U .* (A*Va') ./ (U*(Va*Va') + alpha*UU + eps);
    end
    
elseif strcmp(Type,'S_F') %???????????t???x?j?E?X?m????
    % Semi-NMF: min_U,V ||A - U V|| s.t. V >= 0
    %????????????????????????????????????????????
    %function [Uall,V] = nmf_rnn(Type,A,Uall,V,alpha,beta,iter1,iter2,iter3,delta,j)
    for i = 1:iter1
        %pre_U = U;%8/9????
        %pre_V = V;
        Vsvd = low_rank_appl(V,delta);
        size(A');
        Uallsize=size(Uall);
        size(Va);
%        R = A' - Uall*Va;
        R = A' - Uall*Va;
        VVt = Va * Va';
        [m,n] = size(VVt);
        I = eye(m,n);
        %?????w??3?w?????????x????????
%         sizeV=size(Vsvd.V);
%         sizeS=size(Vsvd.S);
%         sizeU=size(Vsvd.U);
%         Vsvd.U;
%         sizeR=size(R);
        %(R * Vsvd.V)/Vsvd.S;
%         size(Uall);
        Uall = Uall + (((R * Vsvd.V)/Vsvd.S) * Vsvd.U');
        Uallsize2=size(Uall);
        ss = diag(Vsvd.S);
        ss = ss.^2 ./ (alpha+ss.^2);
        
        size(Vsvd.U);
        size(diag(ss));
        size(Vsvd.U');
        %size(Vsvd.U*diag(ss)*Vsvd.U')
        
        Uall = Uall*(Vsvd.U*diag(ss)*Vsvd.U');
        %U = U*(Vsvd.U*diag(ss)*Vsvd.U');
        %D = pre_U - U;%8/9????
        %U = pre_U - 0.5*D;%8/9????
        %       U = A / V;
        if size(Uall,1)>1
            U_orgall = Uall(:,1:end-1);
        else
            U_orgall = Uall(:,1);
        end
        UtAall = U_orgall;
        
        UtApall = (abs(UtAall) + UtAall) / 2;
        UtAmall = (abs(UtAall) - UtAall) / 2;
        
        UtUall = U_orgall'*Uall;
        UtUpall = (abs(UtUall) + UtUall) / 2;
        UtUmall = (abs(UtUall) - UtUall) / 2;
        %print('start')
%        sizeA=size(A);
 %       sizeUall=size(Uall);
        sizeV=size(V);
        size(UtApall);
        size(UtUmall*V);
        beta;
        betaV=size(beta*V);
        size(UtAmall);
        size(UtUpall*V);
        size(eps);
        (UtApall + UtUmall*V);% + (beta*V));
        %(UtAmall + UtUpall*V + (beta*V)); %+ [eps,eps]);

        %kari=((UtApall + UtUmall*V + beta*V) ./ (UtAmall + UtUpall*V +
        %beta*V ));biasver
        %kari=((UtApall + UtUmall*V) ./ (UtAmall + UtUpall*V  +ones(size(UtAmall + UtUpall*V))*eps));
         %sizeV=size(V);
         %sizekari=size(kari);
        %V(2,:) = V(2,:) .* sqrt( kari);%+ eps) );
        %V(:,:) = V(:,:) .*
        V(end,:)=V(end,:).*sqrt((UtApall + UtUmall*V) ./ (UtAmall + UtUpall*V  +ones(size(UtAmall + UtUpall*V))*eps));%+ eps) );
        %????????????????????????????????????????????
    end
elseif strcmp(Type,'NS_F')
    % Nonlinear Semi-NMF: min_U,V ||A - f(U V)|| s.t. V >= 0
    for i = 1:iter1
        usv = low_rank_appl(V,1e-14);
        %function X = non_linear_lsq_rnn(AXorXA,A,Asvd,B,X,lambda,itermax)
        %sizeA=size(A)
        %A(2,:)
        Uallsize=size(Uall);
        Vsize=size(V);
        Uall = non_linear_lsq_rnn('XA',V,usv,A,Uall,alpha,iter2);
        %  X = non_linear_lsq_rnn(AXorXA,A,Asvd,B,X,lambda,itermax)
        usvall = low_rank_appl(Uall,1e-14);
        %usvre = low_rank_appl(Ure,0);
        V = non_linear_lsq_rnn('AX',Uall,usvall,A,V,beta,iter3);%+ non_linear_lsq_rnn('AX',Ure,usvre,A,V,beta,iter3);
        
    end    
elseif strcmp(Type,'S_1') %??????????1?m????
    % Semi-NMF: min_U,V ||A - U V|| s.t. V >= 0
    %????????????????????????????????????????????
    for i = 1:iter1
        Vsvd = low_rank_appl(Va,0);
        R = A - U*Va;
        VVt = Va * Va';
        [m,n] = size(VVt);
        I = eye(m,n);
        %?????w??3?w?????????x????????
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
        %????????????????????????????????????????????
    end
    
elseif strcmp(Type,'S_12') %??????????1?m???????o?C?A?X??????0
    %????????????????????????????????????????????
    for i = 1:iter1
        Vsvd = low_rank_appl(Va,0);
        R = A - U*Va;
        VVt = Va * Va';
        [m,n] = size(VVt);
        I = eye(m,n);
        %II:?P???s?????E????0???s??
        II = I;
        II(end,:) = 0;
        II(:,end) = 0;
        %?????w??3?w?????????x????????
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
        %????????????????????????????????????????????
    end
    

    
elseif strcmp(Type,'NS2')
    
    % Nonlinear Semi-NMF: min_U,V ||A - f(U V)|| s.t. V >= 0
    for i = 1:iter1
        usv = low_rank_appl([V;a'],delta);
        U = non_linear_lsq_br('XA2',V,usv,A,U,alpha,iter2,j);
        
        usv = low_rank_appl(U(:,1:end-1),delta);
        V = non_linear_lsq_br('AX2',U,usv,A,V,beta,iter3,j);
    end
    
elseif strcmp(Type,'NS_r')
    lambda = 0;
    for i = 1:iter1
        usv = low_rank_appl(V,delta);
        U = non_linear_lsq_r('XA',V,usv,A,U,lambda,iter2,j);
        
        usv = low_rank_appl(U,delta);
        V = non_linear_lsq_r('AX',U,usv,A,V,lambda,iter3,j);
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
