function [U,V] = Snmf(Type,A,U,V,iter1,iter2,iter3,delta)

if Type == 'N' % Normal NMF: min_U,V ||A - U V|| s.t. U, V >= 0
    
    for i = 1:iter1
        V = V .* (U'*A) ./ ((U'*U)*V + eps);
        U = U .* (A*V') ./ (U*(V*V') + eps);
    end
    
elseif Type == 'S' % Semi-NMF: min_U,V ||A - U V|| s.t. V >= 0
    
    for i = 1:iter1
        Vsvd = low_rank_appl(V,1e-14);
        Vsvd.U=single(Vsvd.U);
        Vsvd.S=single(Vsvd.S);
        Vsvd.V=single(Vsvd.V);
        
        U = single(((A * Vsvd.V) / Vsvd.S) * Vsvd.U');
        Vinv=single(((A * Vsvd.V) / Vsvd.S) * Vsvd.U');
        %V*Vinv
%         %       U = A / V;
%         %A * Vsvd.V
%         size(U)
%         
%         size(V)
%         size(A)
%         size(Vsvd)
%         %U*Vsvd
%         
        UtA = U'*A;
        UtAp = (abs(UtA) + UtA) / 2;
        UtAm = (abs(UtA) - UtA) / 2;
        UtU = U'*U;
        UtUp = (abs(UtU) + UtU) / 2;
        UtUm = (abs(UtU) - UtU) / 2;
        
        V = V .* sqrt( (UtAp + UtUm*V) ./ (UtAm + UtUp*V + eps) );
    end
    
elseif strcmp(Type,'NS') % Nonlinear Semi-NMF: min_U,V ||A - f(U V)|| s.t. V >= 0
    
    for i = 1:iter1
        usv = low_rank_appl(V,delta);
        U = non_linear_lsq('XA',V,usv,A,U,iter2);
        
        usv = low_rank_appl(U,delta);
        V = non_linear_lsq('AX',U,usv,A,V,iter3);
    end
    
elseif strcmp(Type,'NS_r')
    lambda1 = 1e-2;
    lambda2 = 1e-7;
    for i = 1:iter1
        usv = low_rank_appl(V,delta);
        %U = non_linear_lsq_r('XA',V,usv,A,U,lambda1,iter2);
        U = non_linear_lsq('XA',V,usv,A,U,iter2);
        
        usv = low_rank_appl(U,delta);
        %V = non_linear_lsq_r('AX',U,usv,A,V,lambda2,iter3);
        V = non_linear_lsq('AX',U,usv,A,V,iter3);
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
