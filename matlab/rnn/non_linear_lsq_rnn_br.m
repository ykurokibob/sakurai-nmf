function X = non_linear_lsq_rnn_br(AXorXA,A,Asvd,B,X,lambda,itermax)

U = Asvd.U;
S = Asvd.S;
V = Asvd.V;

size(U);
omega = 1;
if strcmp(AXorXA,'AX') % min_X || B - f(A X) || ?t???x?j?E?X?m????
    for k = 1:itermax
        R = B - appl_f(A*X);
        X = X + omega * (V * (S \ (U' * R)));
        X = appl_f(X); % non-negative constraint
    end
    
    %     %fprintf('AX')
    %     %a = ones(size(X,2),1);
    %     %Xa = [X;a'];
    %     Xa = X;
    %     usv = low_rank_appl(A(:,1:end-1)'*A(:,1:end-1),1e-14);
    %     %AA=A'*A
    %     %usv = low_rank_appl(A'*A,0);
    %     %usv = low_rank_appl(A,0);
    %     for k = 1:itermax
    %         %pre_X = X; %????8/9
    %         R = B - appl_f(A*Xa);
    %         size(X);
    %         X = X + omega * (V * (S \ (U * R)));
    %         %size(A,2)
    %         I = eye(size(A,2));
    % %         sizeV=size(usv.V);
    % %         sizeS=size(usv.S) ;
    % %         sizeU=size(usv.U);
    %         %size(X);
    %         %lambda*(usv.V * (usv.S \usv.U'))
    %
    %         sa=(I+lambda*(usv.V * (usv.S \usv.U')));
    %         X=sa\X;
    %         %X = (I+lambda*(usv.V * (usv.S \usv.U')))\X;
    %         %D = pre_X - X; %????
    %         %X = pre_X -0.5*D; %????
    %         X = appl_f(X); % non-negative constraint;
    %         % Xa = [X;a'];
    %         Xa = X;
    %     end
elseif strcmp(AXorXA,'XA') % min_X || B - f(X A) || ?t???x?j?E?X?m
    %fprintf('XA')
    % ????
    %a = ones(size(A,2),1);
    %  Aa = [A;a'];
    Aa = A;
    for k = 1:itermax
        k=k;
        %pre_X = X; %????8/9
        sizeB=size(B)
        sizeX=size(X)
        sizeAa=size(Aa)
        
        R = B - appl_f(X*Aa);
        sizeR=size(R)
        sizeV=size(V)
        sizeS=size(S)
        sizeU=size(U)
        omega=ones(size((((R * V) / S) * U)))'
        sizeRVSU=(((R * V) / S) * U)
        %         X = X + omega * (((R * V) / S) * U);
        %         sizeX1=size(X);
        %         ss = diag(S);
        %         ss = ss.^2./(lambda+ss.^2);
        % %        X = X*(U(1,1)*diag(ss)*U(1,1));
        %         sizex=size(X)
        %         sizess=size(diag(ss))
        %         sizeUssU=U*diag(ss)*U
        %         X = X*(U*diag(ss)*U);
        %         X2=X;
        %         sizeX2=size(X2);
        %         %D = pre_X - X; %????
        %         %X = pre_X -0.5*D; %????
    end
elseif strcmp(AXorXA,'AX2') % min_X || B - f(A X) || 1?m????
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
elseif strcmp(AXorXA,'XA2') % min_X || B - f(X A) || ?t???x?j?E?X?m
    % ???? ?o?C?A?X??0
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
