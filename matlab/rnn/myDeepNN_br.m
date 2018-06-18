function [WZ,WZ_train,WZ_val,resvec] = myDeepNN_br(A,Y,A_test,Y_test,param)
tic;
t0 = toc;
tc = 0;
fprintf('-- AutoEncoder ------------------------------------- \n');
fprintf('Iter    sec   norm   train  test \n');

% set parameter
[n,m] = size(A);
L = size(param.hidden,2)+1;

delta = param.delta;
aeitr = param.aeitr;
ftitr = param.ftitr;
nsnmf = param.nsnmf;
batch = param.batch;
lambda = param.lambda;

% AutoEncoder
a = ones(size(A,2),1);
Ausv = low_rank_appl([A;a'],delta(1)); %***???X***
t1 = toc;
fprintf('---------------------------------------------------- \n');
fprintf('Time for SVD of A   : %6.2f [sec.] \n',t1-t0);
fprintf('---------------------------------------------------- \n');

id = randperm(m);
AI = A(:,id(1:batch(1)));
YI = Y(:,id(1:batch(1)));

AIusv.U = Ausv.U;
AIusv.S = Ausv.S;
AIusv.V = Ausv.V(id(1:batch(1)),:);
% i=1;%noAE
% k = param.hidden(i);%noAE
% a = ones(size(AI,2),1);%noAE
% 
% WZ(1).W = zeros(k,size(AI,1)+1);%noAE

WZ(1).W = autoencoder_br(1,AI,AIusv,param); %***???X***
a = ones(size(AI,2),1);
WZ(1).Z = appl_f(WZ(1).W*[AI;a']); %***???X***

for i = 2:L-1
    a = ones(size(WZ(i-1).Z,2),1); %***???X***
    usv = low_rank_appl([WZ(i-1).Z;a'],delta(2));
    WZ(i).W = autoencoder_br(i,WZ(i-1).Z,usv,param); %***???X***; AE
    
%     k = param.hidden(i);%noAE
%     WZ(i).W = zeros(k,size(WZ(i-1).Z,1)+1);%noAE
    
    WZ(i).Z = appl_f(WZ(i).W*[WZ(i-1).Z;a']); %***???X***
end

a = ones(size(WZ(L-1).Z,2),1); %***???X***
usv = low_rank_appl([WZ(L-1).Z;a'],delta(2)); %***???X***
WZ(L).W = YI * usv.V / usv.S * usv.U'; %***???X***
%WZ(L).W = YI / WZ(L-1).Z;

l = 1;
tc0 = toc;
resvec(l,:) = check_br(0,toc-t0,A,A_test,Y,Y_test,WZ,L); tc = tc + (toc-tc0);

t2 = toc;
fprintf('---------------------------------------------------- \n');
fprintf('Time for AutoEncoder: %6.2f [sec.] \n',t2-t1);
fprintf('-- Iteration --------------------------------------- \n');
fprintf('Iter    sec   norm   train  test \n');

% Alternative minimization iteration (Back propagation)
for itr = 1:ftitr(1)
    
    id = randperm(m);
    for kk = 1:m/batch(2)
        ids = (kk-1)*batch(2)+1;
        ide = kk*batch(2);
        AI = A(:,id(ids:ide));
        YI = Y(:,id(ids:ide));
        AIusv.V = Ausv.V(id(ids:ide),:);
        
        WZ = compute_z_br(AI,WZ,L); %***???X***
        [WZ(L).W,WZ(L-1).Z] = nmf_br('S_F',YI,WZ(L).W,WZ(L-1).Z, ...
            lambda(1),lambda(2),ftitr(2),0,0,delta(2)); %***???X***
        
        for i = L-2:-1:1
            %nmf???????????????????????????????????????????
            [WZ(i+1).W,WZ(i).Z] = nmf_br('NS_F',WZ(i+1).Z,WZ(i+1).Wre, ...
                WZ(i).Z,lambda(1),lambda(2),ftitr(3),nsnmf(1),nsnmf(2),delta(2)); %***???X***
            [WZ(i+1).Win,WZ(i).Z] = nmf_br('NS_F',WZ(i+1).Z,WZ(i+1).Win, ...
                WZ(i).Z,lambda(1),lambda(2),ftitr(3),nsnmf(1),nsnmf(2),delta(2)); %***???X***
        end
        
        WZ(1).W = non_linear_lsq_br('XA',AI,AIusv,WZ(1).Z, WZ(1).W,lambda(1),nsnmf(1)); %***???X***
        % function X = non_linear_lsq_rnn(AXorXA,A,Asvd,B,X,lambda,itermax)
        
        %%-----7/26????--------------------------------------------
        %         WZ = compute_z_br(AI,WZ,L);
        %         WZ = compute_w_br(AI,WZ,L,lambda,nsnmf(1));
        %%---------------------------------------------------------
        l = l + 1;
        resvec(l,:) = check_br(itr,toc-t0,A,A_test,Y,Y_test,WZ,L);
    end
    %     plot(A,Y,'bo')
    %     hold on
    %     output = compute_z_br(A_test,WZ,L);
    %     plot(A_test,output(L).Z,'r');
    %     drawnow;
    %     pause(0.2)
    %     hold off
end

WZ_train = compute_z_br(A,WZ,L);
WZ_val = compute_z_br(A_test,WZ,L);

t3 = toc;
fprintf('---------------------------------------------------- \n');
fprintf('Time for iteration:   %6.2f [sec.] \n',t3-t1);
fprintf('Total time:           %6.2f [sec.] \n',t3-t0);
fprintf('Total time - check:   %6.2f [sec.] \n',t3-t0-tc);
fprintf('---------------------------------------------------- \n');

end
