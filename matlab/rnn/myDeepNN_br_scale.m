function [WZ,resvec] = myDeepNN_br_scale(A,Y,A_test,Y_test,param)
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
Ausv = low_rank_appl([A;a'],delta(1)); %***変更***
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

WZ(1).W = autoencoder_br_scale(1,AI,AIusv,param); %***変更***
%WZ(1).W = autoencoder_br(1,AI,AIusv,param);

%autoencoderなし--------------
[row,col] = size(WZ(1).W);
WZ(1).W = randn(row,col);
%------------------------------

a = ones(size(AI,2),1);
WZ(1).Z = appl_f(WZ(1).W*[AI;a']); %***変更***

for i = 2:L-1
  a = ones(size(WZ(i-1).Z,2),1); %***変更***
  usv = low_rank_appl([WZ(i-1).Z;a'],delta(2));
  WZ(i).W = autoencoder_br_scale(i,WZ(i-1).Z,usv,param); %***変更%***;
  %WZ(i).W = autoencoder_br(i,WZ(i-1).Z,usv,param);
  
  %autoencoderなし--------------
  [row,col] = size(WZ(i).W);
  WZ(i).W = randn(row,col);
  %------------------------------
  
  WZ(i).Z = appl_f(WZ(i).W*[WZ(i-1).Z;a']); %***変更***
end

a = ones(size(WZ(L-1).Z,2),1); %***変更***
usv = low_rank_appl([WZ(L-1).Z;a'],delta(2)); %***変更***
WZ(L).W = YI * usv.V / usv.S * usv.U'; %***変更***
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
        
      WZ = compute_z_br(AI,WZ,L); %***変更***
      [WZ(L).W,WZ(L-1).Z] = nmf_br_scale('S_F',YI,WZ(L).W,WZ(L-1).Z,lambda ,...
                                   ftitr(2),0,0,delta(2)); %***変更
                                                           %***
        for i = L-2:-1:1
          %nmf縺悟他縺ｰ繧後◆縺ｨ縺阪?陦悟?繧剃ｿ晏ｭ倥☆繧?
          [WZ(i+1).W,WZ(i).Z] = nmf_br_scale('NS_F',WZ(i+1).Z,WZ(i+1).W, ...
                                       WZ(i).Z,lambda,ftitr(3),nsnmf(1),nsnmf(2),delta(2)); %***変更***
        end
        
        %        WZ(1).W = non_linear_lsq_br('XA',AI,AIusv,WZ(1).Z,WZ(1).W,1e-3,nsnmf(1)); %***変更***
       WZ(1).W = non_linear_lsq_br_scale('XA',AI,AIusv,WZ(1).Z,WZ(1).W,nsnmf(1)); %***変更***
    
       l = l + 1;
        
       resvec(l,:) = check_br(itr,toc-t0,A,A_test,Y,Y_test,WZ,L);
       plot(A,Y,'bo')
        hold on
        plot(AI,YI,'bo','MarkerFaceColor','b')
        output = compute_z_br(A_test,WZ,L);
        plot(A_test,output(L).Z,'r');
        plot(A_test,Y_test,'k');
        ylim([-1.2,1.2]);
        legend('train data','minibatch data','output','truth')
        drawnow;
        pause(0.5)
        hold off
    end
end


t3 = toc;
fprintf('---------------------------------------------------- \n');
fprintf('Time for iteration:   %6.2f [sec.] \n',t3-t1);
fprintf('Total time:           %6.2f [sec.] \n',t3-t0);
fprintf('Total time - check:   %6.2f [sec.] \n',t3-t0-tc);
fprintf('---------------------------------------------------- \n');

end
