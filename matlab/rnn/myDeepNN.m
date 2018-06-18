function [WZ,resvec] = myDeepNN(A,Y,A_test,Y_test,param)
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

% AutoEncoder
Ausv = low_rank_appl(A,delta(1));
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

i=1;%noAE
k = param.hidden(i);%noAE
a = ones(size(AI,2),1);%noAE

WZ(1).W = zeros(k,size(AI,1));%noAEnobr

%WZ(1).W = autoencoder(1,AI,AIusv,param);
[row,col] = size(WZ(1).W);
WZ(1).W = randn(row,col);
WZ(1).Z = appl_f(WZ(1).W*AI);

for i = 2:L-1
    usv = low_rank_appl(WZ(i-1).Z,delta(2));
    %WZ(i).W = autoencoder(i,WZ(i-1).Z,usv,param);
         k = param.hidden(i);%noAE
     WZ(i).W = zeros(k,size(WZ(i-1).Z,1));%noAEnobr

    [row,col] = size(WZ(i).W);
    WZ(i).W = randn(row,col);
    WZ(i).Z = appl_f(WZ(i).W*WZ(i-1).Z);
end

usv = low_rank_appl(WZ(L-1).Z,delta(2));
WZ(L).W = YI * usv.V / usv.S * usv.U';
%WZ(L).W = YI / WZ(L-1).Z;

l = 1;
tc0 = toc; resvec(l,:) = check(0,toc-t0,A,A_test,Y,Y_test,WZ,L); tc = tc + (toc-tc0);

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
        
        WZ = compute_z(AI,WZ,L);
        [WZ(L).W,WZ(L-1).Z] = nmf('S',YI,WZ(L).W,WZ(L-1).Z,ftitr(2),0,0,0);
        
        for i = L-2:-1:1
          %nmf???????????????????????????????????????????
          [WZ(i+1).W,WZ(i).Z] = nmf('NS',WZ(i+1).Z,WZ(i+1).W,WZ(i).Z,ftitr(3),nsnmf(1),nsnmf(2),delta(2));
        end
        
        WZ(1).W = non_linear_lsq('XA',AI,AIusv,WZ(1).Z,WZ(1).W,nsnmf(1));
    
		l = l + 1;
		resvec(l,:) = check(itr,toc-t0,A,A_test,Y,Y_test,WZ,L);
        
        %plot(A,Y,'bo','MarkerFaceColor','b');
        %hold on
        %plot(AI,YI,'bo','MarkerFaceColor','b')
        %output = compute_z(A_test,WZ,L);
        %plot(A_test,output(L).Z,'r');
        %plot(A_test,Y_test,'k');
        %ylim([-1.2,1.2]);
%         ylim([min(Y)*1.2,max(Y)*1.2]);
%         xlabel('x');
%         ylabel('y');
%         %legend('train data','minibatch data','output','truth')
%         legend('train data','output','truth')
%         grid on
%         drawnow;
%         pause(0.5)
%         hold off
    end
end


t3 = toc;
fprintf('---------------------------------------------------- \n');
fprintf('Time for iteration:   %6.2f [sec.] \n',t3-t1);
fprintf('Total time:           %6.2f [sec.] \n',t3-t0);
fprintf('Total time - check:   %6.2f [sec.] \n',t3-t0-tc);
fprintf('---------------------------------------------------- \n');

end
