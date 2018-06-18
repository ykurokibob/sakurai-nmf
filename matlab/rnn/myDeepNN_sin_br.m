function [WZ,resvec,F] = myDeepNN_sin_br(A,Y,A_test,Y_test,param)
tic;
t0 = toc;
tc = 0;
fprintf('-- AutoEncoder ------------------------------------- \n');
fprintf('Iter    sec   loss   val_loss   train  test  norm_pro  \n');

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
%if Af==1
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
AIusv.V = Ausv.V;
%AIusv.V = Ausv.V(id(1:batch(1)),:);

WZ(1).W = autoencoder_br(1,AI,AIusv,param); %***???X***
[row,col] = size(WZ(1).W);
WZ(1).W = randn(row,col);
a = ones(size(AI,2),1);
WZ(1).Z = appl_f(WZ(1).W*[AI;a']); %***???X***

for i = 2:L-1
  a = ones(size(WZ(i-1).Z,2),1); %***???X***
  usv = low_rank_appl([WZ(i-1).Z;a'],delta(2));
  WZ(i).W = autoencoder_br(i,WZ(i-1).Z,usv,param); %***???X***;
  [row,col] = size(WZ(i).W);
  WZ(i).W = randn(row,col);
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
fprintf('Iter    sec   loss   val_loss   train  test   norm_pro  \n');

%???????t???[??????
figure
%u= uicontrol('Style','slider','Position',[10 50 20 340],...
%    'Min',1,'Max',ftitr(1)*m/batch(2),'Value',1);
ax = gca;
F(ftitr(1)*m/batch(2)) = struct('cdata',[],'colormap',[]);
f_loop = 1;
%%----------------------


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
          [WZ(i+1).W,WZ(i).Z] = nmf_br('NS_F',WZ(i+1).Z,WZ(i+1).W, ...
                                       WZ(i).Z,lambda(1),lambda(2),ftitr(3),nsnmf(1),nsnmf(2),delta(2)); %***???X***
        end
        
        WZ(1).W = non_linear_lsq_br('XA',AI,AIusv,WZ(1).Z,WZ(1).W,lambda(1),nsnmf(1)); %***???X***
    
        %%-----????--------------------------------------------
%         WZ = compute_z_br(AI,WZ,L);
%         WZ = compute_w_br(AI,WZ,L,lambda,nsnmf(1));
        %%---------------------------------------------------------
        l = l + 1;
        resvec(l,:) = check_br(itr,toc-t0,A,A_test,Y,Y_test,WZ,L);
        
        plot(A,Y,'bo','MarkerFaceColor','b')
        %plot(A,Y,'bo')
        hold on
        %plot(AI,YI,'bo','MarkerFaceColor','b')
        output = compute_z_br(A_test,WZ,L);
        plot(A_test,Y_test,'k');
        plot(A_test,output(L).Z,'r');
        ylim([min(Y)*1.2,max(Y)*1.2]);
        xlabel('x')
        ylabel('y')
        %legend('train data','minibatch data','output','truth','Location','northwest')
        legend('train data','output','truth')
        %title(strcat('#epoch=',num2str(itr),' #minibatch=',num2str(kk)));
        %legend('train data','truth','output')
        grid on
        drawnow;
        %u.Value = f_loop;
        F(f_loop) = getframe(gcf);
        f_loop = f_loop + 1;
        pause(0.5)
       
        hold off
    end
    
%     plot(A,Y,'bo')
%     hold on
%     certain = compute_z_br(AI,WZ,L);
%     plot(AI,certain(L).Z,'bo','MarkerFaceColor','b')
%     output = compute_z_br(A_test,WZ,L);
%     plot(A_test,output(L).Z,'r');
%     plot(A_test,Y_test,'k');
%     ylim([-1.2,1.2]);
%     legend('train data','certain-data','output','truth')
%     drawnow;
%     pause(0.2)
%     hold off
end


t3 = toc;
fprintf('---------------------------------------------------- \n');
fprintf('Time for iteration:   %6.2f [sec.] \n',t3-t1);
fprintf('Total time:           %6.2f [sec.] \n',t3-t0);
fprintf('Total time - check:   %6.2f [sec.] \n',t3-t0-tc);
fprintf('---------------------------------------------------- \n');

end
