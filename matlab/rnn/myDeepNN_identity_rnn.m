%function [WZ,resvec,F] = myDeepNN_identity_br(A,Y,A_test,Y_test,param)
%function [WZ,resvec,F] = myDeepNN_identity_br(A,Y,A_test,Y_test,param,sumXlist,tr_ids,te_ids)
%function [WZ,resvec,CC] = myDeepNN_identity_rnn(A,Y,A_test,Y_test,param,sumXlist,tr_ids,te_ids,WZ)
function [WZ,resvec,CC] = myDeepNN_identity_rnn(A,Y,A_test,Y_test,param,tr_ids,te_ids,WZ)
tic;
t0 = toc;
tc = 0;
fprintf('-- AutoEncoder ------------------------------------- \n');
fprintf('Iter    sec   loss   val_loss   train  test  norm_pro  \n');
minloss=200;
stopper=0;
% set parameter
[m,n] = size(A);
L = size(param.hidden,2);
%L = param.ftitr(1);

delta = param.delta;
aeitr = param.aeitr;
ftitr = param.ftitr;
nsnmf = param.nsnmf;
batch = param.batch;
lambda = param.lambda;

% AutoEncoder
%if Af==1
%a = ones(size(A,2),1);
a = A;

Ausv = low_rank_appl(A,delta(1));

%%
%Ab = low_rank_appl(A(1,:),delta(1)); %***???X***
% Ausv.U=zeros(1,size(A,1));
% Ausv.S=zeros(1,size(A,1));
% Ausv.V=zeros(1,size(A,1));
%
% for j=1:size(A,1)
%     Ab = low_rank_appl(A(j,1),delta(1)); %***???X***
%     Ausv.U(j,1)=Ab.U;
%     Ausv.S(j,1)=Ab.S;
%     Ausv.V(j,1)=Ab.V;
%
% end
%%
t1 = toc;
fprintf('---------------------------------------------------- \n');
fprintf('Time for SVD of A   : %6.2f [sec.] \n',t1-t0);
fprintf('---------------------------------------------------- \n');
%%
%for j = 1:10
% in_dims=1;
% hid_dims=2;
% out_dims=1;
% WZ(1).Wre=rand(in_dims,hid_dims)*(1e-05);
% WZ(1).Win=rand(hid_dims,hid_dims)*(1e-05);
% WZ(1).Wout=rand(hid_dims,out_dims)*(1e-05);
%ones(1,3)IN*ones(3,4)hid*ones(4,1)OUT
%1-3-4-1
%end
%WZ(1).Wre=10;
%WZ(1).Win=1;
%%
WRe1=WZ(1).Wre;
WIn1=WZ(1).Win;
WOut1=WZ(1).Wout;


%for j = 1:size(A,1)
%    WZ(1).Z=appl_f([WZ(1).Wre,WZ(1).Win]*[0;a(1,j)]);
%end
%id = randperm(m);
%id = randperm(m);
%m
%id = randperm(m)
%AI = A(id(1:batch(1)),:);
%YI = Y(id(1:batch(1)),:);

AI = A;
YI = A_test;
YI_size=size(YI);
AIusv.U = Ausv.U;
AIusv.S = Ausv.S;
AIusv.V = Ausv.V;
 useAE=true;

l = 1;
tc0 = toc;
%resvec(l,:) = check_br(0,toc-t0,A,A_test,Y,Y_test,WZ,L);
tc = tc + (toc-tc0);

t2 = toc;
fprintf('---------------------------------------------------- \n');
fprintf('Time for AutoEncoder: %6.2f [sec.] \n',t2-t1);
fprintf('-- Iteration --------------------------------------- \n');
fprintf('Iter    sec   loss   val_loss   train  test   norm_pro  \n');

%???????t???[??????
%figure
%u= uicontrol('Style','slider','Position',[10 50 20 340],...
%    'Min',1,'Max',ftitr(1)*m/batch(2),'Value',1);
%ax = gca;
%F(ftitr(1)*m/batch(2)) = struct('cdata',[],'colormap',[]);
f_loop = 1;
%%----------------------

% WZW_list.Wre;
% WZW_list.Win;
% WZW_list.Wout;

%% Alternative minimization iteration (Back propagation)
for itr = 1:ftitr(1)
    %    param.lambda
    %for itr = 1:2
    %numitr=itr
    %for j = 1:size(A,2)
    %id = randperm(m);
    %sizea = size(Ausv.V(1,j))
    %AsuvV=Ausv.V;
    %        for kk = 1:m/batch(2)
    %        ids = (kk-1)*batch(2)+1;
    %        ide = kk*batch(2);
    %         AI = A(:,id(ids:ide));
    %        YI = Y(:,id(ids:ide));
    %       AI = A(:,id(1:size(A,1)));
    %       YI = Y(:,id(1:size(A,1)));
    %AIusv.V = Ausv.V(1,j);
    
    WZ = compute_rnn(AI,WZ,L,param.hid_dims); %***???X***
    %     Err=zeros(1,2);
    %     for i=1:2
    %         Err(1,i)=sum(WZ(i).Z)-sumXlist(i+1);
    %     end
    %     Err=Err;
    %WZ(1).Wre
    checker = WZ(L-1).Z;
    [WZ(1).Wout,WZ(L-1).Z] = nmf_rnn('S_F',Y,WZ(1).Wout,WZ(L-1).Z,lambda(1),lambda(2),ftitr(2),0,0,delta(2)); %***???X***
    %[Wall,Zall] = nmf_rnn('S_F',Y,Wall,Zall,lambda(1),lambda(2),ftitr(2),nsnmf(1),nsnmf(2),delta(2)); %***???X***
    
    %function [Uin,Ure,V] =           nmf_rnn(Type,A  ,Uin      ,Ure      ,V         ,alpha,beta,iter1,iter2,iter3,delta,j)
    Wall=[WZ(1).Wre,WZ(1).Win];
    %     Wall(1);
    %     size(WZ(L-1).Z);
    %     size(A);
    %     size((A(:,L-1))');
    %     %Zall=[WZ(L-1).Z];
    Zall=[WZ(L-1).Z;(A(:,L-1))'];
    %Zall=[,WZ(L-1).Z];
    %     Zallsize=size(Zall);
    
    %[WZ(L).Win,WZ(L).Wre,WZ(L-1).Z] = nmf_rnn('S_F',YI,WZ(L).Win,WZ(L).Wre,WZ(L-1).Z,lambda(1),lambda(2),ftitr(2),0,0,delta(2)); %***???X***
    %[Wall,Zall] = nmf_rnn('S_F',Y,Wall,Zall,lambda(1),lambda(2),ftitr(2),1,1,delta(2)); %***???X***
    %[Wall,Zall] = nmf_rnn('S_F',Y,Wall,Zall,lambda(1),lambda(2),ftitr(2),nsnmf(1),nsnmf(2),delta(2)); %***???X***
    %[WZ(L).Z,Zall] = nmf_rnn('S_F',Y,WZ(L).Z,Zall,lambda(1),lambda(2),ftitr(2),nsnmf(1),nsnmf(2),delta(2)); %***???X***
    %[WZ(L).Win,WZ(L-1).Z] = nmf_rnn('S_F',YI,WZ(L).Win,WZ(L-1).Z, ...
    %lambda(1),lambda(2),ftitr(2),0,0,delta(2)); %***???X***
    %Zall=[Zall;(A(:,L-1))'];
    %Wall=WZ(1).Wout
    for i = L-2:-1:1
        %Zallsize=size(Zall)
        %Zall=[Zall(2:end,:);(A(:,i))'];
        Zall=[Zall(1:end-1,:);(A(:,i))'];
        %Zall=[Zall(1:end-1,:);(A(:,i))'];
        %Zallsize=size(Zall)
        %Zall=[(A(:,i))';WZ(i).Z];
        %nmf???????????????????????????????????????????
        %[U,V] = nmf_br(Type,A,U,V,alpha,beta,iter1,iter2,iter3,delta)
        %[Wall,WZ(i).Z] = nmf_rnn('NS_F',WZ(i+1).Z,Wall, ...
        % WZ(i).Z,lambda(1),lambda(2),ftitr(3),nsnmf(1),nsnmf(2),delta(2)); %***???X***
        i;
        [Wall,Zall] = nmf_rnn('NS_F',WZ(i+1).Z,Wall,Zall,lambda(1),lambda(2),ftitr(3),nsnmf(1),nsnmf(2),delta(2));
        %[Wall,Zall(1:end-1,:)] = nmf_rnn('NS_F',WZ(i+1).Z,Wall,Zall,lambda(1),lambda(2),ftitr(3),nsnmf(1),nsnmf(2),delta(2));
        %***???X***
        %[WZ(i+1).Win,WZ(i+1).Wre,WZ(i).Z] = nmf_rnn('NS_F',WZ(i+1).Z,WZ(i+1).Win,WZ(i+1).Wre, ...
        % WZ(i).Z,lambda(1),lambda(2),ftitr(3),nsnmf(1),nsnmf(2),delta(2)); %***???X***
        %[WZ(i+1).Win,WZ(i).Z] = nmf_rnn('NS_F',WZ(i+1).Z,WZ(i+1).Win, ...
        %WZ(i).Z,lambda(1),lambda(2),ftitr(3),nsnmf(1),nsnmf(2),delta(2)); %***???X***
        Wall9to2=Wall;
    end
    Wallsize=size(Wall);
    WZ(1).Wre=Wall(:,1:param.hid_dims);
    %    WZ(1).Wre=Wall(:,1:4);
    WZ(1).Win=Wall(:,end);
    %aa
    %sizeaisuvv=size(AIusv.V);
    %Zallsize=size(Zall);
    %Zall;
    A1=Zall(end,:);
    %zero1=zeros(size(A1))
    A1usv = low_rank_appl(A1,delta(1));
    %zero1usv = low_rank_appl(zero1,delta(1));
    
    
    %     elseif strcmp(Type,'NS_F')
    %   % Nonlinear Semi-NMF: min_U,V ||A - f(U V)|| s.t. V >= 0
    %     for i = 1:iter1
    %         usv = low_rank_appl(V,0);
    %         %function X = non_linear_lsq_rnn(AXorXA,A,Asvd,B,X,lambda,itermax)
    %         %sizeA=size(A)
    %         %A(2,:)
    %         Uall = non_linear_lsq_rnn('XA',V,usv,A,Uall,alpha,iter2);
    %         %Ure = non_linear_lsq_rnn('XA',V,usv,A,Ure,alpha,iter2);
    %         usvall = low_rank_appl(Uall,0);
    %         %usvre = low_rank_appl(Ure,0);
    %         V = non_linear_lsq_rnn('AX',Uall,usvall,A,V,beta,iter3);%+ non_linear_lsq_rnn('AX',Ure,usvre,A,V,beta,iter3);
    %
    %     end
    
    
    %A1usv=A1
    %sizeWre=size(WZ(1).Wre);
    WZ(1).Win = non_linear_lsq_rnn('XA',A1,A1usv,(WZ(1).Z),WZ(1).Win,lambda(2),nsnmf(1)); %***???X**
    %WZ(1).Wre = non_linear_lsq_rnn('XA',zero1,zero1usv,(WZ(1).Z),WZ(1).Wre,lambda(2),nsnmf(1)); %***???X***
    %sizeWre=size(WZ(1).Wre);?
    WRe=WZ(1).Wre;
    WIn=WZ(1).Win;
    
    %function X = non_linear_lsq_rnn(AXorXA,A,Asvd,B,X,lambda,itermax)
    %WZ(1).W = non_linear_lsq_br('XA',AI,AIusv,WZ(1).Z,WZ(1).W,lambda(1),nsnmf(1)); %***???X***
    
    %WZ(k).Wre = non_linear_lsq_rnn('XA',AI(k,j),AIusv,WZ(k).Z,WZ(k).Wre,lambda(1),nsnmf(1)); %***???X***
    
    %      WZW_list(itr,1)=WRe;
    %      WZW_list(itr,2)=WIn;
    %      WZW_list(itr,3)=WZ(1).Wout;
    
    %%
    %end
    %%-----????--------------------------------------------
    %         WZ = compute_z_br(AI,WZ,L);
    %         WZ = compute_w_br(AI,WZ,L,lambda,nsnmf(1));
    %%---------------------------------------------------------
    
    l = l + 1;
    CC(l,1) = costcheck(WZ,Y,A,param);
    CC(l,2) = costcheck(WZ,Y_test,A_test,param);
    if l==2
        minloss=CC(l,1);
    end
    resvec(l,:) = check_rnn(itr,toc-t0,A,A_test,Y,Y_test,WZ,L,param);
    newcost=CC(l,1);
%    newcost > CC(l-1,1)
    if newcost > minloss;
        stopper=stopper+1;
        
    else
        minloss=newcost;
        stopper=0;
    end
        
     if stopper>800
         break
     end
    %      if minLoss<CC(l,1)
    %          minLoss=CC(l,1);
    %          minW=WZW_list(itr,:);
    %
    %      end
    %     figure(2)
    %     plot(A,Y,'bo','MarkerFaceColor','b');
    %     %plot(A,Y,'bo')
    %     hold on
    %     %plot(AI,YI,'bo','MarkerFaceColor','b')
    %     output = compute_z_rnn(A_test,WZ,L);
    %     plot(A_test,Y_test,'k');
    %     plot(A_test,output(L).Z,'r');
    %     ylim([min(Y)*0,max(Y)*1.2]);
    %     xlabel('x');
    %     ylabel('y');
    %     %legend('train data','minibatch data','output','truth','Location','northwest')
    %     legend('train data','output','truth');
    %     %title(strcat('#epoch=',num2str(itr),' #minibatch=',num2str(kk)));
    %     %legend('train data','truth','output')
    %     grid on
    %     drawnow;
    %     %u.Value = f_loop;
    %     F(f_loop) = getframe(gcf);
    %     f_loop = f_loop + 1;
    %     %pause(0.5)
    %
    %     hold off
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
%end


%figure
W1Flag = false;
if W1Flag > 0
    %subplot(1,2,1)       % add first plot in 2 x 1 grid
    %plot(x,y1)
    %title('Subplot 1')
    figure(1)
    semilogy(WZW_list);
    ylabel('weights value')
    xlabel('iteration')
    title('Wgraph')
    legend('Wre','Win','Wout','Location','northeast')
    
    subplot(1,2,2)
end% add second plot in 2 x 1 grid
%plot(x,y2,'+')       % plot using + markers
%title('Subplot 2')
semilogy(CC);
str_WRe=sprintf('%1.3e',WRe1);
str_WIn=sprintf('%1.3e',WIn1);
str_WOut=sprintf('%1.3e',WOut1);
Loss_MIN=sprintf('%1.7f',min(nonzeros(CC)));
%title(strcat('loss graph weight= ',str_WRe,',',str_WIn,',',str_WOut));
time=datetime;
title(strcat(datestr(time),', min loss = ',Loss_MIN));%    fprintf('%3d  %6.2f  %4.6f  %4.6f\n',itr,tt,norm_loss,norm_val_loss);
%WRe1=WZ(1).Wre;
%WIn1=WZ(1).Win;
%WOut1=WZ(1).Wout;

ylabel('loss value')
xlabel('iteration')


t3 = toc;
fprintf('---------------------------------------------------- \n');
fprintf('Time for iteration:   %6.2f [sec.] \n',t3-t1);
fprintf('Total time:           %6.2f [sec.] \n',t3-t0);
fprintf('Total time - check:   %6.2f [sec.] \n',t3-t0-tc);
fprintf('---------------------------------------------------- \n');
%WZW_list
%WZ(2).Wre=minW(1) ;      %minW=WZW_list(itr,:);
%WZ(2).Win=minW(2);
%WZ(2).Wout=minW(3);
end
