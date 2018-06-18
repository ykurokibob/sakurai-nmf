%function [WZ,resvec,F] = myDeepNN_identity_br(A,Y,A_test,Y_test,param)
%function [WZ,resvec,F] = myDeepNN_identity_br(A,Y,A_test,Y_test,param,sumXlist,tr_ids,te_ids)
function [CC,WZ,resvec] = myDeepNN_identity_br(A,Y,A_test,Y_test,param,sumXlist,tr_ids,te_ids)
tic;
t0 = toc;
tc = 0;
fprintf('-- AutoEncoder ------------------------------------- \n');
fprintf('Iter    sec   loss   val_loss   train  test  norm_pro  \n');

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

t1 = toc;
fprintf('---------------------------------------------------- \n');
fprintf('Time for SVD of A   : %6.2f [sec.] \n',t1-t0);
fprintf('---------------------------------------------------- \n');
%for j = 1:10
 WZ(1).Wre=1;
 WZ(1).Win=1;
%end
%WZ(1).Wre=4;
%WZ(1).Win=1;
WRe1=WZ(1).Wre
WIn1=WZ(1).Win


%for j = 1:size(A,1)
%    WZ(1).Z=appl_f([WZ(1).Wre,WZ(1).Win]*[0;a(1,j)]);
%end
%id = randperm(m);
%id = randperm(m);
size(A)
AI = A;
%YI = Y;
%YI_size=size(YI);
AIusv.U = Ausv.U;
AIusv.S = Ausv.S;
AIusv.V = Ausv.V;

% for j = 1:size(A,1)
% WZ(1).W = autoencoder_br(1,AI,AIusv,param); %***???X***
% [row,col] = size(WZ(1).W);
% WZ(1).W = randn(row,col);
% a = ones(size(AI,2),1);
% WZ(1).Z = appl_f(WZ(1).W*AI+a(j)'); %***???X***
%
% for i = 2:L
%   a = zeros(size(WZ(i-1).Z,2),1); %***???X***
%   usv = low_rank_appl([WZ(i-1).Z;a'],delta(2));
%   WZ(i).W = autoencoder_br(i,WZ(i-1).Z,usv,param); %***???X***;
%   [row,col] = size(WZ(i).W);
%   WZ(i).W = randn(row,col);
%   WZ(i).Z = appl_f(WZ(i).W*[WZ(i-1).Z;a']); %***???X***
% end

%a = ones(size(WZ(L-1).Z,2),1); %***???X***

%usv = low_rank_appl([WZ(L-1).Z;a'],delta(2)); %***???X***
%WZ(L).W = YI * usv.V / usv.S * usv.U'; %***???X***
%WZ(L).W = YI / WZ(L-1).Z;

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
ax = gca;
%F(ftitr(1)*m/batch(2)) = struct('cdata',[],'colormap',[]);
f_loop = 1;
%%----------------------
WZW_list=zeros(ftitr(1));

%% Alternative minimization iteration (Back propagation)
for itr = 1:ftitr(1)
     %param.lambda
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
    
    WZ = compute_z_rnn(AI,WZ,L); %***???X***
    Err=zeros(1,2);
    for i=1:2
        Err(1,i)=sum(WZ(i).Z)-sumXlist(i+1);
    end
    Err=Err;
    %       [WZ(L).W,WZ(L-1).Z] = nmf_br('S_F',YI,WZ(L).W,WZ(L-1).Z, ...
    %                                    lambda(1),lambda(2),ftitr(2),0,0,delta(2)); %***???X***
    %function [Uin,Ure,V] =           nmf_rnn(Type,A  ,Uin      ,Ure      ,V         ,alpha,beta,iter1,iter2,iter3,delta,j)
    Wall=[WZ(1).Wre,WZ(1).Win];
    Wall;
    size(WZ(L-1).Z);
    %Zall=[WZ(L-1).Z;(A(:,L-1))'];
    Zall=[WZ(L-1).Z;(A(:,L))'];
    %Zall=[,WZ(L-1).Z];
    size(Zall);
    
    %[WZ(L).Win,WZ(L).Wre,WZ(L-1).Z] = nmf_rnn('S_F',YI,WZ(L).Win,WZ(L).Wre,WZ(L-1).Z,lambda(1),lambda(2),ftitr(2),0,0,delta(2)); %***???X***
    %[Wall,Zall] = nmf_rnn('S_F',Y,Wall,Zall,lambda(1),lambda(2),ftitr(2),1,1,delta(2)); %***???X***
    %    [Wall,Zall] = nmf_rnn('S_F',Y,Wall,Zall,lambda(1),lambda(2),ftitr(2),nsnmf(1),nsnmf(2),delta(2)); %***???X***
    [Wall,Zall] = nmf_rnn('S_F',Y,Wall,Zall,lambda(1),lambda(2),ftitr(2),nsnmf(1),nsnmf(2),delta(2)); %***???X***
    %[WZ(L).Win,WZ(L-1).Z] = nmf_rnn('S_F',YI,WZ(L).Win,WZ(L-1).Z, ...
    %lambda(1),lambda(2),ftitr(2),0,0,delta(2)); %***???X***
    Wall10=Wall;
    for i = L-1:-1:1
        Zall=[Zall(2,:);(A(:,i))'];
        %Zall=[(A(:,i))';WZ(i).Z];
        %nmf???????????????????????????????????????????
        %[U,V] = nmf_br(Type,A,U,V,alpha,beta,iter1,iter2,iter3,delta)
        %[Wall,WZ(i).Z] = nmf_rnn('NS_F',WZ(i+1).Z,Wall, ...
        % WZ(i).Z,lambda(1),lambda(2),ftitr(3),nsnmf(1),nsnmf(2),delta(2)); %***???X***
        i;
        [Wall,Zall] = nmf_rnn('NS_F',WZ(i+1).Z,Wall,Zall,lambda(1),lambda(2),ftitr(3),nsnmf(1),nsnmf(2),delta(2));
        %***???X***
        %[WZ(i+1).Win,WZ(i+1).Wre,WZ(i).Z] = nmf_rnn('NS_F',WZ(i+1).Z,WZ(i+1).Win,WZ(i+1).Wre, ...
        % WZ(i).Z,lambda(1),lambda(2),ftitr(3),nsnmf(1),nsnmf(2),delta(2)); %***???X***
        %[WZ(i+1).Win,WZ(i).Z] = nmf_rnn('NS_F',WZ(i+1).Z,WZ(i+1).Win, ...
        %WZ(i).Z,lambda(1),lambda(2),ftitr(3),nsnmf(1),nsnmf(2),delta(2)); %***???X***
        Wall9to2=Wall;
    end
    WZ(1).Wre=Wall(1);
    WZ(1).Win=Wall(2);
    
    sizeaisuvv=size(AIusv.V);
    %for k=1:size(A,1)
    Zallsize=size(Zall);
    Zall;
    A1=Zall(2,:);
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
    WZ(1).Win = non_linear_lsq_rnn('XA',A1,A1usv,(WZ(1).Z),WZ(1).Win,lambda(2),nsnmf(1)); %***???X***
    %WZ(1).Wre = non_linear_lsq_rnn('XA',zero1,zero1usv,(WZ(1).Z),WZ(1).Wre,lambda(2),nsnmf(1)); %***???X***
    %sizeWre=size(WZ(1).Wre);?
    WRe=WZ(1).Wre;
    WIn=WZ(1).Win;
    
    %function X = non_linear_lsq_rnn(AXorXA,A,Asvd,B,X,lambda,itermax)
    %WZ(1).W = non_linear_lsq_br('XA',AI,AIusv,WZ(1).Z,WZ(1).W,lambda(1),nsnmf(1)); %***???X***
    
    %WZ(k).Wre = non_linear_lsq_rnn('XA',AI(k,j),AIusv,WZ(k).Z,WZ(k).Wre,lambda(1),nsnmf(1)); %***???X***
    WZW_list(itr,1)=WRe;
    WZW_list(itr,2)=WIn;
    %%
    %end
    %%-----????--------------------------------------------
    %         WZ = compute_z_br(AI,WZ,L);
    %         WZ = compute_w_br(AI,WZ,L,lambda,nsnmf(1));
    %%---------------------------------------------------------
    
    l = l + 1;
    resvec(l,:) = check_rnn(itr,toc-t0,A,A_test,Y,Y_test,WZ,L);
    CC(l,1) = costcheck(WZ(1).Wre,WZ(1).Win);

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
%figure(1)
%plot(WZW_list)

figure
subplot(1,2,1)       % add first plot in 2 x 1 grid
plot(WZW_list);
ylabel('weights value')
xlabel('iteration')
title('Wgraph')
legend('Wre','Win','Location','northeast')

%figure(2)
subplot(1,2,2)       % add second plot in 2 x 1 grid

semilogy(CC(1:size(CC,1)-1))

str_WRe=sprintf('%1.3e',WRe1);
str_WIn=sprintf('%1.3e',WIn1);
%str_WOut=sprintf('%1.3e',WOut1);
title(strcat('loss graph weight= ',str_WRe,',',str_WIn));
%    fprintf('%3d  %6.2f  %4.6f  %4.6f\n',itr,tt,norm_loss,norm_val_loss);
%WRe1=WZ(1).Wre;
%WIn1=WZ(1).Win;
%WOut1=WZ(1).Wout;

ylabel('loss value')
xlabel('iteration')

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



t3 = toc;
fprintf('---------------------------------------------------- \n');
fprintf('Time for iteration:   %6.2f [sec.] \n',t3-t1);
fprintf('Total time:           %6.2f [sec.] \n',t3-t0);
fprintf('Total time - check:   %6.2f [sec.] \n',t3-t0-tc);
fprintf('---------------------------------------------------- \n');

end
