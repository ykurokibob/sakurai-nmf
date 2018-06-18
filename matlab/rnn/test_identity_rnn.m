clear all;
close all
seed = 0;
rng(seed, 'twister');

%format shorte
flag=2;
%data_set  = 'mnist';
%data_set  = 'cifar';
%data_size = 'small';%???org:??????????????????
%data_size = 'org';
%[n_train,X_train,Y_train,n_test,X_test,Y_test] = load_data(data_set,data_size);

%????????
%-----------------------------
sigma = 1; %?m?C?Y?????U
times=1;
%% ?f?[?^????

if flag==1
    di=7;
    tt=2^di;
    DATA=zeros(tt,di);
    for i=1:tt
        a=dec2bin(i-1,di)-48;
        DATA(i,:)=a;
    end
    ids=randperm(tt);
    n_train = floor(tt*0.8);
    n_test = floor(tt*0.2)+1;
    tr_ids=ids( 1:floor(tt*0.8) );
    te_ids=ids( floor(tt*0.8):tt );
    X_train=DATA(tr_ids,:);
    %Y_train=DATA(te_ids,:);
    %X_test=zeros(floor(tt*0.8),1);
    X_test=DATA(te_ids,:);
    Y_train=zeros(floor(tt*0.8),1);
    Y_test=zeros(floor(tt*0.2)+2,1);
    for i1=1:floor(tt*0.8)
        Y_train(i1,1)=sum(X_train(i1,:))*times;
    end
    for i2=1:floor(tt*0.2)+2
        Y_test(i2,1)=sum(X_test(i2,:))*times;
    end
    
    sumXlist=zeros(1,di+1);
    
    for i=1:di
        sumXlist(1,i+1)=sumXlist(1,i)+sum(X_train(:,i))*times;
    end
else if flag==2
        di=20;
        cycle=180;
        tt=di*cycle;
        ids=randperm(tt);
        
        tr_ids=ids( 1:floor(tt/6*5) );
        te_ids=ids( floor(tt/6*5)+1:tt );
        
        n_train = di*cycle/6*5;
        n_test=di*cycle/6*5;
        sincur = linspace(0,2*pi*cycle,di*cycle);
        tmp1 = randn(1,n_train);
        %tmp1 = tmp1>0;
        %tmp2 = -1*(tmp1-1);
        ylabel = (sin(sincur)+0.05*randn(1,di*cycle))';
        %plot(sincur,Y_train)
        %noise = sqrt(sigma).*randn(1,n_train);
        %noise = sigma*rand(1,n_train);
        
        
        Y_train=ylabel(di+1:di*cycle*5/6,1);
        Y_test=ylabel(di*cycle*5/6+di+1:di*cycle,1);
        X_train=zeros(size(Y_train,1),di);
        X_test=zeros(size(Y_test,1),di);
        for i=1:size(Y_train,1)
            X_train(i,:)=ylabel(i:di+i-1,1);
        end
        for j=1:size(Y_test,1)
            X_test(j,:)=ylabel((tt/6*5)+j:tt/6*5+di+j-1,1);
        end
        %     X_train=(5000,50)
        %     X_test=(1000,50)
        
        %noise = noise - sigma/2;
        %noise1 = sqrt(sigma).*randn(1,n_train);
        %noise2 = sqrt(0.001).*randn(1,n_train);
        %Y_train = 1./(1+exp(-X_train))+1.8*exp(-(X_train-1).^2)+1.2*exp(-(X_train+1).^2)+noise;
        %Y_train = sin(X_train*2*pi)+noise1.*tmp1+noise2.*tmp2;
        %Y_train = csvread('~/Documents/python/Keras/Y_train.csv');
        %Y_train = Y_train';
        %n_test = 1024;
        %xx = linspace(-2,1,n_test);_
        %X_test = xx;
        %Y_test = 1./(1+exp(-X_test))+1.8*exp(-(X_test-1).^2)+1.2*exp(-(X_test+1).^2);
        %-----------------------------
        
        %????????
        %------------------------------
        %     n_train = 4;
        % %     X_train = [rand(1,n_train)*2*pi;rand(1,n_train)*2-1];
        % %      plot(X_train(1,:),X_train(2,:),'o');
        %     % hold on
        %     % x = linspace(0,2*pi,50);
        %     % plot(x,sin(x),'r');
        %     %
        %     Y_train = zeros(2,n_train);
        %     for i = 1:n_train
        %         if X_train(2,i) > sin(X_train(1,i))
        %             Y_train(1,i) = 1;
        %         else
        %             Y_train(2,i) = 1;
        %         end
        %     end
        %
        %     n_test = 2000;
        %     X_test = [rand(1,n_test)*2*pi;rand(1,n_test)*2-1];
        %     Y_test = zeros(1,n_test);
        %     for i = 1:n_test
        %         if X_test(2,i) > sin(X_test(1,i))
        %             Y_test(1,i) = 1;
        %         else
        %             Y_test(2,i) = 1;
        %         end
    end
    
end
%------------------------------

% Defaul parameters
param.n_train = n_train;
param.n_test = n_test;
%

%param.hidden = [500];
%param.hidden = [10];
param.hidden = zeros(1,di+1);
%param.bias = zeros(2,di+1);
%param.hidden = zeros(1,10+1);
%param.hidden = one(1,10+1)*50;
%param.hidden = [1000,500];
%param.hidden = [1500,1000, 500];    % size of hidden units
param.delta  = [1e-14, 1e-14];
%param.delta  = [0, 0];

% threshold for low-rank approx. of the input data X and others
%param.aeitr  = [ 5, 10, 10];
param.aeitr  = [ 5, 10, 10];

% # iters for AE          (outer, NMF, nonlinear LSQ)
param.ftitr  = [400,  1,  1];
%param.ftitr  = [2000,  1,  1];
%param.ftitr  = [20,  1,  1];
% # iters for fine tuning (outer, semi-NMF, nonlinear semi-NMF)

param.nsnmf  = [1, 1];        % # iters for LSQs in nonlinear semi-NMF
%param.nsnmf  = [25, 25];

%param.batch  = [n_train, n_test];   % size of batch for AE and fine tuning
param.batch  = [500, 500];   % size of batch for AE and fine tuning
param.lambda = [1e-5,1e-5];
%param.lambda = [1,1];
%param.lambda = 1e-2;
%
% Set parameters

% For CIFAR10
%param.delta  = [5e-3, 1e-14];
% threshold for SVD of the input data X and others
%param.aeitr  = [20, 10, 10];
% # iters for AE          (outer, NMF, nonlinear LSQ)
%param.nsnmf  = [25, 25];        % # iters for LSQs in nonlinear semi-NMF
%
% Write parameters

rng(seed, 'twister');
disp(param)
%[WZ,resvec(:,:,seed+1),F] = myDeepNN_identity(X_train,Y_train,X_test,Y_test,param);
%[WZ,resvec(:,:,seed+1),F] = myDeepNN_identity_br(X_train,Y_train,X_test,Y_test,param);
%[WZ,resvec(:,:,seed+1),F] = myDeepNN_identity_br(X_train,X_test,Y_train,Y_test,param);

if flag==1
    kei=0.1;
    add=0;
    param.in_dims=1;
    param.hid_dims=1;
    param.out_dims=1;
    param.bias_dims=1;
else if flag==2
        kei=0.01;
        add=0;
        param.in_dims=1;
        param.hid_dims=4;
        param.out_dims=1;
        param.bias_dims=1;
        
    end
end
%-5 + (5+5)*rand(10,1)
WZ(1).Wre=-1+2*rand(param.hid_dims,param.hid_dims)*kei+add;
WZ(1).Win=-1+2*rand(param.hid_dims,param.in_dims)*kei+add;
WZ(1).Wout=-1+2*rand(param.out_dims,param.hid_dims)*kei+add;
WZ(1).Wbias=ones(param.bias_dims,param.out_dims);
param.hidden = zeros(param.hid_dims,di+1);

%    WZ(1).Wre=ones(param.hid_dims,param.hid_dims);
%    WZ(1).Win=ones(param.hid_dims,param.in_dims)*0.52860844;
%    WZ(1).Wout=ones(param.out_dims,param.hid_dims)*1.10927045;

% WZ(1).Wre=1;
% WZ(1).Win=1;%rand(param.hid_dims,param.in_dims)*(1);
% WZ(1).Wout=1;%rand(param.out_dims,param.hid_dims)*(1);

[WZ,resvec,CC] = myDeepNN_identity_rnn(X_train,Y_train,X_test,Y_test,param,tr_ids,te_ids,WZ);



%function [Uin,Ure,V] = nmf_rnn(Type,A,Uin,Ure,V,alpha,beta,iter1,iter2,i
%save('./DATA/0606_rnn_1L_ep500_resvec.mat', 'param', 'resvec','WZ')
%save('./DATA/resvec_br_2layer_epoch300_alpha0.mat', 'param', 'resvec','WZ')

%%
if flag==2
    nextdata=X_test(di*2,:);
    inidata=nextdata;
    L = size(param.hidden,2);
    for i=1:di
        add=compute_rnn(nextdata,WZ,L,param.hid_dims);
        add(L).Z
        inidata=[inidata(:);add(L).Z]';
        nextdata=inidata(i+1:end);
    end
    figure(2)
    plot(inidata)
    
    nextdata=X_train(di*2,:);
    inidata=nextdata;
    L = size(param.hidden,2);
    for i=1:di
        add=compute_rnn(nextdata,WZ,L,param.hid_dims);
        add(L).Z
        inidata=[inidata(:);add(L).Z]';
        nextdata=inidata(i+1:end);
    end
    figure(2)
    plot(inidata)
end
