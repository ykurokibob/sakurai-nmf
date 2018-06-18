clear all;
close all
seed = 0;
rng(seed, 'twister');

format shorte

%data_set  = 'mnist';
%data_set  = 'cifar';
%data_size = 'small';%???org:??????????????????
%data_size = 'org';
%[n_train,X_train,Y_train,n_test,X_test,Y_test] = load_data(data_set,data_size);

%????????
%-----------------------------
sigma = 0.0001; %?m?C?Y?????U
%% ?f?[?^????


di=50;
cycle=100;
testR = -0.05 + (0.05+0.05)*rand(1,di);
dataR = -0.05 + (0.05+0.05)*rand(1,di*cycle);
test=sin(linspace(0,2*pi,di));
data=sin(linspace(2*pi,2*pi*cycle+2*pi,di*cycle));
testN=sin(linspace(0,2*pi,di))+testR;
dataN=sin(linspace(2*pi,2*pi*cycle+2*pi,di*cycle))+dataR;

X_train=data;
%Y_train=DATA(te_ids,:);
%X_test=zeros(floor(tt*0.8),1);
X_test=test;
Y_train=reshape([data(1,2:di*cycle),data(1,1)],cycle,50);
Y_test=[test(1,2:di),test(1,1)];
data=reshape(sin(linspace(2*pi,2*pi*cycle+2*pi,di*cycle)),cycle,50);


%X_test = X_test';
%Y_test = Y_test';
%X_train = X_train';
%Y_train = Y_train';
%%
n_train = 50;
n_test = 50;
%X_train = linspace(-2,1,n_train);
%X_train = csvread('~/Documents/python/Keras/X_train.csv');
%X_train = X_train';
%tmp1 = randn(1,n_train);
%tmp1 = tmp1>0;
%tmp2 = -1*(tmp1-1);
%Y_train = sin(X_train*2*pi)+0.05*randn(1,n_train);
%noise = sqrt(sigma).*randn(1,n_train);
%noise = sigma*rand(1,n_train);
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
% n_train = 10000;
% X_train = [rand(1,n_train)*2*pi;rand(1,n_train)*2-1];
% % plot(X_train(1,:),X_train(2,:),'o');
% % hold on
% % x = linspace(0,2*pi,1000);
% % plot(x,sin(x),'r');
% %
% Y_train = zeros(2,n_train);
% for i = 1:n_train
%     if X_train(2,i) > sin(X_train(1,i))
%         Y_train(1,i) = 1;
%     else
%         Y_train(2,i) = 1;
%     end
% end
%
% n_test = 2000;
% X_test = [rand(1,n_test)*2*pi;rand(1,n_test)*2-1];
% Y_test = zeros(1,n_test);
% for i = 1:n_test
%     if X_test(2,i) > sin(X_test(1,i))
%         Y_test(1,i) = 1;
%     else
%         Y_test(2,i) = 1;
%     end
% end
%------------------------------

% Defaul parameters
param.n_train = n_train;
param.n_test = n_test;
%

%param.hidden = [500];
%param.hidden = [10];
param.hidden = zeros(1,di+1);
%param.hidden = zeros(1,10+1);
%param.hidden = one(1,10+1)*50;
%param.hidden = [1000,500];
%param.hidden = [1500,1000, 500];    % size of hidden units
param.delta  = [1e-5, 1e-5];
%param.delta  = [0, 0];

% threshold for low-rank approx. of the input data X and others
%param.aeitr  = [ 5, 10, 10];
param.aeitr  = [ 1, 1, 1];

% # iters for AE          (outer, NMF, nonlinear LSQ)
param.ftitr  = [60,  1,  1];
%param.ftitr  = [2000,  1,  1];
%param.ftitr  = [20,  1,  1];
% # iters for fine tuning (outer, semi-NMF, nonlinear semi-NMF)

param.nsnmf  = [1, 1];        % # iters for LSQs in nonlinear semi-NMF
%param.nsnmf  = [25, 25];

%param.batch  = [n_train, n_test];   % size of batch for AE and fine tuning
param.batch  = [10, 10];   % size of batch for AE and fine tuning
param.lambda = [1e-3,1e-3];
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
[WZ,resvec] = myDeepNN_sin_rnn(X_train,Y_train,X_test,Y_test,param);

%function [Uin,Ure,V] = nmf_rnn(Type,A,Uin,Ure,V,alpha,beta,iter1,iter2,i
%save('./DATA/resvec_br_3layer_epoch100_alpha6.mat', 'param', 'resvec','WZ')
%save('./DATA/resvec_br_2layer_epoch300_alpha0.mat', 'param', 'resvec','WZ')