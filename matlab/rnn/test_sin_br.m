clear all;
close all
seed = 0;
rng(seed, 'twister');



format shorte

%data_set  = 'mnist';
%data_set  = 'cifar';
%data_size = 'small';%???org:??????????????????
data_size = 'org';
%[n_train,X_train,Y_train,n_test,X_test,Y_test] = load_data(data_set,data_size);

%????????
%-----------------------------
sigma = 0.3; %?m?C?Y?????U
n_train = 100;
X_train = linspace(0,2,n_train);
%X_train = csvread('~/Documents/python/Keras/X_train.csv');
%X_train = X_train';
tmp1 = randn(1,n_train);
tmp1 = tmp1>0;
tmp2 = -1*(tmp1-1);
%Y_train = sin(X_train*2*pi)+0.05*randn(1,n_train);
%noise = sqrt(sigma).*randn(1,n_train);
noise = (rand(1,n_train)*2-1)*sigma;
%noise1 = sqrt(sigma).*randn(1,n_train);
%noise2 = sqrt(0.001).*randn(1,n_train);
Y_train = sin(X_train*2*pi)+noise;
%Y_train = sin(X_train*2*pi)+noise1.*tmp1+noise2.*tmp2;
%Y_train = csvread('~/Documents/python/Keras/Y_train.csv');
%Y_train = Y_train';

n_test = 1000;
xx = linspace(0,2,n_test);
X_test = xx;
Y_test = sin(X_test*2*pi);
%-----------------------------
csvwrite('sin_X_train.csv',X_train);
csvwrite('sin_Y_train.csv',Y_train);
csvwrite('sin_X_test.csv',X_test);
csvwrite('sin_Y_test.csv',Y_test);

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
param.hidden = [1000,500];
%param.hidden = [1500,1000, 500];    % size of hidden units
param.delta  = [1e-8, 1e-8];
%param.delta  = [0, 0];

% threshold for low-rank approx. of the input data X and others
%param.aeitr  = [ 5, 10, 10];
param.aeitr  = [ 5, 10, 10];

% # iters for AE          (outer, NMF, nonlinear LSQ)

%param.ftitr  = [100,  1,  1];   
param.ftitr  = [10,  1,  1];
% # iters for fine tuning (outer, semi-NMF, nonlinear semi-NMF)

%param.nsnmf  = [10, 1];        % # iters for LSQs in nonlinear semi-NMF
param.nsnmf  = [10, 1];

param.batch  = [50, 50];   % size of batch for AE and fine tuning
param.lambda = [1e-2,1e-2];
%param.lambda = [0,0];
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
%Af=0;
[WZ,resvec(:,:,seed+1),F] = myDeepNN_sin_br(X_train,Y_train,X_test,Y_test,param);

%save('./DATA/resvec_br_3layer_epoch100_alpha6.mat', 'param', 'resvec','WZ')
%save('./DATA/resvec_br_2layer_epoch300_alpha0.mat', 'param', 'resvec','WZ')