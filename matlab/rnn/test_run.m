clear all;
close all;
seed = 0;
rng(seed, 'twister')

%ŠÖ”‹ßŽ—
%-----------------------------
sigma = 0.01; %ƒmƒCƒY‚Ì•ªŽU
n_train = 150;
%X_train = linspace(0,1,n_train);
X_train = csvread('~/Documents/python/Keras/X_train.csv');
X_train = X_train';
tmp1 = randn(1,n_train);
tmp1 = tmp1>0;
tmp2 = -1*(tmp1-1);
%Y_train = sin(X_train*2*pi)+0.05*randn(1,n_train);
noise1 = sqrt(sigma).*randn(1,n_train);
noise2 = sqrt(0.001).*randn(1,n_train);
%Y_train = sin(X_train*2*pi)+noise;
%Y_train = sin(X_train*2*pi)+noise1.*tmp1+noise2.*tmp2;
Y_train = csvread('~/Documents/python/Keras/Y_train.csv');
Y_train = Y_train';

n_test = 1000;
xx = linspace(0,1,n_test);
X_test = xx;
Y_test = sin(X_test*2*pi);
%-----------------------------



% Defaul parameters
param.n_train = n_train;
param.n_test = n_test;
%
param.hidden = [500];    % size of hidden units 2layer
%param.hidden = [1500,1000,500];    % size of hidden units 3layer
param.delta  = [1e-8, 1e-8];  % threshold for low-rank approx. of the input data X and others
param.aeitr  = [ 1, 1, 1];   % # iters for AE          (outer, NMF, nonlinear LSQ)
param.ftitr  = [100,  1,  1];   % # iters for fine tuning (outer, semi-NMF, nonlinear semi-NMF)
param.nsnmf  = [1, 1];        % # iters for LSQs in nonlinear semi-NMF
param.batch  = [50, 50];   % size of batch for AE and fine tuning
%param.lambda = 0.0;
param.lambda = [0,0];
%
% Set parameters

% For CIFAR10
%param.delta  = [5e-3, 1e-14];  % threshold for SVD of the input data X and others
%param.aeitr  = [20, 10, 10];   % # iters for AE          (outer, NMF, nonlinear LSQ)
%param.nsnmf  = [25, 25];        % # iters for LSQs in nonlinear semi-NMF
%
% Write parameters
disp(param)

[WZ,resvec(:,:,seed+1)] = myDeepNN_sin(X_train,Y_train,X_test,Y_test,param);


%save('./DATA/resvec_2layer_epoch50.mat', 'param', 'resvec','WZ')
%save('./DATA/resvec_3layer_epoch100.mat', 'param', 'resvec','WZ')
%save('./DATA/resvec_2layer_epoch500.mat', 'param', 'resvec','WZ')
