clear all;


format shorte

%data_set  = 'mnist';
data_set  = 'cifar';
data_size = 'small';%???org:??????????????????
%data_size = 'org';
[n_train,X_train,Y_train,n_test,X_test,Y_test] = load_data(data_set,data_size);

% Defaul parameters
param.n_train = n_train;
param.n_test = n_test;
%

%param.hidden = [1000, 500];
param.hidden = [1500,1000, 500];    % size of hidden units
param.delta  = [5e-3, 1e-14];
% threshold for low-rank approx. of the input data X and others
param.aeitr  = [ 20, 25, 25];
% # iters for AE          (outer, NMF, nonlinear LSQ)

param.ftitr  = [20,  1,  1];   
% # iters for fine tuning (outer, semi-NMF, nonlinear semi-NMF)

param.nsnmf  = [10, 1];        % # iters for LSQs in nonlinear semi-NMF
param.batch  = [5000, 5000];   % size of batch for AE and fine tuning
param.lambda = [1e-5,1e-7];
%
% Set parameters

%For CIFAR10
param.delta  = [5e-3, 1e-14];
%threshold for SVD of the input data X and others
param.aeitr  = [20, 10, 10];
%# iters for AE          (outer, NMF, nonlinear LSQ)
param.nsnmf  = [25, 25];        % # iters for LSQs in nonlinear semi-NMF
%
% Write parameters

for seed = 0:0
rng(seed, 'twister');
disp(param)
[WZ,resvec(:,:,seed+1)] = myDeepNN_br(X_train,Y_train,X_test,Y_test,param);

end
%save('./DATA/resvec_br_3layer_epoch100_alpha6.mat', 'param', 'resvec','WZ')
save('./DATA/resvec_br_3layer_epoch100_ab57_cifar.mat', 'param', 'resvec','WZ')