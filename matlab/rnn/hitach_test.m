clear all;
close all
seed = 0;
rng(seed, 'twister');
format shorte

data = csvread('test_data_20170623_1a.csv');
data = data(1:end-4,:);
X = data(:,1:3);
Y = data(:,4:7);
m = size(X,1);
n_test = round(m/10);
n_train = m-n_test;
data_id = randperm(m);

WZ = cell(10,1);
WZ_val = cell(10,1);
WZ_train = cell(10,1);
resvec = cell(10,1);
hidden = {[64];
          [64,64];
          [64,64,64];
          [64,64,64,64];
          [64,64,64,64,64];
          [64,64,64,64,64,64];
          [64,64,64,64,64,64,64];
          [64,64,64,64,64,64,64,64];
          [64,64,64,64,64,64,64,64,64];
          [64,64,64,64,64,64,64,64,64,64]};

for k = 6:6
for i = 1:1
    %n_train = round(size(X,1)*0.9);
    data_ids = (i-1)*n_test+1;
    data_ide = i*n_test;
    X_test = X(data_id(data_ids:data_ide),:)';
    Y_test = Y(data_id(data_ids:data_ide),:)';
    Y_min = zeros(1,4);
    Y_max = zeros(1,4);

    train_id = [data_id(1:data_ids-1),data_id(data_ide+1:end)];
    X_train = (X(train_id,:))';
    Y_train = Y(train_id,:)';

    for j = 1:4
        Y_min(j) = min(Y_train(j,:));
        Y_max(j) = max(Y_train(j,:)-Y_min(j));
        Y_train(j,:) = (Y_train(j,:)-Y_min(j))/Y_max(j);
        Y_test(j,:) = (Y_test(j,:)-Y_min(j))/Y_max(j);
    end



    %ŠÖ”‹ßŽ—
    %-----------------------------
    % sigma = 0.01; %ƒmƒCƒY‚Ì•ªŽU
    % n_train = 100;
    % %X_train = linspace(0,1,n_train);
    % X_train = csvread('~/Documents/python/Keras/X_train.csv');
    % X_train = X_train';
    % tmp1 = randn(1,n_train);
    % tmp1 = tmp1>0;
    % tmp2 = -1*(tmp1-1);
    % %Y_train = sin(X_train*2*pi)+0.05*randn(1,n_train);
    % noise1 = sqrt(sigma).*randn(1,n_train);
    % noise2 = sqrt(0.001).*randn(1,n_train);
    % %Y_train = sin(X_train*2*pi)+noise;
    % %Y_train = sin(X_train*2*pi)+noise1.*tmp1+noise2.*tmp2;
    % Y_train = csvread('~/Documents/python/Keras/Y_train.csv');
    % Y_train = Y_train';
    % 
    % n_test = 1000;
    % xx = linspace(0,1,n_test);
    % X_test = xx;
    % Y_test = sin(X_test*2*pi);
    %-----------------------------

    %•ª—Þ–â‘è
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

    %param.hidden = [64,64,64,64,64,64,64,64,64,64];
    param.hidden = hidden{k};
    %param.hidden = [256,256,256];
    %param.hidden = [500,250];
    %param.hidden = [1000,500];
    %param.hidden = [1500,1000, 500];    % size of hidden units
    param.delta  = [1e-4, 1e-4];
    %param.delta  = [0, 0];

    % threshold for low-rank approx. of the input data X and others
    %param.aeitr  = [ 1, 5, 5];
    param.aeitr  = [ 5, 10, 10];
    %param.aeitr = [1,5,5];

    % # iters for AE          (outer, NMF, nonlinear LSQ)

    %param.ftitr  = [100,  1,  1];   
    param.ftitr  = [10,  1,  1];
    % # iters for fine tuning (outer, semi-NMF, nonlinear semi-NMF)

    %param.nsnmf  = [10, 1];        % # iters for LSQs in nonlinear semi-NMF
    param.nsnmf  = [5, 1];

    param.batch  = [332*3, 332*3];   % size of batch for AE and fine tuning
    %param.batch = [332,332];
    param.lambda = [0,0];
    %param.lambda = [1e-5,1e-5];
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


    %rng(seed, 'twister');
    disp(param)
    [WZ{i,k},WZ_train{i,k},WZ_val{i,k},resvec{i,k}] = myDeepNN_br_h(X_train,Y_train,X_test,Y_test,param,Y_max,Y_min);

    %save('./DATA/resvec_br_3layer_epoch100_alpha6.mat', 'param', 'resvec','WZ')
    %save('./DATA/resvec_br_2layer_epoch300_alpha0.mat', 'param', 'resvec','WZ')
end
end