clear all;
close all
seed = 0;
rng(seed, 'twister');



format shorte

data_set  = 'mnist';
%data_set  = 'cifar';
%data_size = 'small';%???org:??????????????????
data_size = 'org';
[n_train,X_train,Y_train,n_test,X_test,Y_test] = load_data(data_set,data_size);

%?????@?????W
%-----------------------------
% sigma = 0.0;
% n_train = 1000;
% n2_train = n_train/2;
% 
% noise = (rand(1,n2_train)*2-1)*0.1;
% %r1 = rand(1,n2_train)*0.4+noise;
% %r1 = rand(1,n2_train)*0.45;
% u1 = rand(1,n2_train)*0.5*0.45^2;
% r1 = sqrt(2*u1)+noise;
% theta1 = rand(1,n2_train)*2*pi;
% %r2 = rand(1,n2_train)*0.5+0.5+noise;
% %r2 = rand(1,n2_train)*0.55+0.45;
% %u2 = rand(1,n2_train)*0.5*0.55^2;
% u2 = (0.5 - 0.45^2/2)*rand(1,n2_train) + 0.45^2/2;
% r2 = sqrt(2*u2)+noise;
% theta2 = rand(1,n2_train)*2*pi;
% X_train1 = zeros(2,n2_train);
% X_train2 = zeros(2,n2_train);
% X_train1(1,:) = r1.*cos(theta1);
% X_train1(2,:) = r1.*sin(theta1);
% X_train2(1,:) = r2.*cos(theta2);
% X_train2(2,:) = r2.*sin(theta2);
% X_train = [X_train1, X_train2];
% 
% Y_train1 = zeros(2,n2_train);
% Y_train1(1,:) = 1;
% Y_train2 = zeros(2,n2_train);
% Y_train2(2,:) = 1;
% Y_train = [Y_train1, Y_train2];
% 
% 
% % n_test = 1000;
% % n2_test = n_test/2;
% % r1 = rand(1,n2_test)*0.4;
% % theta1 = rand(1,n2_test)*2*pi;
% % r2 = rand(1,n2_test)*0.5+0.5;
% % theta2 = rand(1,n2_test)*2*pi;
% % X_test1 = zeros(2,n2_test);
% % X_test2 = zeros(2,n2_test);
% % X_test1(1,:) = r1.*cos(theta1);
% % X_test1(2,:) = r1.*sin(theta1);
% % X_test2(1,:) = r2.*cos(theta2);
% % X_test2(2,:) = r2.*sin(theta2);
% % X_test = [X_test1,X_test2];
% % 
% % Y_test1 = zeros(2,n2_test);
% % Y_test1(1,:) = 1;
% % Y_test2 = zeros(2,n2_test);
% % Y_test2(2,:) = 1;
% % Y_test = [Y_test1, Y_test2];
%---------------------------------

%????3
%------------------------------
% sigma = 0.0;
% n_train = 300;
% n2_train = n_train/2;
% X_train = (rand(2,n_train)*2-1)*sqrt(pi/2);
% noise = (rand(1,n_train)*2-1)*0.2;
% 
% Y_train = zeros(2,n_train);
% for i = 1:n_train
%     if X_train(1,i)^2 + X_train(2,i)^2 < 1
%         Y_train(1,i) = 1;
%     else
%         Y_train(2,i) = 1;
%     end
% end
%--------------------------------
% 
% 
% point = 80;
% n_test = point*point;
% x = linspace(-sqrt(pi/2),sqrt(pi/2),point);
% y = linspace(-sqrt(pi/2),sqrt(pi/2),point);
% X_test = zeros(2,point*point);
% for i = 1:point
%     for j = 1:point
%     X_test(1,j+(i-1)*point) = x(j);
%     X_test(2,j+(i-1)*point) = y(i);
%     end
% end
% 
% Y_test = zeros(2,point*point);
% for i = 1:point
%     for j = 1:point
%         if x(j)^2 + y(i)^2 < 1
%             Y_test(1,j+(i-1)*point) = 1;
%         else
%             Y_test(1,j+(i-1)*point) = 1;
%         end
%     end
% end
% 

[n_train,X_train,Y_train,n_test,X_test,Y_test] = load_data(data_set,data_size);

% Defaul parameters
param.n_train = n_train;
param.n_test = n_test;
%

param.hidden = [100,50];
%param.hidden = [1000,500];
%param.hidden = [1500,1000, 500];    % size of hidden units
param.delta  = [1e-4, 1e-4];
%param.delta  = [0, 0];

% threshold for low-rank approx. of the input data X and others
%param.aeitr  = [ 5, 10, 10];
param.aeitr  = [5, 10, 10];

% # iters for AE          (outer, NMF, nonlinear LSQ)

%param.ftitr  = [100,  1,  1];   
param.ftitr  = [50,  1,  1];
% # iters for fine tuning (outer, semi-NMF, nonlinear semi-NMF)

%param.nsnmf  = [10, 1];        % # iters for LSQs in nonlinear semi-NMF
param.nsnmf  = [10, 1];

param.batch  = [5000, 5000];   % size of batch for AE and fine tuning
%param.lambda = [1e-2,1e-2];
%param.lambda = [0,0];
param.lambda = 0;
%
% Set parameters

% % For CIFAR10
% param.delta  = [5e-3, 1e-14];
% %threshold for SVD of the input data X and others
% param.aeitr  = [20, 10, 10];
% %# iters for AE          (outer, NMF, nonlinear LSQ)
% param.nsnmf  = [25, 25];        % # iters for LSQs in nonlinear semi-NMF

% Write parameters


rng(seed, 'twister');
disp(param)
[WZ,WZ_train,WZ_test,resvec(:,:,seed+1)] = myDeepNN_class(X_train,Y_train,X_test,Y_test,param);

%[WZ,WZ_train,WZ_test,resvec(:,:,seed+1)] = myDeepNN_class(X_train,Y_train,X_test,Y_test,param);

%save('./DATA/resvec_br_3layer_epoch100_alpha6.mat', 'param', 'resvec','WZ')
%save('./DATA/resvec_br_2layer_epoch300_alpha0.mat', 'param', 'resvec','WZ')