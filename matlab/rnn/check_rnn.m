function resvec = check_rnn(itr,tt,A,A_test,Y,Y_test,WZ,L,param)

WZ_tmp = compute_rnn(A,WZ,L,param.hid_dims);
Z = WZ_tmp(L).Z;
%Z = appl_f(Z);
size(A_test);
WZ_test = compute_rnn(A_test,WZ,L,param.hid_dims);
%WZ_tmp = compute_z_rnn(Y,WZ,L);
Z_test = WZ_test(L).Z;
%Z_test = appl_f(Z_test);
%
% [~,id_Ytrain] = max(Y);
% [~,id_Ztrain] = max(Z);
% [~,id_Ytest] = max(Y_test);
% [~,id_Ztest] = max(Z_test);

% p_train = sum(id_Ztrain==id_Ytrain)/length(id_Ytrain)*100;
% p_test = sum(id_Ztest==id_Ytest)/length(id_Ytest)*100;
% p_train = 100 - p_train;
% p_test = 100 - p_test;
%RR = norm(Y-Z,'fro')/norm(Y,'fro');
%norm_loss = norm((Y'-Z),'fro')^2/norm(Y','fro')^2;
%norm_loss = norm((Y'-Z),'fro')^2/norm(Y','fro')^2;
norm_loss = sum((Y'-Z).^2)/max(size(Z));
size(Y_test);
size(Z_test);
%norm_val_loss = norm((Y_test'-Z_test),'fro')^2/norm(Y_test','fro')^2;
norm_val_loss = sum((Y_test'-Z_test).^2)/max(size(Z_test));

resvec = [itr,tt,norm_loss,norm_val_loss];
if mod(itr,100)==0 || itr==1
    fprintf('%3d  %6.2f  %4.6f  %4.6f \n',itr,tt,norm_loss,norm_val_loss);
end
end
