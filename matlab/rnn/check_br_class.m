function resvec = check_br_class(itr,tt,A,A_test,Y,Y_test,WZ,L)

WZ_tmp = compute_z_br(A,WZ,L);
Z = WZ_tmp(L).Z;
%Z = appl_f(Z);
WZ_tmp = compute_z_br(A_test,WZ,L);
Z_test = WZ_tmp(L).Z;
%Z_test = appl_f(Z_test);

[~,id_Ytrain] = max(Y);
[~,id_Ztrain] = max(Z);
[~,id_Ytest] = max(Y_test);
[~,id_Ztest] = max(Z_test);

p_train = sum(id_Ztrain==id_Ytrain)/length(id_Ytrain)*100;
p_test = sum(id_Ztest==id_Ytest)/length(id_Ytest)*100;
p_train = 100 - p_train;
p_test = 100 - p_test;
RR = norm(Y-Z,'fro')/norm(Y,'fro');

norm_pro = 1;
for i = 1:L
    norm_pro = norm_pro * norm(WZ(i).W, 'fro');
end

resvec = [itr,tt,RR,p_train,p_test,norm_pro];
    fprintf('%3d  %6.2f  %4.6f  %4.6f  %4.6f\n',itr,tt,RR,p_train,p_test);
end
