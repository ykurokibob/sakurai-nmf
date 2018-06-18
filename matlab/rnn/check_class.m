function resvec = check_class(itr,tt,A,A_test,Y,Y_test,WZ,L)

WZ_tmp = compute_z(A,WZ,L);
Z = WZ_tmp(L).Z;
%Z = appl_f(Z);
WZ_tmp = compute_z(A_test,WZ,L);
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

resvec = [itr,tt,RR,p_train,p_test];
fprintf('%3d  %6.2f  %4.3f  %6.3f  %6.3f \n',itr,tt,RR,p_train,p_test);

end
