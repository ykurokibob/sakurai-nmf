function resvec = check_br_test(itr,tt,A,A_test,Y,Y_test,WZ,L)

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

loss = zeros(1,4);
val_loss = zeros(1,4);
n_train = size(Y,2);
n_test = size(Y_test,2);
for i = 1:4
%     loss(i) = norm((Y(i,:)-Z(i,:))./Y(i,:),1)/n_train;
%     val_loss(i) = norm((Y_test(i,:)-Z_test(i,:))./Y_test(i,:),'fro')/n_test;
        %loss(i) = mean((Y(i,:)-Z(i,:)).^2);
        val_loss(i) = mean(abs(Y_test(i,:)-Z_test(i,:))/abs(Y_test(i,:)));
        %loss(i) = norm(Y_test(i,:)-Z_test(i,:),'fro')^2/n_test;
end
%val_loss = norm(Y_test-Z_test,'fro')^2/size(Y_test,2);
norm_pro = 1;
for i = 1:L
  norm_pro = norm_pro * norm(WZ(i).W, 'fro');
end

resvec = [itr,tt,loss,val_loss,p_train,p_test,norm_pro];
%fprintf('%3d  %6.2f  %4.3f  %4.3f  %6.3f  %6.3f   %4.3f\n',itr,tt,loss,val_loss,p_train,p_test,norm_pro);
%fprintf('%3d  %6.2f  %4.3f  %4.3f\n',itr,tt,loss,val_loss);
fprintf('val_loss:%3d  %6.2f  %6.5f  %6.5f  %6.5f  %6.5f\n',itr,tt,val_loss(1),val_loss(2),val_loss(3),val_loss(4));
%fprintf('loss:%3d  %6.2f  %6.5f  %6.5f  %6.5f  %6.5f\n',itr,tt,loss(1),loss(2),loss(3),loss(4));
end
