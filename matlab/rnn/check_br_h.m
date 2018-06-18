function resvec = check_br_h(itr,tt,A,A_test,Y,Y_test,WZ,L,Y_max,Y_min)

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

%ÉXÉPÅ[ÉãÇñﬂÇ∑
for i = 1:4
    Y(i,:) = Y(i,:)*Y_max(i)+Y_min(i);
    Y_test(i,:) = Y_test(i,:)*Y_max(i)+Y_min(i);
    Z(i,:) = (Z(i,:)*Y_max(i))+Y_min(i);
    Z_test(i,:) = (Z_test(i,:)*Y_max(i))+Y_min(i);
end


loss = zeros(1,4);
val_loss = zeros(1,4);
all_loss = zeros(1,4);
n_train = size(Y,2);
n_test = size(Y_test,2);
all_Y = [Y,Y_test];
all_Z = [Z,Z_test];
for i = 1:4
    %val_loss(i) = mean(abs([Y_test(i,:),Y(i,:)]-[Z_test(i,:),Z(i,:)])./abs([Y_test(i,:),Y(i,:)]));
    val_loss(i) = mean(abs(Y_test(i,:)-Z_test(i,:))./abs(Y_test(i,:)));
    loss(i) = mean(abs(Y(i,:)-Z(i,:))./abs(Y(i,:)));
    all_loss(i) = mean(abs(all_Y(i,:)-all_Z(i,:))./abs(all_Y(i,:)));
end
%val_loss = norm(Y_test-Z_test,'fro')^2/size(Y_test,2);
norm_pro = 1;
for i = 1:L
  norm_pro = norm_pro * norm(WZ(i).W, 'fro');
end

resvec = [itr,tt,loss,val_loss,all_loss];
%fprintf('%3d  %6.2f  %4.3f  %4.3f  %6.3f  %6.3f   %4.3f\n',itr,tt,loss,val_loss,p_train,p_test,norm_pro);
%fprintf('%3d  %6.2f  %4.3f  %4.3f\n',itr,tt,loss,val_loss);
%fprintf('val_loss:%3d  %6.2f  %6.5f  %6.5f  %6.5f  %6.5f\n',itr,tt,val_loss(1),val_loss(2),val_loss(3),val_loss(4));
fprintf('all_loss:%3d  %6.2f  %6.5f  %6.5f  %6.5f  %6.5f\n',itr,tt,all_loss(1),all_loss(2),all_loss(3),all_loss(4));
%fprintf('loss:%3d  %6.2f  %6.5f  %6.5f  %6.5f  %6.5f\n',itr,tt,loss(1),loss(2),loss(3),loss(4));
end
