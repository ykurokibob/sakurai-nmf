function WZ = compute_w_br(A,WZ,L,lambda,iter)

a = ones(size(A,2),1);

usv = low_rank_appl([WZ(L-1).Z;a'],1e-14);
WZ(L).W = ((WZ(L).Z*usv.V)/usv.S)*usv.U';

for i = L-1:-1:2
  Zusv = low_rank_appl([WZ(i-1).Z;a'],1e-14);
  WZ(i).W = non_linear_lsq_br('XA',WZ(i-1).Z,Zusv,WZ(i).Z,WZ(i).W, ...
                              lambda(1),iter);
  
end

Zusv = low_rank_appl([A;a'],1e-14);
WZ(1).W = non_linear_lsq_br('XA',A,Zusv,WZ(1).Z,WZ(1).W,lambda(1), ...
                            iter);

%%-----------------------------------------

%%-----------------------------------------
end