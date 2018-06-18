function WZ = compute_z_br_h(A,WZ,L,Y_max,Y_min)

a = ones(size(A,2),1);
WZ(1).Z = appl_f(WZ(1).W*[A;a']);
for i = 2:L-1
  a = ones(size(WZ(i-1).Z,2),1);
  WZ(i).Z = appl_f(WZ(i).W*[WZ(i-1).Z;a']);
end
a = ones(size(WZ(L-1).Z,2),1);
WZ(L).Z = WZ(L).W * [WZ(L-1).Z;a'];
for i = 1:4
    WZ(L).Z(i,:) = WZ(L).Z(i,:)*Y_max(i)+Y_min(i);
end
%WZ(L).Z = appl_f(WZ(L).Z);

end
