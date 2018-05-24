function WZ = compute_z_br(A,WZ,L)

a = ones(size(A,2),1);
WZ(1).Z = appl_f(WZ(1).W*[A;a']);
for i = 2:L-1
  a = ones(size(WZ(i-1).Z,2),1);
  WZ(i).Z = appl_f(WZ(i).W*[WZ(i-1).Z;a']);
end
a = ones(size(WZ(L-1).Z,2),1);
WZ(L).Z = WZ(L).W * [WZ(L-1).Z;a'];
%WZ(L).Z = appl_f(WZ(L).Z);

end
