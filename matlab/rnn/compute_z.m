function WZ = compute_z(A,WZ,L)

WZ(1).Z = appl_f(WZ(1).W*A);
for i = 2:L-1
    WZ(i).Z = appl_f(WZ(i).W*WZ(i-1).Z);
end
WZ(L).Z = WZ(L).W * WZ(L-1).Z;
%WZ(L).Z = appl_f(WZ(L).Z);

end
