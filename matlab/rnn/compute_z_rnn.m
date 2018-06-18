function WZ = compute_z_rnn(A,WZ,L)
%a = ones(size(A,2),1);
% j
% sizeA=size(A);
Z0=zeros(1,size(A,1));
size(WZ(1).Win);
size(WZ(1).Wre);
size([WZ(1).Wre,WZ(1).Win]);
size(A(:,1)');
size(Z0);

%WZ(1).Z = appl_f([WZ(1).Win,WZ(1).Wre]*[A(:,1)';Z0']);
WZ(1).Z = appl_f([WZ(1).Wre,WZ(1).Win]*[Z0;A(:,1)']);
% sizeZ=size(WZ(1).Z);
% sizewin1=size(WZ(1).Win);
% sizewre1=size(WZ(1).Wre);
% for i=1:L
%     WZ(i).Win
%     WZ(i).Wre
%     WZ(i).Z
% end    
size(WZ(1).Z);
for i = 2:L-1

%     
%     sizewin=size(WZ(i).Win);
%     sizewre=size(WZ(i).Wre);
    %a = ones(size(WZ(i-1).Z,2),1);
%     WZ(i).Win;
%     WZ(i).Wre;
%     WZ(i-1).Z;
%     A(i,j);
   % i
    %WZ(i).Z = appl_f([WZ(i).Win,WZ(i).Wre]*[WZ(i-1).Z;A(i,j)]);
    %WZ(i).Z = appl_f([WZ(i).Win,WZ(i).Wre]*[A(i,j);WZ(i-1).Z]);
    %WZ(i).Z = appl_f([WZ(i).Win,WZ(i).Wre]*[A(:,i)';WZ(i-1).Z]);
    WZ(i).Z = appl_f([WZ(1).Wre,WZ(1).Win]*[WZ(i-1).Z;A(:,i)']);
    
end
%WZ(L).Z = [WZ(1).Wout]*[WZ(L-1).Z];
%WZZ=size(WZ(10).Z);
%a = ones(size(WZ(L-1).Z,2),1);
%WZ(L).Z = 1. * [WZ(L-1).Z];
%WZ(L).Z = appl_f(WZ(L).Z);

end
