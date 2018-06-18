function WZ = compute_rnn(A,WZ,L,hid)
Z0=zeros(hid,size(A,1));

WZ(1).Z = appl_f([WZ(1).Wre,WZ(1).Win]*[Z0;A(:,1)']);
size(WZ(1).Z);
for i = 2:L-1

%     
%     sizewin=size(WZ(i).Win);
%     sizewre=size(WZ(i).Wre);
    %a = ones(size(WZ(i-1).Z,2),1);
     WZ(1).Win;
     WZ(1).Wre;
     size(WZ(i-1).Z);
     i;
     size(A(:,i)');
   % i
    %WZ(i).Z = appl_f([WZ(i).Win,WZ(i).Wre]*[WZ(i-1).Z;A(i,j)]);
    %WZ(i).Z = appl_f([WZ(i).Win,WZ(i).Wre]*[A(i,j);WZ(i-1).Z]);
    %WZ(i).Z = appl_f([WZ(i).Win,WZ(i).Wre]*[A(:,i)';WZ(i-1).Z]);
    
    WZ(i).Z = appl_f([WZ(1).Wre,WZ(1).Win]*[WZ(i-1).Z;A(:,i)']);
    
end
WZ(L).Z = [WZ(1).Wout]*[WZ(L-1).Z];

%WZ(L).Z;


%WZZ=size(WZ(10).Z);
%a = ones(size(WZ(L-1).Z,2),1);
%WZ(L).Z = 1. * [WZ(L-1).Z];
%WZ(L).Z = appl_f(WZ(L).Z);

end
