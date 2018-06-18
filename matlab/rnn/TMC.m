T=20;
shield=0;
MA=70;
Mm=100-MA;
Tn=4+shield;
left=Tn
%THP=zeros(1,Tn);
turn=8;
liveM=ones(1,turn);
TnT=ones(1,turn)*Tn;
Dm=zeros(1,turn);
Dice=zeros(1,100);
for i=1:100
    Dice(i)=i;
end
%Dmin=1;
%for t=1:Tn
THP = floor((sum(floor((6).*rand(5,4) + 1))+6)/2);%sum(floor((Dice(6)).*rand(5,Tn) + 1)))/2;
THP =[THP,ones(1,shield)*max(THP)];
THP=sort(THP);
iter=100;
per=zeros(1,100);
%end
count=1
%for count=1:iter
    for i=1:turn
        liveM(i+1)= liveM(i)*((100-T)/100)^TnT(i);
        MF=floor((100).*rand(1,1) + 1);
        if MF<=MA
            Dm(i)= floor(rand(1,1)*10)+1+floor(rand(1,1)*10)+1+floor(rand(1,1)*10)+1;
            if Dm(i)>=max(THP(1:left))
                TnT(i)=TnT(i)-1;
                left=left-1;
            end
            
        end
        TnT(i+1)=TnT(i);
        
        if TnT(i+1)==0
            liveM*100
            per(1,count)=min(liveM)*100;
            break
        end
        
    end
    liveM*100 
    per(1,count)=min(liveM)*100;
%end
per(1,1)