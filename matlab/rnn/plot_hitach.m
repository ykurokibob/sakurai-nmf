B = zeros(10,4);
for i = 1:10
    for j = 1:4
        B(i,j) = resvec{1,i}(end,j+10);
    end
end

layer = linspace(1,10,10);

plot(layer,B(:,1),'b','lineWidth',2);
hold on
plot(layer,B(:,2),'g','lineWidth',2);
plot(layer,B(:,3),'r','lineWidth',2);
plot(layer,B(:,4),'k','lineWidth',2);
xlabel('n\_imlayer');
grid on
legend('B1','B2','B3','B4');

