% size = 3;
% x = linspace(0,1,size);
% y = linspace(0,1,size);
% xy = zeros(1,9);
% xy(5) = 1;
% 
% input = zeros(2,size*size);
% for i = 1:3
%     for j = 1:3
%         input(1,j+(i-1)*size) = x(j);
%         input(2,j+(i-1)*size) = y(i);
%     end
% end
% 
% for i = 1:size
%     for j = 1:size
%         if xy(j+(i-1*size) == 0
%             if i == 
%             fill([x(j),x(j+1)],[y(i),y(i+1)],'r');
%         else
%                 xx = [input(1,



for i = 1:n_test
    [~,id] = max(WZ_test(3).Z(:,i));
    if id == 1
        plot(X_test(1,i),X_test(2,i),'bs','MarkerFaceColor','b')
    else
        plot(X_test(1,i),X_test(2,i),'rs','MarkerFaceColor','r')
    end
    hold on;
end

for i = 1:n_train
    [~,id] = max(Y_train(:,i));
    if id == 1
        plot(X_train(1,i),X_train(2,i),'ko')
    else
        plot(X_train(1,i),X_train(2,i),'kx')
    end
end
theta = linspace(0,2*pi,1000);
plot(sin(theta),cos(theta),'k','LineWidth',2.0);

%t = linspace(0,2*pi,1000);
%plot(sin(t)*0.45,cos(t)*0.45,'k');
xlim([-sqrt(pi/2),sqrt(pi/2)]);
ylim([-sqrt(pi/2),sqrt(pi/2)]);
        