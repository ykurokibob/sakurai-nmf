
n_test = 324;
data = zeros(n_test,8);

data(:,1:4) = Y_test';
data(:,5:8) = WZ_val(end).Z';

csvwrite('test_data.csv',data);

