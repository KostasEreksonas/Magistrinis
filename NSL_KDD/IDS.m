clear all;

% -------------------------------------------------------------------------
% Load data
names = importdata("nsl-kdd\names");

test_plus = readtable("nsl-kdd\KDDTest+");
test_plus.Properties.VariableNames = names;
test_plus(:, 42) = [];

all_data = readtable("nsl-kdd\KDDTest-21");
all_data.Properties.VariableNames = names;
all_data(:, 42) = [];

train_plus = readtable("nsl-kdd\KDDTrain+");
train_plus.Properties.VariableNames = names;
train_plus(:, 42) = [];

train_20 = readtable("nsl-kdd\KDDTrain+_20Percent");
train_20.Properties.VariableNames = names;
train_20(:, 42) = [];

all_data = [train_plus;train_20;test_plus;all_data];

% -------------------------------------------------------------------------
% Prepare data

% Encode protocol
protocol_type = all_data{:,names(2,:)};
protocol_type = categorical(protocol_type);
protocol_type_onehot = onehotencode(protocol_type,2);
protocol_type_onehot = array2table(protocol_type_onehot);

% Encode service
service = all_data{:,names(3,:)};
service = categorical(service);
service_onehot = onehotencode(service,2);
service_onehot = array2table(service_onehot);

% Encode flag
flag = all_data{:,names(4,:)};
flag = categorical(flag);
flag_onehot = onehotencode(flag,2);
flag_onehot = array2table(flag_onehot);

% Get class column
all_data_class = all_data(:,'class_num');

% Remove categorical data from the table
all_data = removevars(all_data, {'protocol_type'});
all_data = removevars(all_data, {'service'});
all_data = removevars(all_data, {'flag'});
all_data = removevars(all_data, {'class_num'});

% Normalize numerical data
for col = 1:width(all_data)
    t_arr = table2array(all_data(:,col));
    norm = normalize(t_arr,'range');
    all_data{:,col} = (norm);
end

% Add one-hot encoded data to the table
all_data = [all_data protocol_type_onehot];
all_data = [all_data service_onehot];
all_data = [all_data flag_onehot];

% Add class_num variable to dataset for classifierLearner
data_classifier = [all_data all_data_class]; % Data for classifierLearner

% Prepare data for nftool
data_nn = table2array(all_data);
data_nn_class = table2array(all_data_class);

nftool