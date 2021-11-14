addpath('MAT\')
files = dir('MAT\Voltage*CH1.mat');
for i = 1:length(files)
    FEAT_MAT = [];
    test_data = files(i).name;
    test_mat = create_feature_matrix('Rat_Features',test_data);
end
