addpath('MAT\')
files = dir('absz\HumanData*');
for i = 1:length(files)
    test_data = files(i).name;
    test_mat = create_feature_matrix_humans('Human_Features',test_data);
end
