chdir("..")
files = dir('absz\HumanData*');
folder = files(1).folder;
chdir('DWT')
for i = 1:length(files)
    i
    test_data = files(i).name;
    test_mat = create_feature_matrix_humans('Human_Features',[folder,'/',test_data]);
end
