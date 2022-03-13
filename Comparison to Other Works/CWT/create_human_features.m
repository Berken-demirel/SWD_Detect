chdir('..')
files = dir('absz\HumanData*');
folder = files(1).folder;
chdir('CWT')
for i = 1:length(files)
    i
    test_data = [folder,'/',files(i).name];
    test_mat = create_feature_matrix_humans('Human_Features',test_data);
end
