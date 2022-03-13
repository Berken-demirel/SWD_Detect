chdir('..')
files = dir('MAT\Voltage*CH1.mat');
folder = files(1).folder;
chdir('DWT')
for i = 1:length(files)
    i
    FEAT_MAT = [];
    test_data = [folder,'\',files(i).name];
    test_mat = create_feature_matrix('Rat_Features',folder,test_data);
end
