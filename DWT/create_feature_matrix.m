function Feature_Matrix = create_feature_matrix(folderpath, filepath)
rs_data = load(filepath).Voltage_CH1;
animal = filepath(15:end-7);
samples = size(rs_data,1);
freq_bands = [0 3.75 ; 3.75 7.5 ; 7.5 15];
Feature_Matrix = zeros(samples,12);

for i = 1:samples
    data = rs_data(i,:);
    [A1,D1] = dwt(data,'db2');
    [A2,D2] = dwt(A1,'db2');
    tf_data = cell(1,3);
    tf_data{1} = D1;
    tf_data{2} = D2;
    tf_data{3} = A2;
    for band = 1:3
        coefs = tf_data{band};
        Feature_Matrix(i,4*(band-1)+1) = mean(coefs);
        Feature_Matrix(i,4*(band-1)+2) = std(coefs);
        Feature_Matrix(i,4*(band-1)+3) = max(coefs);
        Feature_Matrix(i,4*(band-1)+4) = min(coefs);
        continue
    end
    
end
labels = load(['labels_Animal',animal,'.mat']).Final_labels;
labels = labels(1,:);
Feature_Matrix = [Feature_Matrix labels'];
save([folderpath,'\Feats_',animal,'.mat'],'Feature_Matrix')
end
