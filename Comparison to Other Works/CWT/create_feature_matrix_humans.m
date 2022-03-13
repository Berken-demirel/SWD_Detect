function Feature_Matrix = create_feature_matrix_humans(folderpath, filepath)
% create future matrix for humans

rs_data = load(filepath).data;
animal = regexp(filepath,'\d*','Match');
animal = animal{end};
samples = size(rs_data,1);
freq_bands = [4.4 8.2 ; 8.8 16.4 ; 17.6 32.8 ; 35.1 65.5];
Feature_Matrix = zeros(samples,12);

for i = 1:samples
    data = rs_data(i,:);
    [tf_data,f] = cwt(data,'amor',256);
    % Frequency band (4.4,8.2-8.8,16.4-17.6,32.8-35.1,65.5)
    for band = 1:4
        [sidx,fidx] = find_closest_index(f,freq_bands(band,1),freq_bands(band,2));
        segment = tf_data(fidx:sidx,:);
        abs_sum = sum(abs(segment'));
        Feature_Matrix(i,3*(band-1)+1) = mean(abs_sum);
        Feature_Matrix(i,3*(band-1)+2) = std(abs_sum);
        Feature_Matrix(i,3*(band-1)+3) = max(abs_sum);
        continue
    end
    
end
labels = double(load(filepath).labels);
Feature_Matrix = [Feature_Matrix labels'];
save([folderpath,'\Feats_',animal,'.mat'],'Feature_Matrix')
end


