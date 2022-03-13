% Animal list
humans = [675 1113 1413 1795 1984 2448 2657 3053 3281 3306 3635 8608];

% Empty table
sz = [12 5];
varTypes = ["double","double","double","double","double"];
varNames = ["ID","Accuracy","Sensitivity","Specificity","FD_per_hour"];
metric_table = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);
metric_table.ID = humans';

files = dir('Human_Features\*.mat');
folder = files(1).folder;

% LOOCV loop
for i = 1:length(files)
    display(['Now at file ',num2str(i)]);
    name = files(i).name;
    % Test Data
    test_idx = i;
    test_feat = load([folder,'\',name]).Feature_Matrix;
    test_labels = test_feat(:,13);
    test_data = test_feat(:,1:12);
    % Train Data
    Train_Feats = [];
    train_idx = setdiff(1:11,i);
    for j = train_idx
        temp_data = load([folder,'\',files(j).name]).Feature_Matrix;
        Train_Feats = [Train_Feats; temp_data];
    end

    % fit SVM 
    svm_model = fitcsvm(Train_Feats(:,1:12),Train_Feats(:,13),...
        'BoxConstraint',10,'KernelScale',10,'Cost',[0 1;1.5 0],...
        'KernelFunction','gaussian');
    predicted_labels = predict(svm_model, test_data);
    CONF_MAT = confusionmat(test_labels,predicted_labels);
    
    % Grab elements of conf mat
    TP = CONF_MAT(2,2);
    FN = CONF_MAT(2,1);
    FP = CONF_MAT(1,2);
    TN = CONF_MAT(1,1);

       % Metrics 
    accuracy = 100*(TP + TN) / (TN + FP + FN + TP);
    sensitivity = 100*TP / (FN + TP);
    specificity = 100*TN / (FP + TN);
    FD = FP * 720 / (TN + FP + FN + TP);
    
    % Fill table
    metric_table.Specificity(i) = specificity;
    metric_table.Sensitivity(i) = sensitivity;
    metric_table.Accuracy(i) = accuracy;
    metric_table.FD_per_hour(i) = FD;
end
metric_table
save('human_results_SVM.mat','metric_table');

% Empty table
sz = [12 5];
varTypes = ["double","double","double","double","double"];
varNames = ["ID","Accuracy","Sensitivity","Specificity","FD_per_hour"];
metric_table = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);
metric_table.ID = humans';

files = dir('Human_Features\*.mat');
folder = files(1).folder;

% LOOCV loop
for i = 1:length(files)
    display(['Now at file ',num2str(i)]);
    name = files(i).name;
    % Test Data
    test_idx = i;
    test_feat = load([folder,'\',name]).Feature_Matrix;
    test_labels = test_feat(:,13);
    test_data = test_feat(:,1:12);
    % Train Data
    Train_Feats = [];
    train_idx = setdiff(1:11,i);
    for j = train_idx
        temp_data = load([folder,'\',files(j).name]).Feature_Matrix;
        Train_Feats = [Train_Feats; temp_data];
    end

    % fit tree
    tree_model = fitctree(Train_Feats(:,1:12),Train_Feats(:,13));
    predicted_labels = predict(tree_model, test_data);
    CONF_MAT = confusionmat(test_labels,predicted_labels);
    
    % Grab elements of conf mat
    TP = CONF_MAT(2,2);
    FN = CONF_MAT(2,1);
    FP = CONF_MAT(1,2);
    TN = CONF_MAT(1,1);

    % Metrics 
    accuracy = 100*(TP + TN) / (TN + FP + FN + TP);
    sensitivity = 100*TP / (FN + TP);
    specificity = 100*TN / (FP + TN);
    FD = FP * 720 / (TN + FP + FN + TP);
    
    % Fill table
    metric_table.Specificity(i) = specificity;
    metric_table.Sensitivity(i) = sensitivity;
    metric_table.Accuracy(i) = accuracy;
    metric_table.FD_per_hour(i) = FD;
end
metric_table
save('human_results_Tree.mat','metric_table');

