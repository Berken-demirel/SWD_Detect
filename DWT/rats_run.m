% Animal list
animals = [15 16 17 86 88 89 90 91 92 103 104];

% Empty table
sz = [11 6];
varTypes = ["double","double","double","double","double","double"];
varNames = ["ID","Accuracy","Sensitivity","Specificity","FD_per_hour","Kappa"];
metric_table = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);
metric_table.ID = animals';

files = dir('Rat_Features\*.mat');
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

    % fit SVM with hyperparameters in the publication
    tic
    svm_model = fitcsvm(Train_Feats(:,1:12),Train_Feats(:,13),...
        'BoxConstraint',10,'KernelScale',10,'Cost',[0 1;1.5 0],...
        'KernelFunction','gaussian');
    toc
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
    
    % Calculate Kappa
    p1 = (TP+FN) / (TP+FP+TN+FN);
    p2 = (TP+FP) / (TP+FP+TN+FN);
    random_accuracy = p1*p2 + (1-p1)*(1-p2);
    kappa = (accuracy*0.01-random_accuracy) / (1-random_accuracy);
    
    % Fill table
    metric_table.Specificity(i) = specificity;
    metric_table.Sensitivity(i) = sensitivity;
    metric_table.Accuracy(i) = accuracy;
    metric_table.FD_per_hour(i) = FD;
    metric_table.Kappa(i) = kappa;
    metric_table
end

save('rat_results_SVM.mat','metric_table');

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
    
    % Calculate Kappa
    p1 = (TP+FN) / (TP+FP+TN+FN);
    p2 = (TP+FP) / (TP+FP+TN+FN);
    random_accuracy = p1*p2 + (1-p1)*(1-p2);
    kappa = (accuracy*0.01-random_accuracy) / (1-random_accuracy);
    
    % Fill table
    metric_table.Specificity(i) = specificity;
    metric_table.Sensitivity(i) = sensitivity;
    metric_table.Accuracy(i) = accuracy;
    metric_table.FD_per_hour(i) = FD;
    metric_table.Kappa(i) = kappa;
    metric_table
end

save('rat_results_Tree.mat','metric_table');




