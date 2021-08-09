%% EDF LABELLER 
% This script returns the voltage and label files from the edf files of
% different animals, subject-by-subject. 

edfs = ["bazal_icv_GA_HCN#15-16-17 030517(2).edf","bazal_icv_GA#86-87-88-89 100818.edf","bazal_icv_GA#90-91-92-93 140818.edf","bazal_icv_GA_HCN#103-104-105-107_251219(1).edf"];

file_names{1} = [15 16 17];
file_names{2} = [86 87 88 89];
file_names{3} = [90 91 92];
file_names{4} = [103 104];

ALLCHANNELS = [1 2;3 4;5 6;7 8];

for edf = 1:4
    animal_channels{edf} = ALLCHANNELS(1:length(file_names{edf}),:);
end

for edf = 1:4
    [hdr, record] = edfread(edfs(edf));
    d2s = 24*3600;
    stime = hdr.starttime;
    stime(3) = ':';
    stime(6) = ':';
    start_time_of_record = timeofday(datetime(stime));
    for animal = file_names{edf}
        animal_idx = find(file_names{edf}==animal);
        [ns,samples] = size(record);
        data = readtable(['epilepsy-times',num2str(animal),'.xlsx']);

        last_cell = num2str(size(data,1));

        excel_times_start = timeofday(datetime(xlsread(['epilepsy-times',num2str(animal),'.xlsx'],['A1:A',char(last_cell)]),'convertfrom','excel'));
        start_time = excel_times_start(1);

        excel_times_finish = timeofday(datetime(xlsread(['epilepsy-times',num2str(animal),'.xlsx'],['B1:B',char(last_cell)]),'convertfrom','excel'));
        finish_time = excel_times_finish(end);

        STARTS = sec2samp(seconds(excel_times_start-start_time_of_record),1000);
        DURATIONS = sec2samp(xlsread(['epilepsy-times',num2str(animal),'.xlsx'],['C1:C',char(last_cell)]),1000);

        labels = repmat({'NOT-EPILEPSY'},1,samples);

        for i = 1:size(DURATIONS,1)
            start = STARTS(i);
            duration = DURATIONS(i);
            labels(1,start:start+duration-1) = {'EPILEPSY'};
        end

        analysis_start = sec2samp(seconds(timeofday(datetime(char(start_time))))-seconds(start_time_of_record),1000);
        analysis_finish = sec2samp(seconds(timeofday(datetime(char(finish_time))))-seconds(start_time_of_record),1000);
        Labels = labels(analysis_start:analysis_finish);
        truncated_record = record(:,analysis_start:analysis_finish);

        [labelidx,labels] = groupidx(Labels);

        Voltage = {};
        Final_labels = {};
        frames = size(truncated_record,2);
        for i = 1:frames
            start = 5000*(i-1)+1;
            finish = 5000*i;
            if finish < frames
                Voltage{end+1} = truncated_record(animal_channels{edf}(animal_idx,:),start:finish)';
                secs_of_epilepsy_in_segment = sum(strcmp(Labels(start:finish),'EPILEPSY'));
                if secs_of_epilepsy_in_segment > 1 * hdr.frequency(2*animal_idx-1)
                    Final_labels{end+1,1} = 1;
                else
                    Final_labels{end+1,1} = 0;
                end
                Final_labels{end,2} = secs_of_epilepsy_in_segment;
            else
                break
            end
        end

        Final_labels = transpose(cell2mat(Final_labels));

        Voltage = vertcat(Voltage{:});

        Voltage_CH1 = reshape(Voltage(:,1),5000,length(Voltage(:,1))/5000)';
        Voltage_CH2 = reshape(Voltage(:,2),5000,length(Voltage(:,2))/5000)';

        save(['DENEME/Voltage_Animal',num2str(animal),'.mat'],'Voltage');
        save(['DENEME/Voltage_Animal',num2str(animal),'CH1.mat'],'Voltage_CH1');
        save(['DENEME/Voltage_Animal',num2str(animal),'CH2.mat'],'Voltage_CH2');
        save(['DENEME/labels_Animal',num2str(animal),'.mat'],'Final_labels')
    end
end

