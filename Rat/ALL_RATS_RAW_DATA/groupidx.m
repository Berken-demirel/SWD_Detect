function [groupidx, groups] = groupidx(labels)
names = labels;
labels = grp2idx(labels);
groupidx = {};
groups = {};
start = 1;
for i = 1:length(labels)-1
    if ~(labels(i)==labels(i+1))
        finish = i;
        groupidx{end+1} = [start finish];
        groups{end+1} = names(i);
        start = i+1;
    end
end
groupidx{end+1} = [start,length(labels)];
groups{end+1} = names(end);
