function train_new_index=generate_train_new_index(sample_train_new)
% 生成不迁移方法的新训练样本索引

class=length(sample_train_new);
otherindex=[];

% 从其余样本中随机选择两倍于新类别样本个数的样本用于训练
for i=1:class-1
    otherindex=[otherindex;sample_train_new{i}];
end
train_new_index=[sample_train_new{end};otherindex(randperm(length(otherindex),2*length(sample_train_new{end})))];
end