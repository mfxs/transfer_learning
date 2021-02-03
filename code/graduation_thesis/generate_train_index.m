function train_index=generate_train_index(sample_train)
% 生成训练样本的索引

class=length(sample_train);
train_index=cell(1,class);
trainnumber_per_class=5;

% 从其余每类中随机选择5个作为训练样本（欠采样）
otherindex=zeros((class-1)*trainnumber_per_class,1);
for i=1:class
    k=1;
    for j=1:class
        if j~=i
        otherindex(k:k-1+trainnumber_per_class)=sample_train{j}(randperm(length(sample_train{j}),trainnumber_per_class));
        k=k+trainnumber_per_class;
        end
    end
    train_index{i}=[sample_train{i};otherindex];
end
end