function train_new_index=generate_train_new_index(sample_train_new)
% ���ɲ�Ǩ�Ʒ�������ѵ����������

class=length(sample_train_new);
otherindex=[];

% ���������������ѡ�������������������������������ѵ��
for i=1:class-1
    otherindex=[otherindex;sample_train_new{i}];
end
train_new_index=[sample_train_new{end};otherindex(randperm(length(otherindex),2*length(sample_train_new{end})))];
end