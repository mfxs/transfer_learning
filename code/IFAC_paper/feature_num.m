function feature_used=feature_num(kernel_weight)
% ����ʵ��ʹ�õ���������

global kernel_length;
feature=length(kernel_length);
scale=length(kernel_weight)/feature;
feature_used=length(find(sum(reshape(kernel_weight,scale,feature),1)));
end