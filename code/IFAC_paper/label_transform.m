function y_vec=label_transform(y_mat)
% 将矩阵标签转化为向量标签

y_vec=zeros(size(y_mat(:,1)));
for i=1:size(y_mat,1)
    y_vec(i)=find(y_mat(i,:)==1);
end
end