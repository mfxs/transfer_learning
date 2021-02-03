function confused_matrix(y_pred,y_test)
% 绘制混淆矩阵

% 计算类别数和初始化混淆矩阵
class=size(y_test,2);
matrix=zeros(class,class);

% 计算混淆矩阵元素
for i=1:size(y_test,1)
    [~,index]=max(y_pred(i,:));
    matrix(index,y_test(i,:)==1)=matrix(index,y_test(i,:)==1)+1;
end

% 画图
imagesc(matrix,[0 20]);
xlabel('Real label');
ylabel('Predict label');
% heatmap=HeatMap(matrix,'rowlabels',1:class,'columnlabels',1:class,'annotate',1,'columnlabelsrotate',0,'symmetric',0,'colormap','bone');
% heatmap.addXLabel('Real label');
% heatmap.addYLabel('Predict label');
end