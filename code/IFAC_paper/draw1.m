%% 导入数据
warning('off');
class=11;
data=xlsread('结果_best2.xls',1,strcat('C',num2str(3*class),':AD',num2str(3*class+2)));
data=100*data;
load('N+3.mat','correct')
%% 绘制不同模型准确率随新样本个数的变化曲线
createfigure([1,3,5,10], data(3,5:end))
createfigure1([1,3,5,10], data(2,:))
createfigure2([1,3,5,10], data(2,:))
createfigure3(correct)