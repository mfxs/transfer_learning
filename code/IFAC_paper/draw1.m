%% ��������
warning('off');
class=11;
data=xlsread('���_best2.xls',1,strcat('C',num2str(3*class),':AD',num2str(3*class+2)));
data=100*data;
load('N+3.mat','correct')
%% ���Ʋ�ͬģ��׼ȷ���������������ı仯����
createfigure([1,3,5,10], data(3,5:end))
createfigure1([1,3,5,10], data(2,:))
createfigure2([1,3,5,10], data(2,:))
createfigure3(correct)