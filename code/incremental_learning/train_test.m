function [x_train,y_train,x_test,y_test]=train_test()
% ����ѧϰѵ�����Ͳ��Լ��Ļ���

% ��������
S=load('D:\mfxs\��ҵ���\data.mat');

% ��ͬ���������ݽ�������
x_phog=[S.x_phog_180_level0 S.x_phog_180_level1 S.x_phog_180_level2 S.x_phog_180_level3 S.x_phog_360_level0 S.x_phog_360_level1 S.x_phog_360_level2 S.x_phog_360_level3];
x_sift=[S.x_sift_K300_grey_r4 S.x_sift_K300_grey_r8 S.x_sift_K300_grey_r12 S.x_sift_K300_grey_r16 S.x_sift_K300_hsv_r4 S.x_sift_K300_hsv_r8 S.x_sift_K300_hsv_r12 S.x_sift_K300_hsv_r16 S.x_sift_K1000_grey_r4 S.x_sift_K1000_grey_r8 S.x_sift_K1000_grey_r12 S.x_sift_K1000_grey_r16 S.x_sift_K1000_hsv_r4 S.x_sift_K1000_hsv_r8 S.x_sift_K1000_hsv_r12 S.x_sift_K1000_hsv_r16];
x_regcov=[S.x_regcov_level0 S.x_regcov_level1 S.x_regcov_level2];
x_lbp=[S.x_lbp_level0 S.x_lbp_level1 S.x_lbp_level2];
x=[x_phog x_sift x_regcov x_lbp];

% ��ȥ��Ч����
[m,~]=find(isnan(x));
x(m,:)=[];
S.y(m,:)=[];

% ��ȥ��Ч����
standard_deviation=std(x);
x(:,standard_deviation==0)=[];

% �����������������
feature=size(x,2);
class=size(S.y,2);
train_num=60;
test_num=20;

% ��ʼ��ѵ�����Ͳ��Լ�
x_train=zeros(class*train_num,feature);
y_train=zeros(class*train_num,class);
x_test=zeros(class*test_num,feature);
y_test=zeros(class*test_num,class);

% ���ѡ��������Ϊѵ�����Ͳ��Լ�
for i=1:class
    index=find(S.y(:,i)==1);
    train_test=index(randperm(length(index),train_num+test_num));
    x_train(train_num*(i-1)+1:train_num*i,:)=x(train_test(1:train_num),:);
    y_train(train_num*(i-1)+1:train_num*i,:)=S.y(train_test(1:train_num),:);
    x_test(test_num*(i-1)+1:test_num*i,:)=x(train_test(train_num+1:train_num+test_num),:);
    y_test(test_num*(i-1)+1:test_num*i,:)=S.y(train_test(train_num+1:train_num+test_num),:);
end

% ���ݱ�׼��
[x_train,mean,sigma]=zscore(x_train);
sigma(sigma==0)=1;
x_test=(x_test-repmat(mean,size(x_test,1),1))./repmat(sigma,size(x_test,1),1);
end