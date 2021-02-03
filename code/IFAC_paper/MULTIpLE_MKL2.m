%% ���빤����·��
addpath('D:\mfxs\SimpleMKL','D:\mfxs\libsvm-3.23\matlab');

%% ��������
disp('loading...');
S=load('data.mat');
disp('loading done');

%% ��������Χ����
global bandwidth_mkl;
bandwidth_mkl=10.^(1:4);
C=10.^(-1:6);
correct=zeros(4,13);

%% ʵ��ʮ��
for time=1:10
    %% ���ݻ���
    [x_train,y_train,x_val,y_val,x_train_new,y_train_new,x_test,y_test]=split2(S);

    %% ԭʼģ��ѵ��
    disp('---------------------------');
    disp('ѵ��ԭʼģ��');
    disp('---------------------------');
    [alpha,bias,sv,kernel_weight,InfoKernel,C_best,hist]=train_mkl(x_train,y_train,x_val,y_val,C,bandwidth_mkl);
    [~,~,temp]=pred(x_test,y_test,alpha,bias,sv,'mkl',[],kernel_weight,InfoKernel);
    correct(1,:)=correct(1,:)+temp;

    %% ģ��Ǩ��
    disp('---------------------------');
    disp('ģ��Ǩ��');
    disp('---------------------------');
    W=cell(1,4);
    W{1}={alpha,sv,bias};
    for i=1:3
        disp(['Ǩ�Ƶ�',num2str(i),'��']);
        W{i+1}=transferlearning(x_train_new{i},y_train_new{i},'mkl',W{i}{1},W{i}{2},C_best,[],kernel_weight,InfoKernel);
        [~,~,temp]=pred(x_test,y_test,W{i+1}{1},W{i+1}{3},W{i+1}{2},'mkl',[],kernel_weight,InfoKernel);
        correct(i+1,:)=correct(i+1,:)+temp;
    end
end
correct=correct/time;

%% ��ͼ
for i=1:4
    subplot(2,2,i);
    bar(correct(i,:),'y','edgecolor','none');
    hold on
    bar(correct(i,1:end-1),'r','edgecolor','none');
    hold on
    bar(correct(i,1:end-2),'b','edgecolor','none');
    hold on
    bar(correct(i,1:end-3),'g','edgecolor','none');
    grid on;
    axis([0 14 0 20]);
    xlabel('Class');
    ylabel('Correct Number');
%     legend('Correct Prediction');
    if i==1
        title('N-class classifier');
    else
        title(['N+',num2str(i-1),'-class classifier']);
    end
end
suptitle('Correct predicted sample');