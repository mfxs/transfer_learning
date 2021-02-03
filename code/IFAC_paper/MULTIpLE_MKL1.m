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
bandwidth=10.^(-1:6);
new_number=[1 3 5 10];

%% ʵ�鿪ʼ
global new;
for new=1:10
    %% ���ݻ���
    disp('ѵ�����Ͳ��Լ��Ļ���');
    [x_train,x_validation,x_train_new,x_test,y_train,y_validation,y_train_new,y_test]...
        =split1(S,new);
    accuracy=zeros(6*length(new_number)+4,3);
    y_pred=cell(length(new_number),6);
    beta_mkl=zeros(size(y_train,2),length(new_number));
    beta_linear=zeros(size(y_train,2),length(new_number));
    beta_rbf=zeros(size(y_train,2),length(new_number));
    beta_akl=zeros(size(y_train,2),length(new_number));
    
    %% ԭʼģ�͹���
    disp('ԭʼ���ģ�͹���');
    [alpha_mkl,bias_mkl,sv_mkl,kernel_weight_mkl,InfoKernel_mkl,C1,hist_mkl]=train_mkl(x_train,y_train,x_validation,y_validation,C,bandwidth_mkl);
    [y_pred_mkl,accuracy(1,:)]=pred(x_test,y_test,alpha_mkl,bias_mkl,sv_mkl,'mkl',[],kernel_weight_mkl,InfoKernel_mkl);
    feature_used(new)=feature_num(kernel_weight_mkl);
    
    disp('ԭʼ����ģ�͹���');
    [alpha_linear,sv_linear,bias_linear,C2,hist_linear]=train_linear(x_train,y_train,x_validation,y_validation,C);
    [y_pred_linear,accuracy(2,:)]=pred(x_test,y_test,alpha_linear,bias_linear,sv_linear,'linear');

    disp('ԭʼRBFģ�͹���');
    [alpha_rbf,sv_rbf,bias_rbf,C3,bandwidth3,hist_rbf]=train_rbf(x_train,y_train,x_validation,y_validation,C,bandwidth);
    [y_pred_rbf,accuracy(3,:)]=pred(x_test,y_test,alpha_rbf,bias_rbf,sv_rbf,'rbf',bandwidth3);
    
    disp('ԭʼƽ����ģ�͹���');
    [alpha_akl,sv_akl,bias_akl,C4,bandwidth4,hist_akl]=train_average(x_train,y_train,x_validation,y_validation,C,bandwidth);
    [y_pred_akl,accuracy(4,:)]=pred(x_test,y_test,alpha_akl,bias_akl,sv_akl,'average',bandwidth4);
    
    %% ԭʼģ��Ǩ��
    for i=1:length(new_number)
        disp('---------------------------');
        disp(strcat('�������������Ϊ',num2str(new_number(i))));
        disp('---------------------------');
        [x_train_new_,y_train_new_]=diff_new_number(x_train_new,y_train_new,new_number(i));
        
        disp('���ģ��Ǩ��');
        [W_mkl{i},beta_mkl(:,i)]=transferlearning(x_train_new_,y_train_new_,'mkl',alpha_mkl,sv_mkl,C1,[],kernel_weight_mkl,InfoKernel_mkl);
        [y_pred{i,1},accuracy(i+4,:)]=pred(x_test,y_test,W_mkl{i}{1},W_mkl{i}{3},W_mkl{i}{2},'mkl',[],kernel_weight_mkl,InfoKernel_mkl);
        
        disp('����ģ��Ǩ��');
        [W_linear{i},beta_linear(:,i)]=transferlearning(x_train_new_,y_train_new_,'linear',alpha_linear,sv_linear,C2,[],[],[]);
        [y_pred{i,2},accuracy(i+8,:)]=pred(x_test,y_test,W_linear{i}{1},W_linear{i}{3},W_linear{i}{2},'linear');
        
        disp('RBFģ��Ǩ��');
        [W_rbf{i},beta_rbf(:,i)]=transferlearning(x_train_new_,y_train_new_,'rbf',alpha_rbf,sv_rbf,C3,bandwidth3,[],[]);
        [y_pred{i,3},accuracy(i+12,:)]=pred(x_test,y_test,W_rbf{i}{1},W_rbf{i}{3},W_rbf{i}{2},'rbf',bandwidth3);
        
        disp('ƽ����ģ��Ǩ��');
        [W_akl{i},beta_akl(:,i)]=transferlearning(x_train_new_,y_train_new_,'average',alpha_akl,sv_akl,C4,bandwidth4,[],[]);
        [y_pred{i,4},accuracy(i+16,:)]=pred(x_test,y_test,W_akl{i}{1},W_akl{i}{3},W_akl{i}{2},'average',bandwidth4);
        
        disp('���¹���ģ��');
        [alpha_new{i},bias_new{i},sv_new{i},kernel_weight_new{i},InfoKernel_new{i},C_new{i},hist_new{i}]...
            =train_mkl(x_train_new_,y_train_new_,x_validation,y_validation,C,bandwidth_mkl);
        [y_pred{i,5},accuracy(i+20,:)]=pred(x_test,y_test,alpha_new{i},bias_new{i},sv_new{i},'mkl',[],kernel_weight_new{i},InfoKernel_new{i});
        
        disp('��Ǩ��ģ��');
        [alpha_ntl{i},sv_ntl{i},bias_ntl{i}]...
            =notransfer(x_train_new_,y_train_new_,alpha_mkl,sv_mkl,bias_mkl,C1,kernel_weight_mkl,InfoKernel_mkl);
        [y_pred{i,6},accuracy(i+24,:)]=pred(x_test,y_test,alpha_ntl{i},bias_ntl{i},sv_ntl{i},'mkl',[],kernel_weight_mkl,InfoKernel_mkl);
    end
    save(['class',num2str(new),'.mat']);

    %% д����
    disp('д�������');
    xlswrite('���.xls',accuracy',1,strcat('C',num2str(3*new),':AD',num2str(3*new+2)));
    
end