function [alpha_best,bias_best,sv_best,kernel_weight_best,InfoKernel,C_best,hist]=train_mkl(x_train,y_train,x_validation,y_validation,C,bandwidth)
% 多分类多核模型

% 参数设置
options.algo='oneagainstall';
options.seuildiffsigma=1e-2;
options.seuildiffconstraint=0.1;
options.seuildualitygap=1e-2;
options.goldensearch_deltmax=1e-1;
options.numericalprecision=1e-8;
options.stopvariation=1;
options.stopKKT=1;
options.stopdualitygap=1;
options.firstbasevariable='first';
options.nbitermax=500;
options.seuil=0.;
options.seuilitermax=10;
options.lambdareg = 1e-8;
options.miniter=0;
options.verbosesvm=0;
options.efficientkernel=0;

% 核数量
% kernel_length=[20,80,320,1280,40,160,640,2560,300,300,300,300,298,300,298,298,1000,1000,1000,1000,991,995,995,993,28,112,448,37,148,592];
% kernel_length=[20,80,320,1280,40,160,640,2560,300,300,300,300,297,298,297,296,999,1000,1000,1000,990,993,992,991,28,112,448,37,148,592];
global kernel_length;
kernel_number=length(kernel_length);
kernel=cell(1,kernel_number);
kerneloptionvec=cell(1,kernel_number);
variableveccell=cell(1,kernel_number);

% 生成核参数
index=1;
for i=1:kernel_number
    variableveccell{i}=index:index-1+kernel_length(i);
    index=index+kernel_length(i);
    kernel{i}='gaussian';
    kerneloptionvec{i}=bandwidth;
end

% 转化标签
y=label_transform(y_train);

% 训练
accuracy_best=0;
hist=[];
[Weight,InfoKernel]=UnitTraceNormalization(x_train,kernel,kerneloptionvec,variableveccell);
K=mklkernel(x_train,InfoKernel,Weight,options);
for c=C
    disp(['C=',num2str(c)]);
    [kernel_weight,a,bias,pos,nbsv]=mklmulticlass(K,y,c,size(y_train,2),options,1);
    kernel_weight=kernel_weight.*Weight;

%     提取模型参数
    alpha=cell(1,length(nbsv));
    sv=cell(1,length(nbsv));
    temp1=1;
    for i=1:length(nbsv)
        alpha{i}=a(temp1:temp1+nbsv(i)-1);
        sv{i}=x_train(pos(temp1:temp1+nbsv(i)-1),:);
        temp1=temp1+nbsv(i);
    end
    
%     记录不同参数下训练和验证的准确率
    disp('Predict on train set.');
    [~,temp2]=pred(x_train,y_train,alpha,bias,sv,'mkl',[],kernel_weight,InfoKernel);
    disp('Predict on validation set.');
    [~,accuracy]=pred(x_validation,y_validation,alpha,bias,sv,'mkl',[],kernel_weight,InfoKernel);
    hist(1,end+1)=temp2(1);
    hist(2,end)=accuracy(1);
    
%     寻找验证准确率最高的参数
    if accuracy(1)>accuracy_best
        alpha_best=alpha;
        sv_best=sv;
        bias_best=bias;
        kernel_weight_best=kernel_weight;
        C_best=c;
        accuracy_best=accuracy(1);
    end
end
end