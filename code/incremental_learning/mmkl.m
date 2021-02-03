function [alpha,bias,supportvector,kernel_weight,InfoKernel]=mmkl(x_train,y_train,bandwidth,C)
% 多分类多核模型

addpath('D:\mfxs\SimpleMKL');

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
kernel_length=[20,80,320,1280,40,160,640,2560,300,300,300,300,297,298,297,296,999,1000,1000,1000,990,993,992,991,28,112,448,37,148,592];
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
y=zeros(size(y_train,1),1);
for i=1:size(y,1)
    y(i)=find(y_train(i,:)==1);
end

% 训练
[Weight,InfoKernel]=UnitTraceNormalization(x_train,kernel,kerneloptionvec,variableveccell);
K=mklkernel(x_train,InfoKernel,Weight,options);
[kernel_weight,a,bias,pos,nbsv,~,~] = mklmulticlass(K,y,C,size(y_train,2),options,1);
kernel_weight=kernel_weight.*Weight;

alpha=cell(1,length(nbsv));
supportvector=cell(1,length(nbsv));
temp=1;
for i=1:length(nbsv)
    alpha{i}=a(temp:temp+nbsv(i)-1);
    supportvector{i}=x_train(pos(temp:temp+nbsv(i)-1),:);
    temp=temp+nbsv(i);
end
end