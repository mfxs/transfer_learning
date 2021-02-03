function [y_best,fy_best]=PSO(x_validation,y_validation,sample_validation,alpha,supportvector,bias,InfoKernel,Weight,kernel_weight)
% 粒子群优化算法
% x代表个体，y代表整体

% 设置模型参数
number_particle=10;
dim=41;
max_iteration=50;
limit=[0,1];
vlimit=[-1,1];
w=0.8;
c1=2;
c2=2;

% 设置初始组合系数为平均，随机初始速度
% x=rand(number_particle,dim);
x=ones(number_particle,dim)/dim+0.02*(rand(number_particle,dim)-0.5);
v=rand(number_particle,dim);

% 初始化个体最佳和整体最佳
x_best=x;
y_best=zeros(1,dim);
fx_best=zeros(number_particle,1);
fy_best=zeros;

% 迭代更新
iteration=1;
record=zeros(max_iteration,1);
while iteration<=max_iteration
    disp(strcat('iteration=',num2str(iteration)));
    tic
%     parfor i=1:number_particle
    for i=1:number_particle
        accuracy=test({alpha,supportvector},bias,x_validation,y_validation,sample_validation,0,'common',InfoKernel,x(i,:)*Weight,x(i,:)*kernel_weight);
        
%         若找到更优的个体适应度
        if fx_best(i)<accuracy(3,1)
            fx_best(i)=accuracy(3,1);
            x_best(i,:)=x(i,:);
        end
    end
    toc
    
%     若找到更优的整体适应度
    if fy_best<max(fx_best)
        [fy_best,nbest]=max(fx_best);
        y_best=x_best(nbest,:);
    end
    
%     更新速度
    v=v*w+c1*rand*(x_best-x)+c2*rand*(repmat(y_best,number_particle,1)-x);
    
%     超出速度限制
    v(v>vlimit(2))=vlimit(2);
    v(v<vlimit(1))=vlimit(1);
    
%     更新位置
    x=x+v;
    
%     超出位置限制
    x(x>limit(2))=limit(2);
    x(x<limit(1))=limit(1);
    
%     记录每次迭代的最佳整体适应度
    record(iteration)=fy_best;
    iteration=iteration+1;
end
end