function [y_best,fy_best]=PSO(x_validation,y_validation,sample_validation,alpha,supportvector,bias,InfoKernel,Weight,kernel_weight)
% ����Ⱥ�Ż��㷨
% x������壬y��������

% ����ģ�Ͳ���
number_particle=10;
dim=41;
max_iteration=50;
limit=[0,1];
vlimit=[-1,1];
w=0.8;
c1=2;
c2=2;

% ���ó�ʼ���ϵ��Ϊƽ���������ʼ�ٶ�
% x=rand(number_particle,dim);
x=ones(number_particle,dim)/dim+0.02*(rand(number_particle,dim)-0.5);
v=rand(number_particle,dim);

% ��ʼ��������Ѻ��������
x_best=x;
y_best=zeros(1,dim);
fx_best=zeros(number_particle,1);
fy_best=zeros;

% ��������
iteration=1;
record=zeros(max_iteration,1);
while iteration<=max_iteration
    disp(strcat('iteration=',num2str(iteration)));
    tic
%     parfor i=1:number_particle
    for i=1:number_particle
        accuracy=test({alpha,supportvector},bias,x_validation,y_validation,sample_validation,0,'common',InfoKernel,x(i,:)*Weight,x(i,:)*kernel_weight);
        
%         ���ҵ����ŵĸ�����Ӧ��
        if fx_best(i)<accuracy(3,1)
            fx_best(i)=accuracy(3,1);
            x_best(i,:)=x(i,:);
        end
    end
    toc
    
%     ���ҵ����ŵ�������Ӧ��
    if fy_best<max(fx_best)
        [fy_best,nbest]=max(fx_best);
        y_best=x_best(nbest,:);
    end
    
%     �����ٶ�
    v=v*w+c1*rand*(x_best-x)+c2*rand*(repmat(y_best,number_particle,1)-x);
    
%     �����ٶ�����
    v(v>vlimit(2))=vlimit(2);
    v(v<vlimit(1))=vlimit(1);
    
%     ����λ��
    x=x+v;
    
%     ����λ������
    x(x>limit(2))=limit(2);
    x(x<limit(1))=limit(1);
    
%     ��¼ÿ�ε��������������Ӧ��
    record(iteration)=fy_best;
    iteration=iteration+1;
end
end