function G=average_kernel(U,V,bandwidth)
% 平均核函数计算

% 每一个特征的长度
global kernel_length;
% feature_length=[20,80,320,1280,40,160,640,2560,300,300,300,300,297,298,297,296,999,1000,1000,1000,990,993,992,991,28,112,448,37,148,592];

% 计算平均核函数
index=1;
G=zeros(size(U,1),size(V,1));
for i=kernel_length
    r2=repmat(sum(U(:,index:index+i-1).^2,2),1,size(V,1))+repmat(sum(V(:,index:index+i-1).^2,2),1,size(U,1))'-2*U(:,index:index+i-1)*V(:,index:index+i-1)';
    for j=bandwidth
        G=G+exp(-r2/(2*j^2));
    end
    index=index+i;
end
G=G/(length(bandwidth)*length(kernel_length));
end