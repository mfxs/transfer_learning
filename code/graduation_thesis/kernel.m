function G=kernel(U,V)
% 平均核函数

% 每一个特征的长度
% feature_length=[20,80,320,1280,40,160,640,2560,300,300,300,300,298,300,298,298,1000,1000,1000,1000,991,995,995,993,28,112,448,37,148,592];
feature_length=[20,80,320,1280,40,160,640,2560,300,300,300,300,297,298,297,296,999,1000,1000,1000,990,993,992,991,28,112,448,37,148,592];

% sigma=10*[1 2 5 7 10];
sigma=100;
index=1;
G=zeros(size(U,1),size(V,1));
for i=feature_length
    r2=repmat(sum(U(:,index:index+i-1).^2,2),1,size(V,1))+repmat(sum(V(:,index:index+i-1).^2,2),1,size(U,1))'-2*U(:,index:index+i-1)*V(:,index:index+i-1)';
    for j=sigma
        G=G+exp(-r2/(2*j^2));
    end
    index=index+i;
end
G=G/(length(sigma)*length(feature_length));
end