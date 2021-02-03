function G=rbf_kernel(U,V,bandwidth)
% RBFºËº¯Êý

G=0;
r2=repmat(sum(U.^2,2),1,size(V,1))+repmat(sum(V.^2,2),1,size(U,1))'-2*U*V';
for i=bandwidth
    G=G+exp(-r2/(2*i^2));
end
end