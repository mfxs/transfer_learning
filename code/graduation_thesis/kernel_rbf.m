function G=kernel_rbf(U,V)
% RBFºËº¯Êý

sigma=100;
r2=repmat(sum(U.^2,2),1,size(V,1))+repmat(sum(V.^2,2),1,size(U,1))'-2*U*V';
G=exp(-r2/(2*sigma^2));
end