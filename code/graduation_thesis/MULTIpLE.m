function [W,B]=MULTIpLE(W,x_train_new,A1,A2,B1,B2,b)
A=A1-[A2 A2*b];
B=B1-[B2;B2'*b];
a=[];
vector=[];
for i=1:size(W{1},2)
    if b(i)~=0
        a=[a;b(i)*W{1}{i}];
        vector=[vector;W{2}{i}];
    end

    W{1}{i}=[W{1}{i};A(:,i)];
    W{2}{i}=[W{2}{i};x_train_new];
end
a=[a;A(:,end)];
vector=[vector;x_train_new];
W{1}{end+1}=a;
W{2}{end+1}=vector;
end