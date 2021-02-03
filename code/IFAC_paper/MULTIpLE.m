function W=MULTIpLE(alpha,sv,x_train_new,A1,A2,B1,B2,b)

A=A1-[A2 A2*b];
B=B1-[B2;B2'*b];
a=[];
vector=[];
for i=1:size(alpha,2)
    if b(i)~=0
        a=[a;b(i)*alpha{i}];
        vector=[vector;sv{i}];
    end

    alpha{i}=[alpha{i};A(:,i)];
    sv{i}=[sv{i};x_train_new];
end
a=[a;A(:,end)];
vector=[vector;x_train_new];
alpha{end+1}=a;
sv{end+1}=vector;
W={alpha,sv,B};
end