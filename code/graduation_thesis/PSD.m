function b=PSD(M,A1,A2,T,y_train_new)
m=size(M,1)-1;
n=size(y_train_new,2)-1;
b=zeros(n,1);
for t=1:T
    Y_=y_train_new-(M(1:m,1:m).*eye(m))^-1*(A1-[A2 A2*b]);
    delta=zeros(n,1);
    for i=1:m
        if y_train_new(i,n+1)~=1
            if 1+Y_(i,n+1)-Y_(i,y_train_new(i,:)==1)>0
                delta=delta+A2(i,:)'/M(i,i);
            end
        elseif maxryi(Y_,i)>0
            delta=delta-A2(i,:)'/M(i,i);
        end
    end
    b=b-delta/(m*sqrt(t));
    for j=1:n
        if b(j)<0
            b(j)=0;
        end
    end
    if sqrt(b'*b)>1
        b=b/sqrt(b'*b);
    end
end
end