function max=maxryi(Y_,i)

n=size(Y_,2)-1;
max=0;
for r=1:n
    if 1+Y_(i,r)-Y_(i,n+1)>max
        max=1+Y_(i,r)-Y_(i,n+1);
    end
end
end