% 绘制准确率随训练集大小的变化曲线
train_number=[1 3 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80];
for new_number=train_number
    [x_train_,y_train_]=diff_new_number(x_train,y_train,new_number);
    [alpha_linear,sv_linear,bias_linear,C2,hist_linear]=train_linear(x_train_,y_train_,x_validation,y_validation,C);
    [~,temp]=pred(x_test,y_test,alpha_linear,bias_linear,sv_linear,'linear');
    accuracy(new_number==train_number)=temp(2);
end
stem(train_number,accuracy);