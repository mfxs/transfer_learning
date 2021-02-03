%% 导入数据和colormap
load('mycolormap.mat');
load('class1.mat','y_pred*','y_test','new_number');
%% 绘制原始模型的混淆矩阵
figure(1);
suptitle('Original Model');
colormap(mycolormap);
subplot(221)
confused_matrix(y_pred_mkl,y_test);
title('Confused matrix of multiple kernel transfer learning');
colorbar;
subplot(222)
confused_matrix(y_pred_linear,y_test);
title('Confused matrix of linear transfer learning');
colorbar;
subplot(223)
confused_matrix(y_pred_rbf,y_test);
title('Confused matrix of rbf kernel transfer learning');
colorbar;
subplot(224)
confused_matrix(y_pred_akl,y_test);
title('Confused matrix of average kernel transfer learning');
colorbar;
%% 绘制多核模型的混淆矩阵
figure(2)
suptitle('Multiple Kernel Transfer Model');
colormap(mycolormap);
for i=1:4
    subplot(2,2,i)
    confused_matrix(y_pred{i,1},y_test);
    title(['Confused matrix of multiple kernel transfer learning with ',num2str(new_number(i)),' new samples']);
    colorbar;
end
%% 绘制线性模型的混淆矩阵
figure(3)
suptitle('Linear Transfer Model');
colormap(mycolormap);
for i=1:4
    subplot(2,2,i)
    confused_matrix(y_pred{i,2},y_test);
    title(['Confused matrix of linear transfer learning with ',num2str(new_number(i)),' new samples']);
    colorbar;
end
%% 绘制高斯核函数模型的混淆矩阵
figure(4)
suptitle('RBF Kernel Transfer Model');
colormap(mycolormap);
for i=1:4
    subplot(2,2,i)
    confused_matrix(y_pred{i,3},y_test);
    title(['Confused matrix of rbf kernel transfer learning with ',num2str(new_number(i)),' new samples']);
    colorbar;
end
%% 绘制平均核函数模型的混淆矩阵
figure(5)
suptitle('Average Kernel Transfer Model');
colormap(mycolormap);
for i=1:4
    subplot(2,2,i)
    confused_matrix(y_pred{i,4},y_test);
    title(['Confused matrix of average kernel transfer learning with ',num2str(new_number(i)),' new samples']);
    colorbar;
end
%% 绘制重新构建模型的混淆矩阵
figure(6)
suptitle('Reconstruct Model');
colormap(mycolormap);
for i=1:4
    subplot(2,2,i)
    confused_matrix(y_pred{i,5},y_test);
    title(['Confused matrix of reconstruct model with ',num2str(new_number(i)),' new samples']);
    colorbar;
end
%% 绘制不迁移模型的混淆矩阵
figure(7)
suptitle('No Transfer Model');
colormap(mycolormap);
for i=1:4
    subplot(2,2,i)
    confused_matrix(y_pred{i,6},y_test);
    title(['Confused matrix of no transfer model with ',num2str(new_number(i)),' new samples']);
    colorbar;
end