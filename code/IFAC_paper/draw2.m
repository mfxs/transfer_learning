%% �������ݺ�colormap
load('mycolormap.mat');
load('class1.mat','y_pred*','y_test','new_number');
%% ����ԭʼģ�͵Ļ�������
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
%% ���ƶ��ģ�͵Ļ�������
figure(2)
suptitle('Multiple Kernel Transfer Model');
colormap(mycolormap);
for i=1:4
    subplot(2,2,i)
    confused_matrix(y_pred{i,1},y_test);
    title(['Confused matrix of multiple kernel transfer learning with ',num2str(new_number(i)),' new samples']);
    colorbar;
end
%% ��������ģ�͵Ļ�������
figure(3)
suptitle('Linear Transfer Model');
colormap(mycolormap);
for i=1:4
    subplot(2,2,i)
    confused_matrix(y_pred{i,2},y_test);
    title(['Confused matrix of linear transfer learning with ',num2str(new_number(i)),' new samples']);
    colorbar;
end
%% ���Ƹ�˹�˺���ģ�͵Ļ�������
figure(4)
suptitle('RBF Kernel Transfer Model');
colormap(mycolormap);
for i=1:4
    subplot(2,2,i)
    confused_matrix(y_pred{i,3},y_test);
    title(['Confused matrix of rbf kernel transfer learning with ',num2str(new_number(i)),' new samples']);
    colorbar;
end
%% ����ƽ���˺���ģ�͵Ļ�������
figure(5)
suptitle('Average Kernel Transfer Model');
colormap(mycolormap);
for i=1:4
    subplot(2,2,i)
    confused_matrix(y_pred{i,4},y_test);
    title(['Confused matrix of average kernel transfer learning with ',num2str(new_number(i)),' new samples']);
    colorbar;
end
%% �������¹���ģ�͵Ļ�������
figure(6)
suptitle('Reconstruct Model');
colormap(mycolormap);
for i=1:4
    subplot(2,2,i)
    confused_matrix(y_pred{i,5},y_test);
    title(['Confused matrix of reconstruct model with ',num2str(new_number(i)),' new samples']);
    colorbar;
end
%% ���Ʋ�Ǩ��ģ�͵Ļ�������
figure(7)
suptitle('No Transfer Model');
colormap(mycolormap);
for i=1:4
    subplot(2,2,i)
    confused_matrix(y_pred{i,6},y_test);
    title(['Confused matrix of no transfer model with ',num2str(new_number(i)),' new samples']);
    colorbar;
end