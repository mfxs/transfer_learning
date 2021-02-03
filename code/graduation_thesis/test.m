function accuracy=test(W,B,x_test,y_test,sample_test,figure_number,s,InfoKernel,Weight,kernel_weight)
% Ԥ����

% ���������������Ԥ�����
class=size(y_test,2);
y_pred=-1*ones(size(y_test));

% ��Բ�ͬʵ��������ò�ͬ���㷽��
for i=1:length(B)
    switch s
        case 'common'
            Kt=mklkernel(x_test,InfoKernel,Weight,W{2}{i},kernel_weight);
        case 'respective'
            Kt=mklkernel(x_test,InfoKernel,Weight(i,:),W{2}{i},kernel_weight(i,:));
        case 'average_kernel'
            Kt=kernel(x_test,W{2}{i});
        case 'linear_model'
            Kt=x_test*W{2}{i}';
        case 'kernel_model'
            Kt=kernel_rbf(x_test,W{2}{i});
    end
    y_pred(:,i)=Kt*W{1}{i}+B(i);
%     b=max(y_pred(:,i));
%     a=min(y_pred(:,i));
%     y_pred(:,i)=2*(y_pred(:,i)-a)/(b-a)-1;
end

% ͳ�Ʋ���������ÿһ�����Ŀ
test_number=zeros(1,class);
for i=1:class
    test_number(i)=length(sample_test{i});
end

% ����Top-1,Top-3��Top-5׼ȷ��
top_number1=1;
top_number2=3;
top_number3=5;
correct1=zeros(1,class);
correct2=zeros(1,class);
correct3=zeros(1,class);
[~,label_real]=max(y_test,[],2);
for i=1:size(y_pred,1)
    [~,pos]=sort(y_pred(i,:));
    if ismember(label_real(i),pos(end+1-top_number1:end))==1
        correct1(label_real(i))=correct1(label_real(i))+1;
    end
    if ismember(label_real(i),pos(end+1-top_number2:end))==1
        correct2(label_real(i))=correct2(label_real(i))+1;
    end
    if ismember(label_real(i),pos(end+1-top_number3:end))==1
        correct3(label_real(i))=correct3(label_real(i))+1;
    end
end

% Top-1
% ����׼ȷ��
accuracy(1,1)=sum(correct1)/size(y_pred,1);
% ԭ����׼ȷ��
accuracy(1,2)=(sum(correct1)-correct1(end))/(size(y_pred,1)-length(sample_test{end}));
% ����׼ȷ��
accuracy(1,3)=correct1(end)/length(sample_test{end});

% Top-3
% ����׼ȷ��
accuracy(2,1)=sum(correct2)/size(y_pred,1);
% ԭ����׼ȷ��
accuracy(2,2)=(sum(correct2)-correct2(end))/(size(y_pred,1)-length(sample_test{end}));
% ����׼ȷ��
accuracy(2,3)=correct2(end)/length(sample_test{end});

% Top-5
% ����׼ȷ��
accuracy(3,1)=sum(correct3)/size(y_pred,1);
% ԭ����׼ȷ��
accuracy(3,2)=(sum(correct3)-correct3(end))/(size(y_pred,1)-length(sample_test{end}));
% ����׼ȷ��
accuracy(3,3)=correct3(end)/length(sample_test{end});

% ��ͼ
global name;
global new;
if figure_number~=0
%     Top-1
    figure(figure_number);
    set(figure_number,'outerposition',get(0,'screensize'));
    barh(test_number);
    hold on;
    barh(correct1,'r');
    for i=1:length(test_number)
        text(test_number(i)+0.5,i,strcat(name{i},',',num2str(correct1(i)),'/',num2str(test_number(i)),',',num2str(100*correct1(i)/test_number(i)),'%'));
    end
    xlabel('������');
    ylabel('����');
    legend('Ԥ��������','Ԥ����ȷ��','location','southeast');
    grid on
    hold off
    saveas(figure_number,strcat('ͼ/',num2str(new),'/',num2str(figure_number),'-top1.jpg'));

%     Top-3
    figure(figure_number);
    set(figure_number,'outerposition',get(0,'screensize'));
    barh(test_number);
    hold on;
    barh(correct2,'r');
    for i=1:length(test_number)
        text(test_number(i)+0.5,i,strcat(name{i},',',num2str(correct2(i)),'/',num2str(test_number(i)),',',num2str(100*correct2(i)/test_number(i)),'%'));
    end
    xlabel('������');
    ylabel('����');
    legend('Ԥ��������','Ԥ����ȷ��','location','southeast');
    grid on
    hold off
    saveas(figure_number,strcat('ͼ/',num2str(new),'/',num2str(figure_number),'-top3.jpg'));

%     Top-5
    figure(figure_number);
    set(figure_number,'outerposition',get(0,'screensize'));
    barh(test_number);
    hold on;
    barh(correct3,'r');
    for i=1:length(test_number)
        text(test_number(i)+0.5,i,strcat(name{i},',',num2str(correct3(i)),'/',num2str(test_number(i)),',',num2str(100*correct3(i)/test_number(i)),'%'));
    end
    xlabel('������');
    ylabel('����');
    legend('Ԥ��������','Ԥ����ȷ��','location','southeast');
    grid on
    hold off
    saveas(figure_number,strcat('ͼ/',num2str(new),'/',num2str(figure_number),'-top5.jpg'));
end
end