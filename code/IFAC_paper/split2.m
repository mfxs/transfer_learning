function [x_train,y_train,x_val,y_val,x_train_new,y_train_new,x_test,y_test]=split2(S)
% 训练集和测试集划分

% 选择旧类别和新类别以及迁移顺序
old_class=[1 3 4 6 9 18 27 29 32 40];
new_class=[2 7 30];
order=randperm(length(new_class));

% 生成类别名
global name;
name=cell(1,42);
name(1:7)={'backpack','baseball-bat','baseball-glove','bathtub','beer-mug','binoculars','bonsai'};
name(8:14)={'boom-box','breadmaker','calculator','cd','cereal-box','chandelier','chess-board'};
name(15:21)={'chopsticks','coffee-mug','computer-keyboard','computer-monitor','computer-mouse','cowboy-hat','desk-globe'};
name(22:28)={'doorknob','dumb-bell','eyeglasses','flashlight','frying-pan','head-phones','knife'};
name(29:35)={'ladder','laptop','lightbulb','mattress','microwave','refrigerator','rotary-phone'};
name(36:42)={'sneaker','spoon','toaster','top-hat','t-shirt','washing-machine','wine-bottle'};
disp('Old Class:');
disp(name(old_class));
disp('New Class:');
disp(name(new_class(order)));

% 将不同特征矩阵连接
x_phog=[S.x_phog_180_level0 S.x_phog_180_level1 S.x_phog_180_level2 S.x_phog_180_level3 S.x_phog_360_level0 S.x_phog_360_level1 S.x_phog_360_level2 S.x_phog_360_level3];
x_sift=[S.x_sift_K300_grey_r4 S.x_sift_K300_grey_r8 S.x_sift_K300_grey_r12 S.x_sift_K300_grey_r16 S.x_sift_K300_hsv_r4 S.x_sift_K300_hsv_r8 S.x_sift_K300_hsv_r12 S.x_sift_K300_hsv_r16 S.x_sift_K1000_grey_r4 S.x_sift_K1000_grey_r8 S.x_sift_K1000_grey_r12 S.x_sift_K1000_grey_r16 S.x_sift_K1000_hsv_r4 S.x_sift_K1000_hsv_r8 S.x_sift_K1000_hsv_r12 S.x_sift_K1000_hsv_r16];
x_regcov=[S.x_regcov_level0 S.x_regcov_level1 S.x_regcov_level2];
x_lbp=[S.x_lbp_level0 S.x_lbp_level1 S.x_lbp_level2];
x=[x_phog x_sift x_regcov x_lbp];
y=S.y;

% 选择使用特征
phog=(1:8);
sift=(9:24);
regcov=(25:27);
lbp=(28:30);
chosen_feature=[phog regcov lbp];
global kernel_length;
kernel_length=[20,80,320,1280,40,160,640,2560,300,300,300,300,300,300,300,300,1000,1000,1000,1000,1000,1000,1000,1000,28,112,448,37,148,592];
for i=length(kernel_length):-1:1
    if ~ismember(i,chosen_feature)
        x(:,sum(kernel_length(1:i-1))+1:sum(kernel_length(1:i)))=[];
        kernel_length(i)=[];
    end
end

% 去除无效样本
[index_nan,~]=find(isnan(x));
x(index_nan,:)=[];
y(index_nan,:)=[];

% 选择指定类别的数据
x_old=x(ismember(label_transform(S.y),old_class),:);
y_old=S.y(ismember(label_transform(S.y),old_class),old_class);
x_new=x(ismember(label_transform(S.y),new_class),:);
y_new=S.y(ismember(label_transform(S.y),new_class),new_class);

% 去除无效特征
minus_kernel_length=zeros(size(kernel_length));
standard_deviation=std(x_old);
for i=find(standard_deviation==0)
    position=0;
    for j=1:length(kernel_length)
        if i>position && i<=position+kernel_length(j)
            minus_kernel_length(j)=minus_kernel_length(j)+1;
            break;
        end
        position=position+kernel_length(j);
    end
end
kernel_length=kernel_length-minus_kernel_length;
kernel_length(kernel_length==0)=[];
x_old(:,standard_deviation==0)=[];
x_new(:,standard_deviation==0)=[];

% 划分比例
num_train=80;
num_val=20;
num_test=20;
num_new_train=4;

% 初始化各部分
x_train=[];
y_train=[];
x_val=[];
y_val=[];
x_train_new=cell(1,length(new_class));
y_train_new=cell(1,length(new_class));
x_test=[];
y_test=[];

% 划分旧类别数据
for i=1:length(old_class)
    index_sample=find(y_old(:,i)==1);
    shuffle=randperm(length(index_sample));
    x_train=[x_train;x_old(index_sample(shuffle(1:num_train)),:)];
    y_train=[y_train;y_old(index_sample(shuffle(1:num_train)),:)];
    x_val=[x_val;x_old(index_sample(shuffle(num_train+1:num_train+num_val)),:)];
    y_val=[y_val;y_old(index_sample(shuffle(num_train+1:num_train+num_val)),:)];
    for j=1:length(new_class)
        x_train_new{j}=[x_train_new{j};x_old(index_sample(shuffle(num_train+num_val+(j-1)*num_new_train+1:num_train+num_val+j*num_new_train)),:)];
        y_train_new{j}=[y_train_new{j};y_old(index_sample(shuffle(num_train+num_val+(j-1)*num_new_train+1:num_train+num_val+j*num_new_train)),:)];
    end
   x_test=[x_test;x_old(index_sample(shuffle(num_train+num_val+length(new_class)*num_new_train+1:num_train+num_val+length(new_class)*num_new_train+num_test)),:)];
   y_test=[y_test;y_old(index_sample(shuffle(num_train+num_val+length(new_class)*num_new_train+1:num_train+num_val+length(new_class)*num_new_train+num_test)),:)];
end

% 划分新类别数据
for i=1:length(new_class)
    index_sample=find(y_new(:,order(i))==1);
    shuffle=randperm(length(index_sample));
    for j=i:length(new_class)
        x_train_new{j}=[x_train_new{j};x_new(index_sample(shuffle((j-1)*num_new_train+1:j*num_new_train)),:)];
        y_train_new{j}=[y_train_new{j} -ones(size(y_train_new{j},1),1);-ones(num_new_train,size(y_train_new{j},2)) ones(num_new_train,1)];
    end
    x_test=[x_test;x_new(index_sample(shuffle(i*num_new_train+1:i*num_new_train+num_test)),:)];
    y_test=[y_test -ones(size(y_test,1),1);-ones(num_test,size(y_test,2)) ones(num_test,1)];
end

% 数据标准化
[~,mean,sigma]=zscore([x_train;x_val]);
sigma(sigma==0)=1;
x_train=(x_train-repmat(mean,size(x_train,1),1))./repmat(sigma,size(x_train,1),1);
x_val=(x_val-repmat(mean,size(x_val,1),1))./repmat(sigma,size(x_val,1),1);
for i=1:length(new_class)
    x_train_new{i}=(x_train_new{i}-repmat(mean,size(x_train_new{i},1),1))./repmat(sigma,size(x_train_new{i},1),1);
end
x_test=(x_test-repmat(mean,size(x_test,1),1))./repmat(sigma,size(x_test,1),1);
end