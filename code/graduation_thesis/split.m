function [x_train,x_validation,x_train_new,x_test,y_train,y_validation,y_train_new,y_test,sample_train,sample_validation,sample_train_new,sample_test]=split(S,newclass)
% 训练集和测试集划分

% 计算总共的类别数
class=10;%size(S.y,2);

% 初始化类别索引元胞
sample=cell(1,class);
sample_train=cell(1,class);
sample_validation=cell(1,class);
sample_train_new=cell(1,class);
sample_test=cell(1,class);

x_phog=[S.x_phog_180_level0 S.x_phog_180_level1 S.x_phog_180_level2 S.x_phog_180_level3 S.x_phog_360_level0 S.x_phog_360_level1 S.x_phog_360_level2 S.x_phog_360_level3];
x_sift=[S.x_sift_K300_grey_r4 S.x_sift_K300_grey_r8 S.x_sift_K300_grey_r12 S.x_sift_K300_grey_r16 S.x_sift_K300_hsv_r4 S.x_sift_K300_hsv_r8 S.x_sift_K300_hsv_r12 S.x_sift_K300_hsv_r16 S.x_sift_K1000_grey_r4 S.x_sift_K1000_grey_r8 S.x_sift_K1000_grey_r12 S.x_sift_K1000_grey_r16 S.x_sift_K1000_hsv_r4 S.x_sift_K1000_hsv_r8 S.x_sift_K1000_hsv_r12 S.x_sift_K1000_hsv_r16];
x_regcov=[S.x_regcov_level0 S.x_regcov_level1 S.x_regcov_level2];
x_lbp=[S.x_lbp_level0 S.x_lbp_level1 S.x_lbp_level2];
x=[x_phog x_sift x_regcov x_lbp];

x=x(1:1423,:);
S.y=S.y(1:1423,1:10);

% % 加入噪声
% x=[x normrnd(0,1,size(x,1),50)];

% 去除无效样本
[m,~]=find(isnan(x));
x(m,:)=[];
S.y(m,:)=[];

% 去除无效特征
standard_deviation=std(x);
x(:,standard_deviation==0)=[];

% 计算总共的特征数
number_feature=size(x,2);

% 统计每一类样本的索引
for i=1:size(S.y,1)
    sample{S.y(i,:)==1}(end+1,1)=i;
end

% 划分比例
rate_train=0.7;
rate_validation=0;
number_new=10;

% 如未指定随机选择一个类别作为新类别
if nargin==0
    newclass=randperm(class,1);
end

% 生成类别名
global name;
name=cell(1,class);
name(1:7)={'backpack','baseball-bat','baseball-glove','bathtub','beer-mug','binoculars','bonsai'};
name(8:14)={'boom-box','breadmaker','calculator','cd','cereal-box','chandelier','chess-board'};
name(15:21)={'chopsticks','coffee-mug','computer-keyboard','computer-monitor','computer-mouse','cowboy-hat','desk-globe'};
name(22:28)={'doorknob','dumb-bell','eyeglasses','flashlight','frying-pan','head-phones','knife'};
name(29:35)={'ladder','laptop','lightbulb','mattress','microwave','refrigerator','rotary-phone'};
name(36:42)={'sneaker','spoon','toaster','top-hat','t-shirt','washing-machine','wine-bottle'};
name{end+1}=name{newclass};
name(newclass)=[];

number_train=0;
number_validation=0;
number_train_new=0;
number_test=0;

% 计算训练和测试样本的索引
for i=1:class
    number=length(sample{i});
    index=randperm(number);
    if i~=newclass
        sample_train{i}=sample{i}(index(1:round(rate_train*number)));
        sample_validation{i}=sample{i}(index(round(rate_train*number)+1:round((rate_train+rate_validation)*number)));
        sample_train_new{i}=sample{i}(index(round((rate_train+rate_validation)*number)+1:round((rate_train+rate_validation)*number)+number_new));
        sample_test{i}=sample{i}(index(round((rate_train+rate_validation)*number)+number_new+1:end));
    else
        sample_train_new{i}=sample{i}(index(1:number_new));
        sample_test{i}=sample{i}(index(number_new+1:end));
    end
    number_train=number_train+length(sample_train{i});
    number_validation=number_validation+length(sample_validation{i});
    number_train_new=number_train_new+length(sample_train_new{i});
    number_test=number_test+length(sample_test{i});
end

x_train=zeros(number_train,number_feature);
x_validation=zeros(number_validation,number_feature);
x_train_new=zeros(number_train_new,number_feature);
x_test=zeros(number_test,number_feature);

y_train=zeros(number_train,class);
y_validation=zeros(number_validation,class);
y_train_new=zeros(number_train_new,class);
y_test=zeros(number_test,class);

index_train=1;
index_validation=1;
index_train_new=1;
index_test=1;

% 生成训练和测试数据
for i=1:class
    x_train(index_train:index_train-1+length(sample_train{i}),:)=x(sample_train{i},:);
    x_validation(index_validation:index_validation-1+length(sample_validation{i}),:)=x(sample_validation{i},:);
    x_train_new(index_train_new:index_train_new-1+length(sample_train_new{i}),:)=x(sample_train_new{i},:);
    x_test(index_test:index_test-1+length(sample_test{i}),:)=x(sample_test{i},:);
    
    y_train(index_train:index_train-1+length(sample_train{i}),:)=S.y(sample_train{i},:);
    y_validation(index_validation:index_validation-1+length(sample_validation{i}),:)=S.y(sample_validation{i},:);
    y_train_new(index_train_new:index_train_new-1+length(sample_train_new{i}),:)=S.y(sample_train_new{i},:);
    y_test(index_test:index_test-1+length(sample_test{i}),:)=S.y(sample_test{i},:);
    
    index_train=index_train+length(sample_train{i});
    index_validation=index_validation+length(sample_validation{i});
    index_train_new=index_train_new+length(sample_train_new{i});
    index_test=index_test+length(sample_test{i});
end

% 数据标准化
[~,mean,sigma]=zscore([x_train;x_validation]);
sigma(sigma==0)=1;
x_train=(x_train-repmat(mean,number_train,1))./repmat(sigma,number_train,1);
x_validation=(x_validation-repmat(mean,number_validation,1))./repmat(sigma,number_validation,1);
x_train_new=(x_train_new-repmat(mean,number_train_new,1))./repmat(sigma,number_train_new,1);
x_test=(x_test-repmat(mean,number_test,1))./repmat(sigma,number_test,1);

% 更新训练和测试数据中每一类的索引
sample_train=cell(1,class);
sample_validation=cell(1,class);
sample_train_new=cell(1,class);
sample_test=cell(1,class);

for i=1:size(y_train,1)
    sample_train{y_train(i,:)==1}(end+1,1)=i;
end

for i=1:size(y_validation,1)
    sample_validation{y_validation(i,:)==1}(end+1,1)=i;
end

for i=1:size(y_train_new,1)
    sample_train_new{y_train_new(i,:)==1}(end+1,1)=i;
end

for i=1:size(y_test,1)
    sample_test{y_test(i,:)==1}(end+1,1)=i;
end

% 除去训练数据中的新类别标签
y_train(:,newclass)=[];
y_validation(:,newclass)=[];
sample_train(newclass)=[];
sample_validation(newclass)=[];

% 将新训练数据和测试数据中的新类别标签移至最后
y_train_new(:,end+1)=y_train_new(:,newclass);
y_test(:,end+1)=y_test(:,newclass);
y_train_new(:,newclass)=[];
y_test(:,newclass)=[];
sample_train_new{end+1}=sample_train_new{newclass};
sample_train_new(newclass)=[];
sample_test{end+1}=sample_test{newclass};
sample_test(newclass)=[];
end