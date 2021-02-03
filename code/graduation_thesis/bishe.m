% 多核学习
% 原始建立模型时有41类，迁移后模型有42类，迁移时训练样本包含42类的样本
% 原始模型采用多核、线性、单核和平均核四种方式构建
% 更新模型采用不迁移方式和基于四种原始模型进行迁移

% 导入工具箱路径
path(path,'D:\mfxs\SimpleMKL');

% 加载数据
disp('loading...');
S=load('data.mat');
disp('loading done');

% 超参数范围
% range_r=2.^(-5:8);
% range_bandwidth=1./sqrt(2*range_r);
range_bandwidth=10*[1 2 5 7 10];
range_C=(100:100:500);

global new;
for new=1
    disp('训练集和测试集的划分');
    [x_train,x_validation,x_train_new,x_test,y_train,y_validation,y_train_new,y_test,sample_train,sample_validation,sample_train_new,sample_test]...
        =split(S,new);
    train_index=generate_train_index(sample_train);
    figure_number=1;
    disp('---------------------------');
    disp('原始模型构建');
    disp('---------------------------');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 原始模型构建
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    disp('原始多核模型构建');
    [kernel_weight,alpha,bias,supportvector,Weight,InfoKernel,kernel,kerneloptionvec,variableveccell,options,C_best1,time_mkl.train,time_mkl.validation]...
        =mkl(x_train,y_train,train_index,x_validation,y_validation,sample_validation,range_bandwidth,range_C,0);
    
    opt_method='PSO';
    switch opt_method
        case 'PSO'
            disp('粒子群优化');
            tstart=tic;
            [x,f]=PSO(x_validation,y_validation,sample_validation,alpha,supportvector,bias,InfoKernel,Weight,kernel_weight);
            time_opt=toc(tstart);
        case 'GA'
            disp('遗传算法优化');
            tstart=tic;
            gaoptions=gaoptimset('PopulationSize',10,'Generations',30,'UseParallel',true,'InitialPopulation',ones(10,41)/41+0.02*(rand(10,41)-0.5),'PlotFcns',@gaplotbestf,'Display','iter');
            [x,f]=ga(@(x) target_function(x,alpha,supportvector,bias,x_validation,y_validation,sample_validation,InfoKernel,Weight,kernel_weight),41,[],[],[],[],0,1,[],gaoptions);
            f=-f;
            time_opt=toc(tstart);
    end
    Weight_mkl=x*Weight;
    kernel_weight_mkl=x*kernel_weight;

    disp('原始多核模型预测');
    tstart=tic;
    accuracy1=test({alpha,supportvector},bias,x_test,y_test,sample_test,figure_number,'common',InfoKernel,Weight_mkl,kernel_weight_mkl);
    time_mkl.test=toc(tstart);
    figure_number=figure_number+1;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     accuracy0{1}=test({alpha,supportvector},bias,x_train,y_train,sample_train,0,'respective',InfoKernel,Weight,kernel_weight);
%     accuracy0{2}=test({alpha,supportvector},bias,x_train,y_train,sample_train,0,'common',InfoKernel,Weight_mkl,kernel_weight_mkl);
%     accuracy0{3}=test({alpha,supportvector},bias,x_validation,y_validation,sample_validation,0,'respective',InfoKernel,Weight,kernel_weight);
%     accuracy0{4}=test({alpha,supportvector},bias,x_validation,y_validation,sample_validation,0,'common',InfoKernel,Weight_mkl,kernel_weight_mkl);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    disp('原始线性模型构建');
    [alpha_lin,supportvector_lin,bias_lin,C_best2,time_lin.train,time_lin.validation]...
        =skl(x_train,y_train,train_index,x_validation,y_validation,sample_validation,range_C,'linear');
    disp('原始线性模型预测');
    tstart=tic;
    accuracy2=test({alpha_lin,supportvector_lin},bias_lin,x_test,y_test,sample_test,figure_number,'linear_model');
    time_lin.test=toc(tstart);
    figure_number=figure_number+1;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    disp('原始核化模型构建');
    [alpha_skl,supportvector_skl,bias_skl,C_best3,time_skl.train,time_skl.validation]...
        =skl(x_train,y_train,train_index,x_validation,y_validation,sample_validation,range_C,'rbf');
    disp('原始核化模型预测');
    tstart=tic;
    accuracy3=test({alpha_skl,supportvector_skl},bias_skl,x_test,y_test,sample_test,figure_number,'kernel_model');
    time_skl.test=toc(tstart);
    figure_number=figure_number+1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    disp('原始平均核模型构建');
    [alpha_akl,supportvector_akl,bias_akl,C_best4,time_akl.train,time_akl.validation]...
        =akl(x_train,y_train,train_index,x_validation,y_validation,sample_validation,range_C);
    disp('原始平均核模型预测');
    tstart=tic;
    accuracy4=test({alpha_akl,supportvector_akl},bias_akl,x_test,y_test,sample_test,figure_number,'average_kernel');
    time_akl.test=toc(tstart);
    figure_number=figure_number+1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 模型迁移
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
    disp('---------------------------');
    disp('原始模型迁移');
    disp('---------------------------');
    
    new_number=[1 3 5 10];
    accuracy5=cell(1,length(new_number));
    accuracy6=cell(1,length(new_number));
    accuracy7=cell(1,length(new_number));
    accuracy8=cell(1,length(new_number));
    accuracy9=cell(1,length(new_number));
    for i=1:length(new_number)
        disp('---------------------------');
        disp(strcat('新类别样本个数为',num2str(new_number(i))));
        disp('---------------------------');
        if i~=length(new_number)
            [x_train_new1,y_train_new1,sample_train_new1]...
                =diff_new_number(x_train_new,y_train_new,new_number(i),sample_train_new);
        else
            x_train_new1=x_train_new;
            y_train_new1=y_train_new;
            sample_train_new1=sample_train_new;
        end
        
        pure='mixed';
        switch pure
            case 'old'
                [x_train_new1,y_train_new1]=pure_old(x_train_new1,y_train_new1);
            case 'new'
                [x_train_new1,y_train_new1]=pure_new(x_train_new1,y_train_new1);
        end

        disp('多核模型迁移');
        [W_mkl,B_mkl,time_mkl.transfer(i,1)]...
            =transferlearning(x_train_new1,y_train_new1,'multi_kernel',alpha,supportvector,C_best1,kernel_weight_mkl,Weight_mkl,InfoKernel);
        disp('迁移模型预测');
        tstart=tic;
        accuracy5{i}=test(W_mkl,B_mkl,x_test,y_test,sample_test,figure_number,'common',InfoKernel,Weight_mkl,kernel_weight_mkl);
        time_mkl.transfer_test(i,1)=toc(tstart);
        figure_number=figure_number+1;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        train_new_index1=generate_train_new_index(sample_train_new1);
        disp('不进行模型迁移');
        [W_ntl,B_ntl,Weight_ntl,kernel_weight_ntl,time_ntl.transfer(i,1)]...
            =notransferlearning(x_train_new1,y_train_new1,train_new_index1,kernel_weight,alpha,bias,supportvector,Weight,kernel,kerneloptionvec,variableveccell,options,C_best1);
        disp('不迁移模型预测');
        tstart=tic;
        accuracy6{i}=test(W_ntl,B_ntl,x_test,y_test,sample_test,figure_number,'respective',InfoKernel,Weight_ntl,kernel_weight_ntl);
        time_ntl.transfer_test(i,1)=toc(tstart);
        figure_number=figure_number+1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        disp('线性模型迁移');
        [W_lin,B_lin,time_lin.transfer(i,1)]...
            =transferlearning(x_train_new1,y_train_new1,'linear_model',alpha_lin,supportvector_lin,C_best2);
        disp('迁移模型预测');
        tstart=tic;
        accuracy7{i}=test(W_lin,B_lin,x_test,y_test,sample_test,figure_number,'linear_model');
        time_lin.transfer_test(i,1)=toc(tstart);
        figure_number=figure_number+1;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        disp('核化模型迁移');
        [W_skl,B_skl,time_skl.transfer(i,1)]...
            =transferlearning(x_train_new1,y_train_new1,'kernel_model',alpha_skl,supportvector_skl,C_best3);
        disp('迁移模型预测');
        tstart=tic;
        accuracy8{i}=test(W_skl,B_skl,x_test,y_test,sample_test,figure_number,'kernel_model');
        time_skl.transfer_test(i,1)=toc(tstart);
        figure_number=figure_number+1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        disp('平均核模型迁移');
        [W_akl,B_akl,time_akl.transfer(i,1)]...
            =transferlearning(x_train_new1,y_train_new1,'average_kernel',alpha_akl,supportvector_akl,C_best4);
        disp('迁移模型预测');
        tstart=tic;
        accuracy9{i}=test(W_akl,B_akl,x_test,y_test,sample_test,figure_number,'average_kernel');
        time_akl.transfer_test(i,1)=toc(tstart);
        figure_number=figure_number+1;
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 写入结果
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    disp('写入计算结果');
    xlswrite('Result.xlsx',C_best1,1,strcat('C',num2str(new*3+1)));
    xlswrite('Result.xlsx',C_best2,1,strcat('F',num2str(new*3+1)));
    xlswrite('Result.xlsx',C_best3,1,strcat('I',num2str(new*3+1)));
    xlswrite('Result.xlsx',C_best4,1,strcat('L',num2str(new*3+1)));
    
    xlswrite('Result.xlsx',100*accuracy1(:,1:2),1,strcat('D',num2str(new*3+1),':E',num2str(new*3+3)));
    xlswrite('Result.xlsx',100*accuracy2(:,1:2),1,strcat('G',num2str(new*3+1),':H',num2str(new*3+3)));
    xlswrite('Result.xlsx',100*accuracy3(:,1:2),1,strcat('J',num2str(new*3+1),':K',num2str(new*3+3)));
    xlswrite('Result.xlsx',100*accuracy4(:,1:2),1,strcat('M',num2str(new*3+1),':N',num2str(new*3+3)));
    
    xlswrite('Result.xlsx',100*[accuracy5{1} accuracy5{2} accuracy5{3} accuracy5{4} accuracy6{1} accuracy6{2} accuracy6{3} accuracy6{4} ...
        accuracy7{1} accuracy7{2} accuracy7{3} accuracy7{4} accuracy8{1} accuracy8{2} accuracy8{3} accuracy8{4} accuracy9{1} accuracy9{2} ...
        accuracy9{3} accuracy9{4}],1,strcat('O',num2str(new*3+1),':BV',num2str(new*3+3)));
    
    xlswrite('Result.xlsx',[time_mkl.train time_mkl.validation time_mkl.test time_opt],2,strcat('C',num2str(4*new),':F',num2str(4*new)));
    xlswrite('Result.xlsx',[time_lin.train time_lin.validation time_lin.test],2,strcat('K',num2str(4*new),':M',num2str(4*new)));
    xlswrite('Result.xlsx',[time_skl.train time_skl.validation time_skl.test],2,strcat('P',num2str(4*new),':R',num2str(4*new)));
    xlswrite('Result.xlsx',[time_akl.train time_akl.validation time_akl.test],2,strcat('U',num2str(4*new),':W',num2str(4*new)));
    
    xlswrite('Result.xlsx',[time_mkl.transfer time_mkl.transfer_test time_ntl.transfer time_ntl.transfer_test],2,strcat('G',num2str(4*new),':J',num2str(4*new+3)));
    xlswrite('Result.xlsx',[time_lin.transfer time_lin.transfer_test],2,strcat('N',num2str(4*new),':O',num2str(4*new+3)));
    xlswrite('Result.xlsx',[time_skl.transfer time_skl.transfer_test],2,strcat('S',num2str(4*new),':T',num2str(4*new+3)));
    xlswrite('Result.xlsx',[time_akl.transfer time_akl.transfer_test],2,strcat('X',num2str(4*new),':Y',num2str(4*new+3)));
end