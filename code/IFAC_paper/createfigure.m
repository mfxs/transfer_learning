function createfigure(X1, YMatrix1)
%CREATEFIGURE1(X1, YMatrix1)
%  X1:  x 数据的向量
%  YMATRIX1:  y 数据的矩阵

%  由 MATLAB 于 19-Oct-2019 15:45:16 自动生成

% 创建 figure
figure1 = figure('PaperSize',[20.99999864 29.69999902],'Color',[1 1 1]);

% 创建 axes
axes1 = axes('Parent',figure1,...
    'Position',[0.132277904328018 0.11796812749004 0.775 0.815]);
hold(axes1,'on');

% 使用 plot 的矩阵输入创建多行
YMatrix1=reshape(YMatrix1,4,6);
plot1 = plot(X1,YMatrix1,'MarkerSize',10,'LineWidth',3,'Parent',axes1);
set(plot1(1),'DisplayName','MULTIpLE-MKL',...
    'MarkerFaceColor',[0 0.447058826684952 0.74117648601532],...
    'Marker','o',...
    'Color',[0 0.447058826684952 0.74117648601532]);
set(plot1(2),'DisplayName','Linear transfer model',...
    'MarkerFaceColor',[0.854901969432831 0.701960802078247 1],...
    'Marker','square',...
    'Color',[0.854901969432831 0.701960802078247 1]);
set(plot1(3),'DisplayName','RBF kernel transfer model',...
    'MarkerFaceColor',[0.466666668653488 0.674509823322296 0.18823529779911],...
    'MarkerSize',8,...
    'Marker','diamond',...
    'Color',[0.466666668653488 0.674509823322296 0.18823529779911]);
set(plot1(4),'DisplayName','Average kernel transfer model',...
    'MarkerFaceColor',[0.87058824300766 0.490196079015732 0],...
    'Marker','^',...
    'Color',[0.87058824300766 0.490196079015732 0]);
set(plot1(5),'DisplayName','Reconstruct model','Marker','pentagram',...
    'Color',[0 0 0]);
set(plot1(6),'DisplayName','Non-transfer model',...
    'MarkerFaceColor',[0.301960796117783 0.745098054409027 0.933333337306976],...
    'Marker','hexagram',...
    'Color',[0.301960796117783 0.745098054409027 0.933333337306976]);

% 创建 ylabel
ylabel('Accuracy/%','BackgroundColor',[1 1 1],'LineStyle','none',...
    'EdgeColor',[1 1 1],...
    'HorizontalAlignment','center',...
    'FontWeight','bold',...
    'FontSize',16,...
    'FontName','微软雅黑');

% 创建 xlabel
xlabel('Number of new training samples per class','FontWeight','bold',...
    'FontSize',16,...
    'FontName','微软雅黑');

% 取消以下行的注释以保留坐标区的 X 范围
% xlim(axes1,[1 10]);
% 设置其余坐标区属性
set(axes1,'FontSize',14,'FontWeight','bold','XGrid','on','YGrid','on');
% 创建 legend
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.144253672081104 0.728291665067213 0.313725482708976 0.173332589752559],...
    'FontSize',10,...
    'FontName','微软雅黑');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.136535307517084 0.239353891336268 0.0411412300683371 0.0323054331864907],...
    'String',{'7.00'},...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.855214123006833 0.594713656387661 0.041141230068337 0.0323054331864908],...
    'String','23.00',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.467970387243735 0.466960352422903 0.0411412300683369 0.0323054331864907],...
    'String','16.50',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.137208070927687 0.11160796641061 0.0411412300683371 0.0323054331864907],...
    'String','0.50',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.247502188574746 0.382964750330208 0.0411412300683372 0.0323054331864908],...
    'String','13.00',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.474217874849256 0.415628066913123 0.0411412300683373 0.0323054331864908],...
    'String','14.50',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.85411983563357 0.415628066913123 0.0411412300683373 0.0323054331864908],...
    'String','14.50',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.305100227790432 0.28371851917443 0.041141230068337 0.0323054331864908],...
    'String','9.50',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.481570816025727 0.352813996561364 0.0411412300683372 0.0323054331864908],...
    'String','13.00',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.85411983563357 0.447035102089002 0.0411412300683371 0.0323054331864908],...
    'String','18.00',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.145786502300237 0.134221031737244 0.0411412300683372 0.0323054331864908],...
    'String','1.50',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.311227678770825 0.219648167415636 0.0411412300683371 0.0323054331864908],...
    'String','6.00',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.414168855241413 0.377939624702068 0.0411412300683371 0.0323054331864908],...
    'String','13.50',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.845541404261021 0.491004951335234 0.0411412300683371 0.0323054331864907],...
    'String','18.50',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.855345325829648 0.871658217666892 0.0411412300683373 0.0323054331864907],...
    'String','40.00',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.484021796417884 0.611607966410611 0.0411412300683372 0.0323054331864909],...
    'String','25.50',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.139659051319844 0.173165755355334 0.0411412300683372 0.0323054331864908],...
    'String','3.50',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.12740414935906 0.32266324279252 0.0411412300683371 0.0323054331864908],...
    'String','10.00',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.307551208182589 0.345276308119153 0.0411412300683371 0.0323054331864907],...
    'String','12.00',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.295296306221805 0.322663242792521 0.0411412300683371 0.0323054331864908],...
    'String','10.00',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.477894345437491 0.301306458872922 0.0411412300683372 0.0323054331864908],...
    'String','10.50',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

% 创建 textbox
annotation(figure1,'textbox',...
    [0.85411983563357 0.35658284078247 0.0411412300683371 0.0323054331864908],...
    'String','13.50',...
    'LineStyle','none',...
    'FontWeight','bold',...
    'FitBoxToText','off',...
    'EdgeColor','none');

