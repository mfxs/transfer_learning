function createfigure3(correct)
figure();
for i=1:4
    subplot(2,2,i);
    hold on
    bar(correct(i,:),'facecolor',[0.929411768913269 0.694117665290833 0.125490203499794],'edgecolor','none');
    bar(correct(i,1:end-1),'facecolor',[0.635294139385223 0.0784313753247261 0.184313729405403],'edgecolor','none');
    bar(correct(i,1:end-2),'facecolor',[0 0.447058826684952 0.74117648601532],'edgecolor','none');
    bar(correct(i,1:end-3),'facecolor',[0 0.498039215803146 0],'edgecolor','none');
    plot([0 14],[20 20],'color',[0.635294139385223 0.0784313753247261 0.184313729405403],'linewidth',2);
    grid on;
    axis([0 14 0 21]);
    set(gca,'fontSize',16,'fontweight','bold');
    xlabel('Class','fontsize',20,'fontweight','bold');
    ylabel('Correct Number','fontsize',20,'fontweight','bold');
    for j=1:i+9
        text(j,correct(i,j)+0.5,num2str(correct(i,j),'%.1f'),'fontweight','bold','horizontalalignment','center');
    end
    if i==1
        title('10-class classifier','fontsize',24,'fontweight','bold');
    else
        title(['10+',num2str(i-1),'-class classifier'],'fontsize',24,'fontweight','bold');
    end
end
end