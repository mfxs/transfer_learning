% 数据导入

path='F:\';

tic
x_lbp_level0=xlsread(strcat(path,'x_lbp_level0.csv'));
x_lbp_level1=xlsread(strcat(path,'x_lbp_level1.csv'));
x_lbp_level2=xlsread(strcat(path,'x_lbp_level2.csv'));
toc

tic
x_regcov_level0=xlsread(strcat(path,'x_regcov_level0.csv'));
x_regcov_level1=xlsread(strcat(path,'x_regcov_level1.csv'));
x_regcov_level2=xlsread(strcat(path,'x_regcov_level2.csv'));
toc

tic
x_phog_180_level0=xlsread(strcat(path,'x_phog_180_level0.csv'));
x_phog_180_level1=xlsread(strcat(path,'x_phog_180_level1.csv'));
x_phog_180_level2=xlsread(strcat(path,'x_phog_180_level2.csv'));
x_phog_180_level3=xlsread(strcat(path,'x_phog_180_level3.csv'));

x_phog_360_level0=xlsread(strcat(path,'x_phog_360_level0.csv'));
x_phog_360_level1=xlsread(strcat(path,'x_phog_360_level1.csv'));
x_phog_360_level2=xlsread(strcat(path,'x_phog_360_level2.csv'));
x_phog_360_level3=xlsread(strcat(path,'x_phog_360_level3.csv'));
toc

tic
x_sift_K300_grey_r4=xlsread(strcat(path,'x_sift_K300_grey_r4.xlsx'));
x_sift_K300_grey_r8=xlsread(strcat(path,'x_sift_K300_grey_r8.xlsx'));
x_sift_K300_grey_r12=xlsread(strcat(path,'x_sift_K300_grey_r12.xlsx'));
x_sift_K300_grey_r16=xlsread(strcat(path,'x_sift_K300_grey_r16.xlsx'));

x_sift_K300_hsv_r4=xlsread(strcat(path,'x_sift_K300_hsv_r4.xlsx'));
x_sift_K300_hsv_r8=xlsread(strcat(path,'x_sift_K300_hsv_r8.xlsx'));
x_sift_K300_hsv_r12=xlsread(strcat(path,'x_sift_K300_hsv_r12.xlsx'));
x_sift_K300_hsv_r16=xlsread(strcat(path,'x_sift_K300_hsv_r16.xlsx'));

x_sift_K1000_grey_r4=xlsread(strcat(path,'x_sift_K1000_grey_r4.xlsx'));
x_sift_K1000_grey_r8=xlsread(strcat(path,'x_sift_K1000_grey_r8.xlsx'));
x_sift_K1000_grey_r12=xlsread(strcat(path,'x_sift_K1000_grey_r12.xlsx'));
x_sift_K1000_grey_r16=xlsread(strcat(path,'x_sift_K1000_grey_r16.xlsx'));

x_sift_K1000_hsv_r4=xlsread(strcat(path,'x_sift_K1000_hsv_r4.xlsx'));
x_sift_K1000_hsv_r8=xlsread(strcat(path,'x_sift_K1000_hsv_r8.xlsx'));
x_sift_K1000_hsv_r12=xlsread(strcat(path,'x_sift_K1000_hsv_r12.xlsx'));
x_sift_K1000_hsv_r16=xlsread(strcat(path,'x_sift_K1000_hsv_r16.xlsx'));
toc

y=xlsread(strcat(path,'y.csv'));