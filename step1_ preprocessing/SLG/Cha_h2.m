data = csvread('E:/Lu/ruxianai/ruxianai_data/spot_location.csv',1,1);
B=zeros(3789,3789);
C=zeros(3789,3789);
for i = 1:3789
    for j = 1:3789
       %eval(['B(',num2str(i),',',num2str(j),') = corr2','(a',num2str(i),',a',num2str(j),');']);
       eval(['B(',num2str(i),',',num2str(j),') =data(',num2str(i),',1)-data(',num2str(j),',1);']);
       eval(['C(',num2str(i),',',num2str(j),') =data(',num2str(i),',2)-data(',num2str(j),',2);']);
    end
end
B=abs(B);
C=abs(C);
csvwrite('row1_ruxianai.csv',B)
csvwrite('row2_ruxianai.csv',C)
%bij=data(i,1)-data(j,1)
 %eval(['B(',num2str(i),',',num2str(j),') =data(',num2str(i),',1)-data(',num2str(j),',1);']);