%% 通过例子来学习 怎么迭代计算状态值函数 state value function 
clc
clear
close

size = 4
values = zeros(size,size); % 状态值函数
p = 0.25;
k = 0;
r_t = 0
r = -1

steps = 4;
for step = 1:steps
    step
    value =values
    for i = 1:size
        for j = 1:size
            if i == 1
                west = values(i,j);
            else
                west = values(i-1,j);
            end
            if i == size
                east = values(i,j);
            else
                east = values(i+1,j);
            end

            if j == 1
                north = values(i,j);
            else
                north = values(i,j-1);
            end
            if j == size
                south = values(i,j);
            else
                south = values(i,j+1);
            end

            if ((i == 1) &&(j==1))||((i == size) &&(j==size)) 
                value(i,j) = r_t + values(i,j)*1.0; % 终止状态
            else
                value(i,j) = r + p*(west+east+north+south);
            end
        end
    end
    values = value;
end