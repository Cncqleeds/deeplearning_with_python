%% �����������������������sigmoidֵ������ֵ 1/��1+exp(-x)��
function s = sigmoid(z)
ez = exp(-z);
s = 1 ./ (1 + ez);
end
