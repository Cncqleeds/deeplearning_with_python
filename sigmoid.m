%% 对任意标量、向量、矩阵求sigmoid值，返回值 1/（1+exp(-x)）
function s = sigmoid(z)
ez = exp(-z);
s = 1 ./ (1 + ez);
end
