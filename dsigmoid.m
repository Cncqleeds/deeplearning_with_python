%% 对任意标量、向量求sigmoid的导数(但不能是导数)，返回值 sigmoid(z).*(1-sigmoid(z))
function s = dsigmoid(z)
s = diag(sigmoid(z).*(1-sigmoid(z)));
% s = diag(s);
end
