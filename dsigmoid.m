%% �����������������sigmoid�ĵ���(�������ǵ���)������ֵ sigmoid(z).*(1-sigmoid(z))
function s = dsigmoid(z)
s = diag(sigmoid(z).*(1-sigmoid(z)));
% s = diag(s);
end
