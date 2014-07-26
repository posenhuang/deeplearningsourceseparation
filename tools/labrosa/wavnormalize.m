function y=wavnormalize(x)

y = x./(max(abs(x))+1e-4);