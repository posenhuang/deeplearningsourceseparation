function [w,v] = paramcell( xi)

n = length( xi);
j = 1:n;
siz = cellfun( @length, xi);
v = cell( 1, n);
for i = 1:n
	x = xi{j(i)};

	s = ones( 1, n); 
	s(i) = numel( x);
	x = reshape( x, s);

	s = siz; 
	s(i) = 1;
	v{i} = repmat( x, s(:)');
end

w = cell( numel( v{1}), 1);
for i = 1:numel( v{1})
	for j = 1:n
		w{i,j} = v{j}{i};
	end
end
