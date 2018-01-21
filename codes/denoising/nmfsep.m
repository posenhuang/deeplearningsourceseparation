function G = nmfsep( ft, fn, L, it)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%

% Learn models
if ~isstruct( L)
	if gpuDeviceCount()
		ft = gpuArray( single( ft));
		fn = gpuArray( single( fn));
	end

	G.w{1} = wpu( ft, L(1), it);
	G.w{2} = wpu( fn, L(2), it);

% Separate
else
	
	if gpuDeviceCount()
		ft = gpuArray( single( ft));
	end
	h = pu( ft, [L.w{:}], 100);

	% Get current part
	f = L.w{1}*h(1:size(L.w{1},2),:);

	% Wienerize
	G = gather( f .* (ft./([L.w{:}]*h)));
end


%-------------------------------------
function w = wpu( x, R, ep)

% Normalize input
g = sum( x, 1);
x = bsxfun( @rdivide, x, g+eps);

% Init distributions
[m,n] = size( x);
w = rand( m, R); w = bsxfun( @rdivide, w, sum( w, 1));
h = rand( R, n); h = bsxfun( @rdivide, h, sum( h, 1));
if gpuDeviceCount()
	w = gpuArray( single( w));
	h = gpuArray( single( h));
end

% Start churning
for e = 1:ep
	% Get tentative estimates
	V = x ./ (eps+w*h);
	nw = w .* (V*h');
	nh = h .* (w'*V);

	% Get new estimates and normalize them
	w = bsxfun( @rdivide, nw, eps+sum( nw));
	h = bsxfun( @rdivide, nh, eps+sum( nh));
end


%-------------------------------------
function h = pu( x, w, ep)

% Normalize input
g = sum( x, 1);
x = bsxfun( @rdivide, x, g+eps);

% Init distributions
[~,n] = size( x);
r = cols( w);
h = rand( r, n);
h = bsxfun( @rdivide, h, sum( h, 1));

if gpuDeviceCount()
	h = gpuArray( single( h));
end

% Start churning
for e = 1:ep

	% Get tentative estimates
	V = x ./ (eps+w*h);
	nh = h .* (w'*V);

	% Normalize
	h = bsxfun( @rdivide, nh, eps+sum( nh, 1));
end

% Put back frame gain info
h = bsxfun( @times, h, g);
