function bare( x, y, m, n, w)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
% Fancy bar plot

% Default bar width
if ~exist( 'w', 'var')
	w = .8;
end

% Make the bar plot
h = bar( x, y, w);

% Add the min/max lines
for j = 1:size( y, 2)
	z = get( get( h(j), 'children'), 'xdata');
	for i = 1:size( y, 1)
		line( [z(1,i) z(1,i)]+(z(3,i)-z(1,i))/2, [m(i,j) n(i,j)], 'color', [.5 .5 .5], 'erasemode', 'xor');
	end
end
