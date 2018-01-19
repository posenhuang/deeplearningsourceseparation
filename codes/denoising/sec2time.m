function [t,f] = sec2time( s)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
c = [
	60
	60*60
	60*60*24
	60*60*24*7
];
t = [];

% Weeks
w = floor( s/c(4));
s = s - w*c(4);
if w > 0
	t = sprintf( '%.0fw:', w);
end

% Days
d = floor( s/c(3));
s = s - d*c(3);
if d > 0 || ~isempty( t)
	t = [t sprintf( '%.0fd:', d)];
end

% Hours
h = floor( s/c(2));
s = s - h*c(2);
if h > 0 || ~isempty( t)
	t = [t sprintf( '%.0fh:', h)];
end

% Minutes
m = floor( s/c(1));
s = s - m*c(1);
if m > 0 || isempty( t)
	t = [t sprintf( '%.0fm:', m)];
end

% Secs
t = [t sprintf( '%.1fs', s)];

% Return in clock format
if nargout > 1
	f = [0 0 w*7+d h m s];
end
