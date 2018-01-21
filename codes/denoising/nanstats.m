function [p,m,n,s] = nanstats( x, d, av1, av2)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
% Averaging?
if ~exist( 'av1', 'var')
	av1 = 1;
end
if ~exist( 'av2', 'var')
	av2 = 1;
end

% Start from higher dimensions
d = sort( d, 'descend');

% Did we get the whole thing?
if isstruct( x)
	% Get each case's stats
	for i = 1:length( x.sdr)
		[p1(i,:),m1(i,:),n1(i,:),s1(i,:)] = nanstats( x.sdr{i}, d, av1, av2);
		[p2(i,:),m2(i,:),n2(i,:),s2(i,:)] = nanstats( x.sir{i}, d, av1, av2);
		[p3(i,:),m3(i,:),n3(i,:),s3(i,:)] = nanstats( x.sar{i}, d, av1, av2);
		[p4(i,:),m4(i,:),n4(i,:),s4(i,:)] = nanstats( x.sto{i}, d, av1, av2);
	end

	% Put them together
	p = [p1 p2 p3 p4];
	m = [m1 m2 m3 m4];
	n = [n1 n2 n3 n4];
	s = [s1 s2 s3 s4];

	% Reshape
	if isvector( p) && size( av1, 2) > 1 && ~isscalar( av1)
		p = reshape( p, size( av1, 2), [])';
		m = reshape( m, size( av1, 2), [])';
		n = reshape( n, size( av1, 2), [])';
		s = reshape( s, size( av1, 2), [])';
	end
	if isvector( p) && size( av2, 1) > 1
		size( p)
		size( av2)
		p = reshape( p, [], size( av2, 1))';
		m = reshape( m, [], size( av2, 1))';
		n = reshape( n, [], size( av2, 1))';
		s = reshape( s, [], size( av2, 1))';
	end
	return
end

% Init
p = x;
m = x;
n = x;
s = x;

% Get each dimension stats
for i = 1:length( d)
	p = nanmean( p, d(i));
	m = nanmax( m, [], d(i));
	n = nanmin( n, [], d(i));
	s = nanstd( s, [], d(i));
end

% Thin out
if length( size( p)) > 2
	p = (av2' * squeeze( p)' * av1)';
	m = (av2' * squeeze( m)' * av1)';
	n = (av2' * squeeze( n)' * av1)';
	s = (av2' * squeeze( s)' * av1)';
else
	p = (av1' * p * av2);
	m = (av1' * m * av2);
	n = (av1' * n * av2);
	s = (av1' * s * av2);
end
return
p = vec( p);
m = vec( m);
n = vec( n);
s = vec( s);
