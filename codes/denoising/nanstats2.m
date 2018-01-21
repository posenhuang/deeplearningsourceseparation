function [p,m,n,s] = nanstats2( x, d, f)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
% Averaging?
if ~exist( 'f', 'var')
	f = @(x)x;
end

% Start from higher dimensions
d = sort( d, 'descend');

% Did we get the whole thing?
if isstruct( x)
	% Get each case's stats
	for i = 1:length( x.sdr)
		[p1(i,:),m1(i,:),n1(i,:),s1(i,:)] = nanstats2( x.sdr{i}, d, f);
		[p2(i,:),m2(i,:),n2(i,:),s2(i,:)] = nanstats2( x.sir{i}, d, f);
		[p3(i,:),m3(i,:),n3(i,:),s3(i,:)] = nanstats2( x.sar{i}, d, f);
		[p4(i,:),m4(i,:),n4(i,:),s4(i,:)] = nanstats2( x.sto{i}, d, f);
	end

	p(:,:,1) = p1; p(:,:,2) = p2; p(:,:,3) = p3; p(:,:,4) = p4;
	m(:,:,1) = m1; m(:,:,2) = m2; m(:,:,3) = m3; m(:,:,4) = m4;
	n(:,:,1) = n1; n(:,:,2) = n2; n(:,:,3) = n3; n(:,:,4) = n4;
	s(:,:,1) = s1; s(:,:,2) = s2; s(:,:,3) = s3; s(:,:,4) = s4;
	p = squeeze( p);
	m = squeeze( m);
	n = squeeze( n);
	s = squeeze( s);

	% Put them together
%	p = [p1 p2 p3 p4];
%	m = [m1 m2 m3 m4];
%	n = [n1 n2 n3 n4];
%	s = [s1 s2 s3 s4];

	% Reshape
%	if isvector( p) && size( av1, 2) > 1
%		p = reshape( p, size( av1, 2), [])';
%		m = reshape( m, size( av1, 2), [])';
%		n = reshape( n, size( av1, 2), [])';
%		s = reshape( s, size( av1, 2), [])';
%	end
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

% Process if needed
p = f( p);
m = f( m);
n = f( n);
s = f( s);
