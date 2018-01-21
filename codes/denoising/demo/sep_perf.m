function [sdr,sir,sar,st,pq,ps] = sep_perf( sep, orig, sr, o)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
% Which measures to compute
if ~exist( 'o', 'var')
	o = 'diaqst';
	if nargout > 0
		o = o(1:nargout);
	end
end

% Default sample rate
if ~exist( 'sr', 'var')
	sr = 16000;
end

% Sort out the sizes
if size( sep, 2) > size( orig, 2)
	sep = sep(:,1:size( orig, 2));
elseif size( sep, 2) < size( orig, 2)
	orig = orig(:,1:size( sep, 2));
end

% BSSEVAL metrics
for i = size( sep, 1):-1:1
	[e1,e2,e3] = bss_decomp_gain( sep(i,:), i, orig);
	[sdr(1,i),sir(1,i),sar(1,i)] = bss_crit( e1, e2, e3);
end

% STOI, higher is better
st = [];
if any( o == 't') || nargout > 3 || nargout == 0
	for i = 1:size( sep, 1)
		st(1,i) = stoi( double( orig(i,:)), double( sep(i,:)), sr);
	end
end

% PEAQ, 0 (imperceptible difference) to -4 (very annoying difference)
pq = [];
if any( o == 'e') || nargout > 4 || nargout == 0
	for i = 1:size( sep, 1)
		pq(1,i) = peaq( orig(i,:), sep(i,:), sr, 'peaq');
	end
end

% PESQ, 1 (bad) to 5 (excellent)
ps = [];
if any( o == 's') || nargout > 5 || nargout == 0
	for i = 1:size( sep, 1)
		ps(1,i) = peaq( orig(i,:), sep(i,:), sr, 'pesq');
	end
end

% Show me
for i = 1:size( sep, 1)
	fprintf( 'SDR: %3.2f, SIR: %3.2f, SAR: %3.2f', sdr(i), sir(i), sar(i));
	if ~isempty( pq)
		fprintf(', PEAQ: %3.2f', pq(i));
	end
	if ~isempty( ps)
		fprintf( ', PESQ: %3.2f', ps(i));
	end
	if ~isempty( st)
		fprintf( ', STOI: %3.2f', st(i));
	end
	fprintf( '\n');
end

% Pack in one structure if asked to
if nargout == 1 && length( o) > 1
	c.sdr = sdr;
	c.sir = sir;
	c.sar = sar;
	c.st = st;
	c.pq = pq;
	c.ps = ps;
	sdr = c;
end
