function [f,fv,tv] = stft2( s, sz, hp, pd, wn, w, pf, sr)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
% Short-time Fourier transform
%
% function [f,fv,tv] = stft( x, sz, hp, pd, wn, w, pf, sr)
%
% Inputs: 
%  x   input time series (must be row vector), or input complex spectrogram (DC to Nyquist)
%  sz  size of the FFT
%  hp  hop size in samples
%  pd  pad size in samples
%  wn  window to use (function name of data vector)
%  w   optional frequency warping (or unwarping) matrix (use wft.m)
%  pf  flag for padding input boundaries (default = 1)
%      if larger than 1 is is the desired size of the output with no padding
%  sr  input sample rate to plot correct frequencies and time axes (default = 1)
%
% Output:
%  f   complex STFT output (only DC to Nyquist components), or time series resynthesis
%  fv  Vector of frequencies for each spectrogram row
%  tv  Vector of times for each spectrogram column


% Default size
if ~exist( 'sz', 'var')
	sz = 1024;
end

% Default hop
if ~exist( 'hp', 'var')
	hp = sz/4;
end

% Default pad
if ~exist( 'pd', 'var')
	pd = 0;
end

% Default window
if ~exist( 'wn', 'var')
	wn = 'hann';
end

% Pad by default
if ~exist( 'pf', 'var')
	pf = 1;
end

% Default sample rate
if ~exist( 'sr', 'var')
	sr = 1;
end

% Just in case
if isempty( s)
	f = [];
	return
end

% Get class type
cl = class( s);

% Forward transform
if isreal( s)

	% Setup the window and guess a gain
	if ischar( wn)
		wn = feval( wn, sz);
	elseif isa( wn, 'function_handle')
		wn = wn( sz);
	end
	wn = wn + 0*s(1); % To promote to GPU data type if needed

	% Pad at edges
	if pf == 1
		x = [zeros( 1, sz+pd-hp, cl) s zeros( 1, sz+pd, cl)]';
	else
		x = s';
	end

	% Pack in a matrix
if 0
	S = zeros( sz, ceil( (length(x)-sz)/hp), cl);
	si = 1:sz;
	for i = 0:hp:length(x)-sz-1
		S(:,i/hp+1) = wn .* x(i+si);
	end
else
	l = ceil( (length( x)-sz)/hp);
	ri = (1:sz)';
	ci = 1 + (0:(l-1))*hp;
	S = zeros( sz, l, cl);
	S(:) = x(ri(:,ones(1,l))+ci(ones(sz,1),:)-1);
	S = bsxfun( @times, S, wn);
end

	% FFT and keep lower half (add eps just in case)
	f = fft( S, sz+pd, 1);
	f = f(1:(sz+pd)/2+1,:) + eps;

	% Get frequency and time vectors
	fv = linspace( 0, sr/2, (sz+pd)/2+1)';
	tv = (1:size(f,2))*hp/sr; % Fix for padding

	% Apply optional warping
	if exist( 'w', 'var') && ~isempty( w) && ~isa( w, 'function_handle')
		if isstruct( w)
			fv = sr*w.f/(2*pi);
			w = w.w;
		end
		f = w*f;
%		fv = w*fv;
	else
		w = [];
	end

	% Show me
	if nargout == 0
		stft_plot( f, fv, tv, sr, w);
	end

else
	
	% Setup the window
	if ischar( wn)
		wn = feval( wn, sz);
	elseif isa( wn, 'function_handle')
		wn = wn( sz);
	end
	if pd
		wn = [wn;zeros(pd,1)];
	end
	wn = wn + 0*s(1);
	
	% Unwarp if needed
	if exist( 'w', 'var') && ~isempty( w)
%		s = pinv(w.w)*s;
		if isstruct( w)
			s = w.w'*s;
		else
			s = w'\s;
		end
	end
	
	% Invert all FFTs
	F = [s; conj( s(end-1:-1:2,:))];
	F = bsxfun( @times, real( ifft( F, [], 1)), wn/(sz/hp));

	% Overlap add
if 0
	f = zeros( 1, size( F, 2)*hp+sz+pd-1, class( s));
	si = 1:(sz+pd);
	for i = 1:size( F, 2)
		f((i-1)*hp+si) = f((i-1)*hp+si) + F(:,i)';
	end
else
	f = zeros( 1, size( F, 2)*hp+sz+pd-1, class( s));
	for i = 1:(sz+pd)/hp
		tf = vec( F(:,i:(sz+pd)/hp:end));
		f((i-1)*hp+(1:length(tf))) = f((i-1)*hp+(1:length(tf))) + tf';
	end
end

	% Remove original padding
	if pf == 1
		f = f(sz+pd-hp+1:end-sz-pd);
	elseif pf > 1
		f = f(1:pf);
	end
end


%----------------
function stft_plot( f, fv, tv, sr, w)

if 0
%if isa( w, 'function_handle') || ~isempty( w)
	pcolor( tv, w( fv), abs( f).^.35), axis xy, shading interp
	td = []; fd = []; cd = [];
	for i = 1:size( f, 1)
		for j = 1:size( f, 2)
			td(:,end+1) = tv(i)+[0 0 1 1 0]';
			fd(:,end+1) = fv(i)+[0 1 1 0 0]';
			cd(end+1) = abs( f).^.4;
		end
	end
	zd = ones( size( td));
%	patch( td, fd, zd, 'b', FaceColor','flat', ...
%		'CData', cd, 'CDataMapping', 'scaled');
else
%	imagesc( tv, fv, abs( f).^.4), axis xy
	imagesc( tv, fv, abs( f).^.4), axis xy
if 0
	ii = [];
	for i = [100 500 1000 2000 3500 12000 20000] % 30000 40000 50000 60000 70000];
		[~,ii(end+1)] = min( abs( fv - i));
	end
	ii = unique( ii);
	set( gca, 'ytick', .5*ii*sr/length(fv), 'yticklabel', 100*round(fv(ii)/100));	
	end
end
if sr == 1
	ylabel( 'Frequency (normalized)')
	xlabel( 'Time (samples)')
else
	ylabel( 'Frequency (Hz)')
	xlabel( 'Time (sec)')
end
