function y = myspecgram(x,n,sr,w,ov)
% Y = myspecgram(X,NFFT,SR,W,OV)
%      Substitute for Matlab's specgram, calculates & displays spectrogram
% $Header: /homes/dpwe/tmp/e6820/RCS/myspecgram.m,v 1.1 2002/08/04 19:20:27 dpwe Exp $

if (size(x,1) > size(x,2))
  x = x';
end

s = length(x);

if nargin < 2
  n = 256;
end
if nargin < 3
  sr = 1;
end
if nargin < 4
  w = n;
end
if nargin < 5
  ov = w/2;
end
h = w - ov;

halflen = w/2;
halff = n/2;   % midpoint of win
acthalflen = min(halff, halflen);

halfwin = 0.5 * ( 1 + cos( pi * (0:halflen)/halflen));
win = zeros(1, n);
win((halff+1):(halff+acthalflen)) = halfwin(1:acthalflen);
win((halff+1):-1:(halff-acthalflen+2)) = halfwin(1:acthalflen);

c = 1;

% pre-allocate output array
ncols = 1+fix((s-n)/h);
d = zeros((1+n/2), ncols);

for b = 0:h:(s-n)
  u = win.*x((b+1):(b+n));
  t = fft(u);
  d(:,c) = t([1:(1+n/2)]');
  c = c+1;
end;

tt = [0:h:(s-n)]/sr;
ff = [0:(n/2)]*sr/n;

if nargout < 1
  imagesc(tt,ff,20*log10(abs(d)));
  axis xy
  xlabel('Time / s');
  ylabel('Frequency / Hz');
else
  y = d;
end
