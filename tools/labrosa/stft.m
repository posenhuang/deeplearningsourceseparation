function D = stft(x, f, w, h, sr)
% D = stft(X, F, W, H, SR)                       Short-time Fourier transform.
%	Returns some frames of short-term Fourier transform of x.  Each 
%	column of the result is one F-point fft (default 256); each
%	successive frame is offset by H points (W/2) until X is exhausted.  
%       Data is hann-windowed at W pts (F), or rectangular if W=0, or 
%       with W if it is a vector.
%       Without output arguments, will plot like sgram (SR will get
%       axes right, defaults to 8000).
%	See also 'istft.m'.
% dpwe 1994may05.  Uses built-in 'fft'
% $Header: /home/empire6/dpwe/public_html/resources/matlab/pvoc/RCS/stft.m,v 1.4 2010/08/13 16:03:14 dpwe Exp $

if nargin < 2;  f = 256; end
if nargin < 3;  w = f; end
if nargin < 4;  h = 0; end
if nargin < 5;  sr = 8000; end

% expect x as a row
if size(x,1) > 1
  x = x';
end

s = length(x);

if length(w) == 1
  if w == 0
    % special case: rectangular window
    win = ones(1,f);
  else
    if rem(w, 2) == 0   % force window to be odd-len
      w = w + 1;
    end
    halflen = (w-1)/2;
    halff = f/2;   % midpoint of win
    halfwin = 0.5 * ( 1 + cos( pi * (0:halflen)/halflen));
    win = zeros(1, f);
    acthalflen = min(halff, halflen);
    win((halff+1):(halff+acthalflen)) = halfwin(1:acthalflen);
    win((halff+1):-1:(halff-acthalflen+2)) = halfwin(1:acthalflen);
  end
else
  win = w;
end

w = length(win);
% now can set default hop
if h == 0
  h = floor(w/2);
end

c = 1;

% pre-allocate output array
d = zeros((1+f/2),1+fix((s-f)/h));

for b = 0:h:(s-f)
  u = win.*x((b+1):(b+f));
  t = fft(u);
  d(:,c) = t(1:(1+f/2))';
  c = c+1;
end;

% If no output arguments, plot a spectrogram
if nargout == 0
  tt = [0:size(d,2)]*h/sr;
  ff = [0:size(d,1)]*sr/f;
  imagesc(tt,ff,20*log10(abs(d)));
  axis('xy');
  xlabel('time / sec');
  ylabel('freq / Hz')
  % leave output variable D undefined
else
  % Otherwise, no plot, but return STFT
  D = d;
end

