function [Y,MX] = logfsgram(X, N, SR, WIN, NOV, FMIN, BPO)
% [Y,MX] = logfsgram(X, N, SR, WIN, NOV, FMIN, BPO)
%    Calculate a log-frequency spectrogram
%    X is input signal; N is parent FFT window; SR is the source samplerate.
%    WIN is actual window length within FFT, NOV is number of overlapping 
%    points between successive windows.
%    Optional FMIN is the lowest frequency to display (80Hz);
%    BPO is the number of bins per octave (12).
%    MX returns the nlogbin x nfftbin mapping matrix;
%    sqrt(MX'*(Y.^2)) is an approximation to the original FFT
%    spectrogram that Y is based on, suitably blurred by going 
%    through the log-F domain.
% 2004-03-30 dpwe@ee.columbia.edu $Header: /homes/dpwe/matlab/columbiafns/RCS/logfsgram.m,v 1.3 2004/04/01 22:39:02 dpwe Exp $

if nargin < 2
  N = 1024;
end
if nargin < 3
  SR = 8000;
end
if nargin < 4
  WIN = [];
end
if nargin < 5
  NOV = [];
end
if nargin < 6
  FMIN = 80;
end
if nargin < 7
  BPO = 12;
end

if isempty(WIN)
  WIN = N;
end
if isempty(NOV)
  NOV = WIN/2;
end

% Calculate underlying STFT
XX = specgram(X,N,SR,WIN,NOV);

% Construct mapping matrix

% Ratio between adjacent frequencies in log-f axis
fratio = 2^(1/BPO);

% How many bins in log-f axis
nbins = floor( log((SR/2)/FMIN) / log(fratio) );

% Freqs corresponding to each bin in FFT
fftfrqs = [0:(N/2)]*(SR/N);
nfftbins = N/2+1;

% Freqs corresponding to each bin in log F output
logffrqs = FMIN * exp(log(2)*[0:(nbins-1)]/BPO);

% Bandwidths of each bin in log F
logfbws = logffrqs * (fratio - 1);

% .. but bandwidth cannot be less than FFT binwidth
logfbws = max(logfbws, SR/N);

% Controls how much overlap there is between adjacent bands
ovfctr = 0.5475;   % Adjusted by hand to make sum(mx'*mx) close to 1.0

% Weighting matrix mapping energy in FFT bins to logF bins
% is a set of Gaussian profiles depending on the difference in 
% frequencies, scaled by the bandwidth of that bin
freqdiff = ( repmat(logffrqs',1,nfftbins) - repmat(fftfrqs,nbins,1) )./repmat(ovfctr*logfbws',1,nfftbins);
mx = exp( -0.5*freqdiff.^2 );
% Normalize rows by sqrt(E), so multiplying by mx' gets approx orig spec back
mx = mx ./ repmat(sqrt(2*sum(mx.^2,2)), 1, nfftbins);

% Perform mapping in magnitude-squared (energy) domain
y = sqrt( mx * (abs(XX).^2) );

% so, we lost phase information...

if nargout < 1
  imagesc([0 length(X)/SR],[1 nbins],20*log10(y));
  axis xy
  xlabel('Time');
  ylabel('Frequency');
  yt = get(gca,'YTick');
  for i = 1:length(yt)
    ytl{i} = sprintf('%.0f',logffrqs(yt(i)));
  end
  set(gca,'YTickLabel',ytl);
else
  Y = y;
  MX = mx;
end
