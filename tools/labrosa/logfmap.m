function [M,N,G] = logfmap(I,L,H)
% [M,N] = logfmap(I,L,H)
%     Return a maxtrix for premultiplying spectrograms to map
%     the rows into a log frequency space.
%     Output map covers bins L to H of input
%     L must be larger than 1, since the lowest bin of the FFT
%     (corresponding to 0 Hz) cannot be represented on a 
%     log frequency axis.  Including bins close to 1 makes 
%     the number of output rows exponentially larger.
%     N returns the recovery matrix such that N*M is approximately I
%     (for dimensions L to H).
%     
% 2004-05-21 dpwe@ee.columbia.edu

% Convert base-1 indexing to base-0
L = L-1;
H = H-1;

ratio = (H-1)/H;
opr = round(log(L/H)/log(ratio));
%ibin = H*exp((opr-[1:opr])*log((H-1)/H));
ibin = L*exp([0:(opr-1)]*-log(ratio));

M = zeros(opr,I);

for i = 1:opr
  % Where do we sample this output bin?
  % Idea is to make them 1:1 at top, and progressively denser below
  % i.e. i = max -> bin = topbin, i = max-1 -> bin = topbin-1, 
  % but general form is bin = A exp (i/B)
%  M(i,round(ibin(i))) = 1;
  tt = pi*([0:(I-1)]-ibin(i));
  M(i,:) = (sin(tt)+eps)./(tt+eps);
end

% Normalize rows, but only if they are boosted by the operation
%G = 1./max(1,diag(M'*M))';
%% Fixup gain in bottom bins
%G(1:find(G==min(G))) = min(G);

G = ones(1,I);
G(1:(H+1)) = [0:H]./H;

% Inverse is just transpose plus scaling
N = (M.*repmat(G,opr,1))';
