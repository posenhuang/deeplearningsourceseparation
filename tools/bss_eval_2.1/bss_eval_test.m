function bss_eval_test(T,N,W,L)
% check the behaviour of some basic functions of the toolbox BSS_EVAL
%
%
% Usage: bss_eval_test(T,N,L)
%
% Input:
%   - T: number of samples of the test signals (default 1000)
%   - N: number of sources in the test signals (default 2)
%   - W: hop size = half-size of the window (default ceil(T/20));
%   - L: number of taps of the filter (default 0)
%
% Ouput: None
%
% Developers:  - Cedric Fevotte (fevotte@tsi.enst.fr) - Emmanuel Vincent
% (emmanuel.vincent@irisa.fr) - Remi Gribonval (remi.gribonval@irisa.fr)

if nargin<1
    T = 1000;
end
if nargin<2
    N=2;
end
if nargin<3
    W=ceil(T/20);
end
if nargin<4
    L=0;
end


    
% 1. Generate some data
S  = rand(N,T);

% 2. check that the SDR,SIR and SAR are almost infinite when performing a
% tvfilt decomposition with half overlapping rectangular windows
win = ones(1,W*2);
[starget,einterf,eartif]=bss_decomp_tvfilt(S(1,:),1,S,win,W,L);
[SDR,SIR,SAR]=bss_crit(starget,einterf,eartif);
disp(['SDR: ' num2str(SDR) ' SIR: ' num2str(SIR) 'SAR: ' num2str(SAR)]);
disp('Results for rectangular windows are expected to exceed 200 dB');


% 3. check that the SDR,SIR and SAR are almost infinite when performing a
% tvfilt decomposition with half overlapping triangular windows
win = triang(W*2)';
[starget,einterf,eartif]=bss_decomp_tvfilt(S(1,:),1,S,win,W,L);
[SDR,SIR,SAR]=bss_crit(starget,einterf,eartif);
disp(['SDR: ' num2str(SDR) ' SIR: ' num2str(SIR) 'SAR: ' num2str(SAR)]);
disp('Results for triangular windows are expected to exceed 200 dB');

%plot(starget,'r')
% hold on;
% triangsum = zeros(N*100+length(win),1);
% for i=1:100
%     idxrange = (i-1)*N+(1:length(win));
%     triangsum(idxrange) = triangsum(idxrange)+win';
% end
% plot(triangsum);


function w = triang(n)
% TRIANG Triangular window.
if rem(n,2)
% It's an odd length sequence
w = 2*(1:(n+1)/2)/(n+1);
w = [w w((n-1)/2:-1:1)]';
else
% It's even
w = (2*(1:(n+1)/2)-1)/n;
w = [w w(n/2:-1:1)]';
end