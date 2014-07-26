function [F_S frames_index] = bss_make_frames(S,WINDOW,NOVERLAP)

% decompose some signal(s) into frames
%
% Usage: [F_S frames_index] = bss_make_frames(S,WINDOW,NOVERLAP)
%
% Input:
%   - S: matrix of size n x T (with T>n),
%   - WINDOW: 1 x W window
%   - NOVERLAP: number of samples overlap
%
% Output:
%   - F_S: 
%       * if n=1, F_S is a n_frames x W matrix containing the frames (of length W) in
%       rows,
%       * if n>1, F_S is a n_frames x W x n tensor containing the frames
%       decomposition of each row of S.
%   - frames_index: index of the beginning of each frame in the rows of S
%
% Developers:  - Cedric Fevotte (cf269@cam.ac.uk) - Emmanuel Vincent
% (vincent@ircam.fr) - Remi Gribonval (remi.gribonval@irisa.fr)

[n,T]=size(S);

if n>T
    disp('Wrong dimensions: must have T>n.')
    return;
end

%%% Default values %%%
W=length(WINDOW); % Length of window

if T < W
    disp('Please choose a window smaller than the signals.')
    return;
end

n_frames = fix((T-NOVERLAP)/(W-NOVERLAP)); % Number of frames
% If needed the very end of the signal is removed.

frames_index = 1 + (0:(n_frames-1))*(W-NOVERLAP); % Index of beginnings of frames

F_S=zeros(n_frames,W,n); % If n=1, F_S is a 2-D tensor (matrix)

for i=1:n
    for k=1:n_frames
        F_S(k,:,i)=(S(i,frames_index(k)+(0:(W-1))).*WINDOW);
    end
end
