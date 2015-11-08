function varargout=bss_decomp_tvgain(varargin)

% decompose an estimated source into target/interference/noise/artefacts components, assuming the admissible distortion is a time-varying gain.
%
% Usage:
%
% [s_target,e_interf[,e_noise],e_artif]=bss_decomp_tvgain(se,index,S[,N],tvshape,tvstep)
%
% Input:
%   - se: row vector of length T containing the estimated source,
%   - index: points which component of S se has to be compared to,
%   - S: n x T matrix containing the original sources,
%   - N: m x T matrix containing the noise on the obseravtions (if any).
%   - tvshape : row vector of length V at most T containing the shape of the elementary 
%     allowed time variations of the gain
%   - tvstep  : hop size (in number of samples) between two consecutive
%     variations of the gain
%
% Output:
%   - s_target: row vector of length T containing the target source(s)
%   contribution,
%   - e_interf: row vector of length T containing the interferences
%   contribution,
%   - e_noise: row vector of length T containing the noise contribution (if
%   any),
%   - e_artif: row vector of length T containing the artifacts
%   contribution.
%
% Developers:  - Cedric Fevotte (fevotte@tsi.enst.fr) - Emmanuel Vincent
% (emmanuel.vincent@irisa.fr) - Remi Gribonval (remi.gribonval@irisa.fr)
       
switch nargin
    case 5
        [varargout{1},varargout{2},varargout{3}]=bss_decomp_tvfilt(varargin{1},varargin{2},varargin{3},varargin{4},varargin{5},0);
    case 6
        [varargout{1},varargout{2},varargout{3},varargout{4}]=bss_decomp_tvfilt(varargin{1},varargin{2},varargin{3},varargin{4},varargin{5},varargin{6},0);
    otherwise
        disp('Wrong number of arguments.')
end

