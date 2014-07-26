function [varargout]=bss_energy_ratios(F_s_target,F_e_interf,varargin)

% compute energy ratios corresponding to SDR/SIR/SNR/SAR given a decomposition of an estimated source into target/interference/noise/artifacts over frames.
%
% Usage:
%
%    [SDR,SIR,SAR]    =bss_energy_ratios(F_s_target,F_e_interf,F_e_artif)
%    [SDR,SIR,SNR,SAR]=bss_energy_ratios(F_s_target,F_e_interf,F_e_noise,F_e_artif)
%
% Input:
%   - F_s_target: n_frames x T matrix containing the frames of the target source contribution,
%   - F_e_interf: n_frames x T matrix containing the frames of the interferences contribution,
%   - F_e_noise: n_frames x T matrix containing the frames of the noise contribution (if any),
%   - F_e_artif: n_frames x T matrix containing the frames of the artifacts contribution.
%
% Ouput:
%   - SDR: n_frames x 1 vector contaning the Source to Distortion Ratios per frame,
%   - SIR: n_frames x 1 vector contaning the Source to Interferences Ratios per frame,
%   - SNR: n_frames x 1 vector contaning the Signal to Noise Ratios (if noise) per frame,
%   - SAR: n_frames x 1 vector contaning the Signal to Artifacts Ratios per frame.
%
% Developers:  - Cedric Fevotte (cf269@cam.ac.uk) - Emmanuel Vincent
% (vincent@ircam.fr) - Remi Gribonval (remi.gribonval@irisa.fr)

switch nargin
    case 3
        F_e_artif=varargin{1};
        % SDR
        F_e_total=F_e_interf+F_e_artif;
        varargout{1}= sum(F_s_target.^2,2)./sum(F_e_total.^2,2);
        % SIR
        varargout{2}=sum(F_s_target.^2,2)./sum(F_e_interf.^2,2);
        % SAR
        varargout{3}=sum((F_s_target+F_e_interf).^2,2)./sum(F_e_artif.^2,2);        
        
    case 4        
        F_e_noise=varargin{1};
        F_e_artif=varargin{2};
        % SDR
        F_e_total=F_e_interf+F_e_noise+F_e_artif;
        varargout{1}=sum(F_s_target.^2,2)./sum(F_e_total.^2,2);
        % SIR
        varargout{2}=sum(F_s_target.^2,2)./sum(F_e_interf.^2,2);
        % SNR
        varargout{3}=sum((F_s_target+F_e_interf).^2,2)./sum(F_e_noise.^2,2);
        % SAR
        varargout{4}=sum((F_s_target+F_e_interf+F_e_noise).^2,2)./sum(F_e_artif.^2,2);        
end