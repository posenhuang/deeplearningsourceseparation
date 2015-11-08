function varargout=bss_crit(varargin)

% compute evaluation criteria given a decomposition of an estimated source into target/interference/noise/artifacts 
% of the form
%
%   se = s_target + e_interf (+ e_noise) + e_artif 
%
% Usage:
%
% 1) Global mode
%
% [SDR,SIR,(SNR,)SAR]=bss_crit(s_target,e_interf[,e_noise],e_artif)
%
% Input:
%   - s_target: row vector of length T containing the target source(s)
%   contribution,
%   - e_interf: row vector of length T containing the interferences
%   contribution,
%   - e_noise: row vector of length T containing the noise contribution 
%   (if any),
%   - e_artif: row vector of length T containing the artifacts
%   contribution.
%
% Output:
%   - SDR: Source to Distortion Ratio,
%   - SIR: Source to Interferences Ratio,
%   - SNR: Signal to Noise Ratio (if e_noise is provided),
%   - SAR: Source to Artifacts Ratio.
%
% 2) Local mode
%
% [SDR,SIR,(SNR,)SAR]=bss_crit(s_target,e_interf[,e_noise],e_artif,WINDOW,NOVERLAP)
%
% Additional input:
%   - WINDOW: 1 x W window
%   - NOVERLAP: number of samples of overlap between consecutive windows
%
% Output:
%   - SDR: n_frames x 1 vector containing local Source to Distortion Ratio,
%   - SIR: n_frames x 1 vector containing local Source to Interferences Ratio,
%   - SNR: n_frames x 1 vector containing local Signal to Noise Ratio,
%   - SAR: n_frames x 1 vector containing local Source to Artifacts Ratio.
%
% Developers:  - Cedric Fevotte (fevotte@tsi.enst.fr) - Emmanuel Vincent
% (emmanuel.vincent@irisa.fr) - Remi Gribonval (remi.gribonval@irisa.fr)


s_target=varargin{1}; e_interf=varargin{2}; 

switch nargin
    case 3
        e_noise=[]; e_artif=varargin{3};
        mode='global';
    case 4
        e_noise=varargin{3}; e_artif=varargin{4};
        mode='global';
    case 5
        e_noise=[]; e_artif=varargin{3};
        WINDOW=varargin{4}; NOVERLAP=varargin{5};
        mode='local';
    case 6
        e_noise=varargin{3}; e_artif=varargin{4};
        WINDOW=varargin{5}; NOVERLAP=varargin{6};
        mode='local';   
end

switch mode        
    case 'global'
        switch isempty(e_noise)
            case 1
                % Computation of the energy ratios
                [SDR,SIR,SAR]=bss_energy_ratios(s_target,e_interf,e_artif);
                varargout{1}=10*log10(SDR); varargout{2}=10*log10(SIR); varargout{3}=10*log10(SAR);
            case 0
                % Computation of the energy ratios
                [SDR,SIR,SNR,SAR]=bss_energy_ratios(s_target,e_interf,e_noise,e_artif);
                varargout{1}=10*log10(SDR); varargout{2}=10*log10(SIR);
                varargout{3}=10*log10(SNR); varargout{4}=10*log10(SAR);                
        end
        
    case 'local'
        
        
        switch isempty(e_noise)
            case 1
                F_s_target=bss_make_frames(s_target,WINDOW,NOVERLAP);
                F_e_interf=bss_make_frames(e_interf,WINDOW,NOVERLAP);
                F_e_artif=bss_make_frames(e_artif,WINDOW,NOVERLAP);
                [SDR,SIR,SAR]=bss_energy_ratios(F_s_target,F_e_interf,F_e_artif);
                varargout{1}=10*log10(SDR); varargout{2}=10*log10(SIR); varargout{3}=10*log10(SAR);
            case 0
                F_s_target=bss_make_frames(s_target,WINDOW,NOVERLAP);
                F_e_interf=bss_make_frames(e_interf,WINDOW,NOVERLAP);
                F_e_noise=bss_make_frames(e_noise,WINDOW,NOVERLAP);
                F_e_artif=bss_make_frames(e_artif,WINDOW,NOVERLAP);
                [SDR,SIR,SNR,SAR]=bss_energy_ratios(F_s_target,F_e_interf,F_e_noise,F_e_artif);
                varargout{1}=10*log10(SDR); varargout{2}=10*log10(SIR);
                varargout{3}=10*log10(SNR); varargout{4}=10*log10(SAR);
        end        
end %mode