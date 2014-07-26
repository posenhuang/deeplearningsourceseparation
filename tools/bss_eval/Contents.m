% BSS_EVAL - Toolbox for evaluation of Blind Audio Source Separation (BASS)
% methods.
% 
% Developers:  - Cedric Fevotte (cf269@cam.ac.uk) - Emmanuel Vincent
% (vincent@ircam.fr) - Remi Gribonval (remi.gribonval@irisa.fr)
% 
% Files
%   bss_crit          - compute evaluation criteria given a decomposition of an estimated source into target/interference/noise/artifacts 
%   bss_decomp_filt   - decompose an estimated source into target/interference/noise/artefacts components, assuming the admissible distortion is a pure time-invariant filter.
%   bss_decomp_gain   - decompose an estimated source into target/interference/noise/artefacts components, assuming the admissible distortion is a pure time-invariant gain.
%   bss_decomp_tvfilt - decompose an estimated source into target/interference/noise/artefacts components, assuming the admissible distortion is a time-varying filter.
%   bss_decomp_tvgain - decompose an estimated source into target/interference/noise/artefacts components, assuming the admissible distortion is a time-varying gain.
%   bss_make_frames   - decompose some signal(s) into frames
%   bss_make_lags     - create a matrix containing lagged versions of some signal(s).
%   bss_proj          - compute the orthogonal projection of x on the subspace spanned by the row(s) of Y.
%   bss_tvproj        - compute the orthogonal projection of x onto the space of shifted windowed versions of the row(s) of Y
%   bss_energy_ratios - compute energy ratios corresponding to SDR/SIR/SNR/SAR given a decomposition of an estimated source into target/interference/noise/artifacts over frames.
