function Parms =  BSS_3_EVAL...
  (wav_truth_signal, wav_truth_noise, wav_pred_signal, wav_pred_noise, wav_mix)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
% Run BSS_EVAL 3.0
%% evaluate
if length(wav_pred_noise)==length(wav_truth_noise)
    sep = [wav_pred_noise , wav_pred_signal]';
    orig = [wav_truth_noise , wav_truth_signal]';
else
    minlength=min( length(wav_pred_noise), length(wav_truth_noise) );

    sep = [wav_pred_noise(1:minlength) , wav_pred_signal(1:minlength)]';
    orig = [wav_truth_noise(1:minlength) , wav_truth_signal(1:minlength)]';
end

[SDR_bss3, SIR_bss3, SAR_bss3, perm] = bss_eval_sources ( sep, orig );

res(1,:)=SDR_bss3(perm)';
res(2,:)=SIR_bss3(perm)';
res(3,:)=SAR_bss3(perm)';

[SDR_, SIR_, SAR_, perm] = bss_eval_sources (  wav_mix', wav_truth_signal');
sdr_ = SDR_(perm)';

Parms.SDR_bss3=res(1,2);
Parms.SIR_bss3=res(2,2);
Parms.SAR_bss3=res(3,2);
Parms.NSDR_bss3=Parms.SDR_bss3-sdr_;

