function test_TSP_general_kl_recurrent(modelname_in, theta, eI, stage, iter)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
eval_types={'dev','test'};
normalize = inline('x./max(abs(x)+1e-3)');
fs = eI.fs;
for ieval=1:numel(eval_types)

    modelname=[modelname_in, '_', eval_types{ieval}];
      
    [mixture, s1, s2]=load_data_mode(eI.data_mode, ieval, eI.DataPath); % 1 for val, 2 for testing    
%%
    winsize = eI.winsize;    nFFT = eI.nFFT;    hop = eI.hop;    scf=eI.scf; %scf = 2/3;
    windows=sin(0:pi/winsize:pi-pi/winsize);
    %% warpped test_feature into one column -- different results from the matrix version
    if eI.train_mode==3
        testmode=3; % setting
    else
        testmode=1; %test
    end
    [test_data_cell, target_ag, mixture_spectrum]=formulate_data_test(mixture, eI, testmode);

    % convert into matrix
    if isfield(eI, 'isdiscrim')  && eI.isdiscrim==2,
        [ cost, grad, numTotal, pred_cell ] = drdae_discrim_joint_kl_obj( theta, eI, test_data_cell, [], mixture_spectrum, true, true);
    elseif isfield(eI, 'isdiscrim')  && eI.isdiscrim==1,
        [ cost, grad, numTotal, pred_cell ] = drdae_discrim_obj( theta, eI, test_data_cell, [], true, true);
    else
        [ cost, grad, numTotal, pred_cell ] = drdae_obj( theta, eI, test_data_cell, [], true, true);
    end
    %%
    if eI.cleanonly==1,
        pred_source_signal=pred_cell{1};
        pred_source_noise=zeros(size(pred_source_signal));
    else
        outputdim=size(pred_cell{1},1)/2;
        pred_source_noise=pred_cell{1}(1:outputdim,:);
        pred_source_signal=pred_cell{1}(outputdim+1:end,:);
    end

    %% input
    spectrum.mix = scf * stft(mixture, nFFT ,windows, hop);
    phase_mix=angle(spectrum.mix);

    if eI.cleanonly==1,
        pred_source_signal=pred_cell{1};
        pred_source_noise=spectrum.mix-pred_source_signal;
    end
    %%
    source_noise =pred_source_noise .* exp(1i.* phase_mix);
    source_signal =pred_source_signal .* exp(1i.* phase_mix);
    wavout_noise = istft(source_noise, nFFT ,windows, hop)';
    wavout_signal = istft(source_signal, nFFT ,windows, hop)';
    
    %  BSS_EVAL ( wav_truth_signal, wav_truth_noise, wav_pred_signal, wav_pred_noise, wav_mix )
    Parms =  BSS_EVAL ( s1, s2, wavout_signal, wavout_noise, mixture );
    if isfield(eI,'ioffset'),
    fprintf('%s %s ioffset:%d iter:%d - no mask - \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n', modelname, stage, eI.ioffset, iter, Parms.SDR, Parms.SIR, Parms.SAR, Parms.NSDR);
       if isfield(eI,'writewav') && eI.writewav==1
        if exist('stage','var')&& (strcmp(stage,'done')||strcmp(stage,'iter'))
        audiowrite([eI.saveDir,modelname,'_ioff',num2str(eI.ioffset),'_nomask_source_noise.wav'], normalize(wavout_noise), fs);
        audiowrite([eI.saveDir,modelname,'_ioff',num2str(eI.ioffset),'_nomask_source_signal.wav'], normalize(wavout_signal), fs);
        end
       end
    else % finish at once
    fprintf('%s %s iter:%d - no mask - \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n', modelname, stage, iter, Parms.SDR, Parms.SIR, Parms.SAR, Parms.NSDR);

     if isfield(eI,'writewav') && eI.writewav==1
       if exist('stage','var')&& (strcmp(stage,'done')||strcmp(stage,'iter')) % not called by save_callback
        audiowrite([eI.saveDir,modelname,num2str(iter),'_nomask_source_noise.wav'], normalize(wavout_noise), fs);
        audiowrite([eI.saveDir,modelname,num2str(iter),'_nomask_source_signal.wav'], normalize(wavout_signal), fs);
       end
     end

    end

    %% binary mask
    masksize=1;
    %   case 1 % binary mask + median filter
    gain=1;
    m= double(abs(pred_source_signal)> (gain*abs(pred_source_noise)));

    source_signal =m .*spectrum.mix;
    source_noise= spectrum.mix-source_signal;

    wavout_noise = istft(source_noise, nFFT ,windows, hop)';
    wavout_signal = istft(source_signal, nFFT ,windows, hop)';

    Parms =  BSS_EVAL ( s1, s2, wavout_signal, wavout_noise, mixture );

     if isfield(eI,'ioffset'),
        fprintf('%s %s ioffset:%d iter:%d - binary mask - \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n', modelname, stage, eI.ioffset, iter, Parms.SDR, Parms.SIR, Parms.SAR, Parms.NSDR);
           if isfield(eI,'writewav') && eI.writewav==1
            if exist('stage','var')&& (strcmp(stage,'done')||strcmp(stage,'iter'))
                audiowrite([eI.saveDir,modelname,'_ioff',num2str(eI.ioffset),'_bmask_source_noise.wav'], normalize(wavout_noise), fs);
                audiowrite([eI.saveDir,modelname,'_ioff',num2str(eI.ioffset),'_bmask_source_signal.wav'], normalize(wavout_signal), fs);
            end
           end
       else % finish at once
        fprintf('%s %s iter:%d - binary mask - \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n', modelname, stage, iter, Parms.SDR, Parms.SIR, Parms.SAR, Parms.NSDR);

         if isfield(eI,'writewav') && eI.writewav==1
           if exist('stage','var')&& (strcmp(stage,'done')||strcmp(stage,'iter')) % not called by save_callback
            audiowrite([eI.saveDir,modelname,num2str(iter),'_bmask_source_noise.wav'], normalize(wavout_noise), fs);
            audiowrite([eI.saveDir,modelname,num2str(iter),'_bmask_source_signal.wav'], normalize(wavout_signal), fs);
           end
         end
     end
    %% softmask
    gain=1;
    % m= double(abs(source_signal)> (gain*abs(source_noise)));
    m= double(abs(pred_source_signal)./(abs(pred_source_signal)+ (gain*abs(pred_source_noise))+eps));

    source_signal =m .*spectrum.mix;
    source_noise= spectrum.mix-source_signal;

    wavout_noise = istft(source_noise, nFFT ,windows, hop)';
    wavout_signal = istft(source_signal, nFFT ,windows, hop)';

    Parms =  BSS_EVAL ( s1, s2, wavout_signal, wavout_noise, mixture );

    if isfield(eI,'ioffset'),
        fprintf('%s %s ioffset:%d iter:%d - soft mask - \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n', modelname, stage, eI.ioffset, iter, Parms.SDR, Parms.SIR, Parms.SAR, Parms.NSDR);

          if isfield(eI,'writewav') && eI.writewav==1
            if exist('stage','var')&& (strcmp(stage,'done')||strcmp(stage,'iter'))
                audiowrite([eI.saveDir,modelname,'_ioff',num2str(eI.ioffset),'_softmask_source_noise.wav'], normalize(wavout_noise), fs);
                audiowrite([eI.saveDir,modelname,'_ioff',num2str(eI.ioffset),'_softmask_source_signal.wav'], normalize(wavout_signal), fs);
            end
          end
    else % finish at once
        fprintf('%s %s iter:%d - soft mask - \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n', modelname, stage, iter, Parms.SDR, Parms.SIR, Parms.SAR, Parms.NSDR);

        if isfield(eI,'writewav') && eI.writewav==1
           if exist('stage','var')&& (strcmp(stage,'done')||strcmp(stage,'iter')) % not called by save_callback
                audiowrite([eI.saveDir,modelname,num2str(iter),'_softmask_source_noise.wav'], normalize(wavout_noise), fs);
                audiowrite([eI.saveDir,modelname,num2str(iter),'_softmask_source_signal.wav'], normalize(wavout_signal), fs);
           end
        end
    end
    %% record max dev soft SDR: dev/test SDR, iter
    global SDR;
    if ieval==1
        if Parms.SDR> SDR.devmax
            SDR.devmax=Parms.SDR;
            SDR.deviter=iter;
        end
    else
        if SDR.deviter==iter
            SDR.testmax=Parms.SDR;
        end
    end
end % eval_type -dev test

return;
end
