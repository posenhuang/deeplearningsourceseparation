function prediction = test_denoising_general_kl_bss3(mixture_wav, theta, eI, stage, iter)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
% Run test and evaluation given a model.
    mixture = mixture_wav;

    winsize = eI.winsize;    nFFT = eI.nFFT;    hop = eI.hop;    scf=eI.scf;
    %% warpped test_feature into one column --
    %  different results from the matrix version
    if eI.train_mode==3
        testmode=3; % formulate one data per cell
    else
        testmode=1; %test
    end
    [test_data_cell, target_ag, mixture_spectrum]= ...
                                formulate_data_test(mixture, eI, testmode);

    %%
    if isfield(eI, 'isdiscrim')  && eI.isdiscrim==2,
        [ cost, grad, numTotal, pred_cell ] = drdae_discrim_joint_kl_obj...
            (theta, eI, test_data_cell, [], mixture_spectrum, true, true);
    elseif isfield(eI, 'isdiscrim')  && eI.isdiscrim==1,
        [ cost, grad, numTotal, pred_cell ] = drdae_discrim_obj(theta, eI, ...
                                            test_data_cell, [], true, true);
    else
        [ cost, grad, numTotal, pred_cell ] = drdae_obj(theta, eI, test_data_cell, [], true, true);
    end

    if eI.cleanonly==1,
        pred_source_signal=pred_cell{1};
        pred_source_noise=zeros(size(pred_source_signal));
    else
        outputdim=size(pred_cell{1},1)/2;
        pred_source_noise=pred_cell{1}(1:outputdim,:);
        pred_source_signal=pred_cell{1}(outputdim+1:end,:);
    end

    %% input
%         spectrum.mix = scf * stft(mixture, nFFT ,windows, hop);          
    wn = sqrt( hann( nFFT, 'periodic')); % hann window        
    if eI.MFCCorlogMelorSpectrum==2 || eI.MFCCorlogMelorSpectrum==3
     	spectrum.mix = scf * stft2( mixture, nFFT, hop, 0, wn); 
    else 
        spectrum.mix = scf * stft3( mixture, nFFT, hop, 0, wn, [], 0);
    end
    
    phase_mix=angle(spectrum.mix);

    if eI.cleanonly==1,
        pred_source_signal=pred_cell{1};
        pred_source_noise=spectrum.mix-pred_source_signal;
    end

    %% softmask
    gain=1;
    m= double(abs(pred_source_signal)./(abs(pred_source_signal)+ (gain*abs(pred_source_noise))+eps));

    source_signal =m .*spectrum.mix;
    source_noise= spectrum.mix-source_signal;

    prediction = source_signal;
%         wavout.noise = istft(source_noise, nFFT ,windows, hop)';
%         wavout.signal = istft(source_signal, nFFT ,windows, hop)';
% 
%         Parms =  BSS_EVAL ( s1, s2, wavout_signal, wavout_noise, mixture );
%         if  isfield(eI,'bss3') && eI.bss3==1
%             Parms_bss3 =  BSS_3_EVAL ( s1, s2, wavout_signal, wavout_noise, mixture );
%         else
%             Parms_bss3.SDR_bss3=0; Parms_bss3.SIR_bss3=0; Parms_bss3.SAR_bss3=0; Parms_bss3.NSDR_bss3=0;
%         end
% 
%         if isfield(eI,'ioffset'),
%             fprintf('%s %s %s ioffset:%d iter:%d - soft mask - \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n', ...
%                 modelname, testname, stage, eI.ioffset, iter, Parms.SDR, Parms.SIR, Parms.SAR, Parms.NSDR);
%             fprintf('%s %s %s ioffset:%d iter:%d - soft mask bss3- \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n', ...
%                 modelname, testname, stage, eI.ioffset, iter,Parms_bss3.SDR_bss3, Parms_bss3.SIR_bss3, Parms_bss3.SAR_bss3, Parms_bss3.NSDR_bss3);
% 
%               if isfield(eI,'writewav') && eI.writewav==1
%                 if exist('stage','var')&& (strcmp(stage,'done')||strcmp(stage,'iter'))
%                     wavwrite(wavout_noise, fs, [eI.saveDir,testname,'_ioff',num2str(eI.ioffset),'_softmask_noise.wav']);
%                     wavwrite(wavout_signal, fs, [eI.saveDir,testname,'_ioff',num2str(eI.ioffset),'_softmask_signal.wav']);
%                 end
%               end
%         else % finish at once
%             fprintf('%s %s %s iter:%d - soft mask - \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n', ...
%                 modelname, testname, stage, iter, Parms.SDR, Parms.SIR, Parms.SAR, Parms.NSDR);
%             fprintf('%s %s %s iter:%d - soft mask bss3- \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n', ...
%                 modelname, testname, stage, iter, Parms_bss3.SDR_bss3, Parms_bss3.SIR_bss3, Parms_bss3.SAR_bss3, Parms_bss3.NSDR_bss3);
% 
%             if isfield(eI,'writewav') && eI.writewav==1
%                if exist('stage','var')&& (strcmp(stage,'done')||strcmp(stage,'iter')) % not called by save_callback
%                     wavwrite(wavout_noise, fs, [eI.saveDir,testname,'_iter',num2str(iter),'_softmask_noise.wav']);
%                     wavwrite(wavout_signal, fs, [eI.saveDir,testname,'_iter',num2str(iter),'_softmask_signal.wav']);
%                end
%             end
%         end

%         GNSDR.soft(ifile) = Parms.NSDR;
%         GNSDR_bss3.soft(ifile) = Parms_bss3.NSDR_bss3;

return;

%% unit test
% (TODO) add
% savedir='results';
% iter=200;
% modelname='model_test';
% load([savedir,modelname, filesep, 'model_',num2str(iter),'.mat']);
% test_mir1k_general_kl_bss3(modelname_in, theta, eI, stage, iter);

end
