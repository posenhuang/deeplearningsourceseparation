function output = test_denoising_general_kl_bss3(mixture_wav, theta, eI, stage, iter)
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

    output.source_signal =m.*spectrum.mix(:, 1:min(size(m,2), size(spectrum.mix,2)));
    output.source_noise= spectrum.mix-output.source_signal;
 
return;

end
