function [DATA, mixture_spectrum, eI] = compute_features(dmix, eI)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
% compute features
% Spectra, log power spectra, MFCC, logmel

    winsize = eI.winsize; nFFT = eI.nFFT; hop = eI.hop; scf = eI.scf;
    windows = sin(0:pi/winsize:pi-pi/winsize);

    spectrum_mix = scf * stft(dmix, eI.nFFT ,windows, hop, eI.fs);
    mixture_spectrum=abs(spectrum_mix);

    if eI.MFCCorlogMelorSpectrum==2, %Spectrum
        DATA = scf * stft(dmix, eI.nFFT ,windows, hop, eI.fs);
        DATA=abs(DATA);
    elseif eI.MFCCorlogMelorSpectrum==3, %log power spectrum
        DATA = scf * stft(dmix, eI.nFFT ,windows, hop, eI.fs);
        DATA=abs(DATA);
        DATA=log(DATA.*DATA+eps);
    else
        %% training features
        filename=[eI.saveDir,'dmix_temp.wav'];
        audiowrite(filename, dmix, eI.fs);

        if eI.framerate==64,
            if eI.MFCCorlogMelorSpectrum==0, %MFCC
                eI.config='mfcc_64ms_step32ms.cfg';
            elseif eI.MFCCorlogMelorSpectrum==1, %logmel
                eI.config='fbank_64ms_step32ms.cfg';
            else % spectrum
                eI.config='spectrum_64ms_step32ms.cfg';
            end
        else % framerate == 32
            if eI.MFCCorlogMelorSpectrum==0, %MFCC
                eI.config='mfcc_32ms_step16ms.cfg';
            elseif eI.MFCCorlogMelorSpectrum==1, %logmel
                eI.config='fbank_32ms_step16ms.cfg';
            else % spectrum
                eI.config='spectrum_32ms_step16ms.cfg';
            end
        end
        command=sprintf('HCopy -A -C %s%s %s%s %s%s',...
            eI.CFGPath,eI.config,eI.saveDir,'dmix_temp.wav',...
            eI.saveDir, 'train.fea');

        system(command);
        [ DATA, HTKCode ] = htkread( [eI.saveDir,'train.fea'] );
        DATA=DATA';
    end
end
