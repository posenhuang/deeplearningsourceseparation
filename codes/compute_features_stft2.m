function [DATA, mixture_spectrum, eI] = compute_features_stft2(dmix, eI)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
% compute features
% Spectra, log power spectra, MFCC, logmel
  nFFT = eI.nFFT; hop = eI.hop; scf = eI.scf;
	wn = sqrt( hann( nFFT, 'periodic')); % hann window
    
  if eI.MFCCorlogMelorSpectrum==2 || eI.MFCCorlogMelorSpectrum==3
    spectrum_mix = scf * stft2( dmix, nFFT, hop, 0, wn);
  else 
      spectrum_mix = scf * stft3( dmix, nFFT, hop, 0, wn, [], 0);
  end

  mixture_spectrum=abs(spectrum_mix);

  if eI.MFCCorlogMelorSpectrum==2, %Spectrum  
      DATA = scf * stft2( dmix, nFFT, hop, 0, wn);
      DATA=abs(DATA);
  elseif eI.MFCCorlogMelorSpectrum==3, %log power spectrum
      DATA = scf * stft2( dmix, nFFT, hop, 0, wn);
      DATA=abs(DATA);
      DATA=log(DATA.*DATA+eps);
  else
      %% training features
      filename=[eI.saveDir,'dmix_temp.wav'];

      dmix=dmix./sqrt(sum(dmix.^2));        
      wavwrite(dmix, eI.fs, filename);

      if eI.framerate==64,
          if eI.MFCCorlogMelorSpectrum==0, %MFCC
               eI.config='mfcc_64ms_step16ms.cfg';
          elseif eI.MFCCorlogMelorSpectrum==1, %logmel
              eI.config='fbank_64ms_step16ms.cfg';
          else % spectrum
              eI.config='spectrum_64ms_step32ms.cfg';
          end
      else % framerate == 32
          if eI.MFCCorlogMelorSpectrum==0, %MFCC
              eI.config='mfcc_32ms_step8ms.cfg';
          elseif eI.MFCCorlogMelorSpectrum==1, %logmel
              eI.config='fbank_32ms_step8ms.cfg';
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