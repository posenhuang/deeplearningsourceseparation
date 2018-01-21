function run_test_single_model
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
% Given a model, evaluate the performance.
    baseDir = '../../../';
    addpath([baseDir, filesep, 'codes']);
    addpath([baseDir, filesep, 'codes', filesep, 'denoising']);

    addpath([baseDir, filesep, 'tools', filesep,'bss_eval']);
    addpath([baseDir, filesep, 'tools', filesep,'bss_eval_3']);
    addpath([baseDir, filesep, 'tools', filesep,'labrosa']);

    ModelPath=[baseDir, filesep, 'codes',filesep,'denoising', filesep, 'demo'];
    
    global SDR;
    global SDR_bss3;

    SDR.deviter=0;   SDR.devmax=0;   SDR.testmax=0;
    SDR.devsar=0; SDR.devsir=0; SDR.testsar=0; SDR.testsir=0;
    SDR_bss3.deviter=0;   SDR_bss3.devmax=0;   SDR_bss3.testmax=0;
    SDR_bss3.devsar=0; SDR_bss3.devsir=0; SDR_bss3.testsar=0; SDR_bss3.testsir=0;

    j=870;
    
    % Load model
    load([ModelPath, filesep, 'denoising_model_', num2str(j),'.mat']);
    eI.saveDir = [baseDir, filesep, 'codes', filesep, 'denoising', ...
        filesep, 'demo', filesep, 'results', filesep];
    %%
    index = 2;
    [speech, fs] = audioread(['wav', filesep, 'original_speech', num2str(index), '.wav']);
    [noise, fs] = audioread(['wav', filesep, 'original_noise',num2str(index),'.wav']);

    x = speech + noise;    
    eI.fs = fs;
    %%
    output = test_denoising_general_kl_bss3(x', theta, eI, 'testall', 0);
    %%
  	sz = 1024.*[1 1/4];
    wn = sqrt( hann( sz(1), 'periodic')); % hann window
    wav_singal = stft2( output.source_signal, sz(1), sz(2), 0, wn);
    wav_noise = stft2( output.source_noise, sz(1), sz(2), 0, wn);
    wav_singal = wav_singal./max(abs(wav_singal));
    wav_noise = wav_noise./max(abs(wav_noise));    

    audiowrite([eI.saveDir, filesep,'separated_speech',num2str(index),'.wav'], wav_singal, fs);
    audiowrite([eI.saveDir, filesep,'separated_noise',num2str(index),'.wav'], wav_noise, fs);
    
    % Get separation stats
    [sdr,sir,sar,stoi] = sep_perf(wav_singal, [speech'; noise'], fs);    
end
