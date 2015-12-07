function test_mir1k_general_kl_bss3(modelname_in, theta, eI, stage, iter)
 % Run test and evaluation given a model.

if strcmp(stage,'testall') || strcmp(stage,'done')
	eval_types={'dev','test'};
else
	eval_types={'dev'};
end
normalize = inline('x./max(abs(x)+1e-3)');

for ieval=1:numel(eval_types)
    eval_files= dir( [eI.DataPath,filesep, eval_types{ieval}, filesep, '*wav']);

    modelname=[modelname_in, '_', eval_types{ieval}];

    % SDR
    GNSDR.no = zeros(numel(eval_files),1 );
    GNSDR.binary = zeros(numel(eval_files),1 );
    GNSDR.soft = zeros(numel(eval_files),1 );
    GNSDR.len =  zeros(numel(eval_files),1 );

    GNSDR_bss3.no = zeros(numel(eval_files),1 );
    GNSDR_bss3.binary = zeros(numel(eval_files),1 );
    GNSDR_bss3.soft = zeros(numel(eval_files),1 );
    GNSDR_bss3.len =  zeros(numel(eval_files),1 );

    % SIR
    GSIR.no = zeros(numel(eval_files),1 );
    GSIR.binary = zeros(numel(eval_files),1 );
    GSIR.soft = zeros(numel(eval_files),1 );

    GSIR_bss3.no = zeros(numel(eval_files),1 );
    GSIR_bss3.binary = zeros(numel(eval_files),1 );
    GSIR_bss3.soft = zeros(numel(eval_files),1 );

    % SAR
    GSAR.no = zeros(numel(eval_files),1 );
    GSAR.binary = zeros(numel(eval_files),1 );
    GSAR.soft = zeros(numel(eval_files),1 );

    GSAR_bss3.no = zeros(numel(eval_files),1 );
    GSAR_bss3.binary = zeros(numel(eval_files),1 );
    GSAR_bss3.soft = zeros(numel(eval_files),1 );

    for ifile=1:numel(eval_files) % each test songs
        testname=eval_files(ifile).name(1:end-4);
        if (strcmp(eval_files(ifile).name,'.') || ...
            strcmp(eval_files(ifile).name,'..')), continue; end

        [test_wav,fs]=audioread([eI.DataPath,eval_types{ieval},filesep,...
                                eval_files(ifile).name]);
        s1 = test_wav(:,2); % singing
        s2 = test_wav(:,1); % music
        eI.fs= fs;
        GNSDR.len(ifile)= size(test_wav,1);
        GNSDR_bss3.len(ifile)= size(test_wav,1);

        maxLength=max([length(s1), length(s2)]);
        s1(end+1:maxLength)=eps;
        s2(end+1:maxLength)=eps;

        s1=s1./sqrt(sum(s1.^2));
        s2=s2./sqrt(sum(s2.^2));
        mixture=s1+s2;

        winsize = eI.winsize;    nFFT = eI.nFFT;    hop = eI.hop;    scf=eI.scf;
        windows=sin(0:pi/winsize:pi-pi/winsize);

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

        Parms =  BSS_EVAL ( s1, s2, wavout_signal, wavout_noise, mixture );

        if isfield(eI,'bss3') && eI.bss3==1
            Parms_bss3 =  BSS_3_EVAL ( s1, s2, wavout_signal, wavout_noise, mixture );
        else
            Parms_bss3.SDR_bss3=0; Parms_bss3.SIR_bss3=0; Parms_bss3.SAR_bss3=0; Parms_bss3.NSDR_bss3=0;
        end

        if isfield(eI,'ioffset'),
           fprintf('%s %s %s ioffset:%d iter:%d - no mask - \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n', ...
               modelname, testname, stage, eI.ioffset, iter, Parms.SDR, Parms.SIR, Parms.SAR, Parms.NSDR);
           fprintf('%s %s %s ioffset:%d iter:%d - no mask bss3- \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n', ...
               modelname, testname, stage, eI.ioffset, iter, Parms_bss3.SDR_bss3, Parms_bss3.SIR_bss3, Parms_bss3.SAR_bss3, Parms_bss3.NSDR_bss3);

           if isfield(eI,'writewav') && eI.writewav==1
            if exist('stage','var')&& (strcmp(stage,'done')||strcmp(stage,'iter'))
            audiowrite([eI.saveDir,testname,'_ioff',num2str(eI.ioffset),'_nomask_noise.wav'], normalize(wavout_noise), fs);
            audiowrite([eI.saveDir,testname,'_ioff',num2str(eI.ioffset),'_nomask_signal.wav'], normalize(wavout_signal), fs);
            end
           end
        else % finish at once
         fprintf('%s %s %s iter:%d - no mask - \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n',...
             modelname, testname, stage, iter, Parms.SDR, Parms.SIR, Parms.SAR, Parms.NSDR);
         fprintf('%s %s %s iter:%d - no mask bss3- \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n',...
             modelname, testname, stage, iter, Parms_bss3.SDR_bss3, Parms_bss3.SIR_bss3, Parms_bss3.SAR_bss3, Parms_bss3.NSDR_bss3);

         if isfield(eI,'writewav') && eI.writewav==1
           if exist('stage','var')&& (strcmp(stage,'done')||strcmp(stage,'iter')) % not called by save_callback
            audiowrite([eI.saveDir,testname,'_iter',num2str(iter),'_nomask_noise.wav'], normalize(wavout_noise), fs);
            audiowrite([eI.saveDir,testname,'_iter',num2str(iter),'_nomask_signal.wav'], normalize(wavout_signal), fs);
           end
         end

        end

        GNSDR.no(ifile) = Parms.NSDR;
        GNSDR_bss3.no(ifile) = Parms_bss3.NSDR_bss3;

        GSIR.no(ifile) = Parms.SIR;
        GSAR.no(ifile) = Parms.SAR;
        GSIR_bss3.no(ifile) = Parms_bss3.SIR_bss3;
        GSAR_bss3.no(ifile) = Parms_bss3.SAR_bss3;

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
        if  isfield(eI,'bss3') && eI.bss3==1
            Parms_bss3 =  BSS_3_EVAL ( s1, s2, wavout_signal, wavout_noise, mixture );
        else
            Parms_bss3.SDR_bss3=0; Parms_bss3.SIR_bss3=0; Parms_bss3.SAR_bss3=0; Parms_bss3.NSDR_bss3=0;
        end

        if isfield(eI,'ioffset'),
            fprintf('%s %s %s ioffset:%d iter:%d - binary mask - \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n',...
                   modelname, testname, stage, eI.ioffset, iter, Parms.SDR, Parms.SIR, Parms.SAR, Parms.NSDR);
            fprintf('%s %s %s ioffset:%d iter:%d - binary mask bss3- \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n',...
                   modelname, testname, stage, eI.ioffset, iter, Parms_bss3.SDR_bss3, Parms_bss3.SIR_bss3, Parms_bss3.SAR_bss3, Parms_bss3.NSDR_bss3);

           if isfield(eI,'writewav') && eI.writewav==1
            if exist('stage','var')&& (strcmp(stage,'done')||strcmp(stage,'iter'))
                audiowrite([eI.saveDir,testname,'_ioff',num2str(eI.ioffset),'_bmask_noise.wav'], normalize(wavout_noise), fs);
                audiowrite([eI.saveDir,testname,'_ioff',num2str(eI.ioffset),'_bmask_signal.wav'], normalize(wavout_signal), fs);
            end
           end
        else % finish at once
            fprintf('%s %s %s iter:%d - binary mask - \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n', ...
                modelname, testname, stage, iter, Parms.SDR, Parms.SIR, Parms.SAR, Parms.NSDR);
            fprintf('%s %s %s iter:%d - binary mask bss3- \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n', ...
                modelname, testname, stage, iter,  Parms_bss3.SDR_bss3, Parms_bss3.SIR_bss3, Parms_bss3.SAR_bss3, Parms_bss3.NSDR_bss3);

            if isfield(eI,'writewav') && eI.writewav==1
               if exist('stage','var')&& (strcmp(stage,'done')||strcmp(stage,'iter')) % not called by save_callback
                 audiowrite([eI.saveDir,testname,'_iter',num2str(iter),'_bmask_noise.wav'], normalize(wavout_noise), fs);
                 audiowrite([eI.saveDir,testname,'_iter',num2str(iter),'_bmask_signal.wav'], normalize(wavout_signal), fs);
               end
            end
        end

        GNSDR.binary(ifile) = Parms.NSDR;
        GNSDR_bss3.binary(ifile) = Parms_bss3.NSDR_bss3;

        GSIR.binary(ifile) = Parms.SIR;
        GSAR.binary(ifile) = Parms.SAR;
        GSIR_bss3.binary(ifile) = Parms_bss3.SIR_bss3;
        GSAR_bss3.binary(ifile) = Parms_bss3.SAR_bss3;

        %% softmask
        gain=1;
        m= double(abs(pred_source_signal)./(abs(pred_source_signal)+ (gain*abs(pred_source_noise))+eps));

        source_signal =m .*spectrum.mix;
        source_noise= spectrum.mix-source_signal;

        wavout_noise = istft(source_noise, nFFT ,windows, hop)';
        wavout_signal = istft(source_signal, nFFT ,windows, hop)';

        Parms =  BSS_EVAL ( s1, s2, wavout_signal, wavout_noise, mixture );
        if  isfield(eI,'bss3') && eI.bss3==1
            Parms_bss3 =  BSS_3_EVAL ( s1, s2, wavout_signal, wavout_noise, mixture );
        else
            Parms_bss3.SDR_bss3=0; Parms_bss3.SIR_bss3=0; Parms_bss3.SAR_bss3=0; Parms_bss3.NSDR_bss3=0;
        end

        if isfield(eI,'ioffset'),
            fprintf('%s %s %s ioffset:%d iter:%d - soft mask - \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n', ...
                modelname, testname, stage, eI.ioffset, iter, Parms.SDR, Parms.SIR, Parms.SAR, Parms.NSDR);
            fprintf('%s %s %s ioffset:%d iter:%d - soft mask bss3- \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n', ...
                modelname, testname, stage, eI.ioffset, iter,Parms_bss3.SDR_bss3, Parms_bss3.SIR_bss3, Parms_bss3.SAR_bss3, Parms_bss3.NSDR_bss3);

              if isfield(eI,'writewav') && eI.writewav==1
                if exist('stage','var')&& (strcmp(stage,'done')||strcmp(stage,'iter'))
                  audiowrite([eI.saveDir,testname,'_ioff',num2str(eI.ioffset),'_softmask_noise.wav'], normalize(wavout_noise), fs);
                  audiowrite([eI.saveDir,testname,'_ioff',num2str(eI.ioffset),'_softmask_signal.wav'], normalize(wavout_signal), fs);
                end
              end
        else % finish at once
            fprintf('%s %s %s iter:%d - soft mask - \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n', ...
                modelname, testname, stage, iter, Parms.SDR, Parms.SIR, Parms.SAR, Parms.NSDR);
            fprintf('%s %s %s iter:%d - soft mask bss3- \tSDR:%.3f\tSIR:%.3f\tSAR:%.3f\tNSDR:%.3f\n', ...
                modelname, testname, stage, iter, Parms_bss3.SDR_bss3, Parms_bss3.SIR_bss3, Parms_bss3.SAR_bss3, Parms_bss3.NSDR_bss3);

            if isfield(eI,'writewav') && eI.writewav==1
               if exist('stage','var')&& (strcmp(stage,'done')||strcmp(stage,'iter')) % not called by save_callback
                 audiowrite([eI.saveDir,testname,'_iter',num2str(iter),'_softmask_noise.wav'], normalize(wavout_noise), fs);
                 audiowrite([eI.saveDir,testname,'_iter',num2str(iter),'_softmask_signal.wav'], normalize(wavout_signal), fs);
               end
            end
        end

        GNSDR.soft(ifile) = Parms.NSDR;
        GNSDR_bss3.soft(ifile) = Parms_bss3.NSDR_bss3;


        GSIR.soft(ifile) = Parms.SIR;
        GSAR.soft(ifile) = Parms.SAR;
        GSIR_bss3.soft(ifile) = Parms_bss3.SIR_bss3;
        GSAR_bss3.soft(ifile) = Parms_bss3.SAR_bss3;

    end % for ifile=1:numel(train_files)

    GNSDR_no  = sum(GNSDR.no.*GNSDR.len)/sum(GNSDR.len);
    GNSDR_binary = sum(GNSDR.binary.*GNSDR.len)/sum(GNSDR.len);
    GNSDR_soft = sum(GNSDR.soft.*GNSDR.len)/sum(GNSDR.len);

    GNSDR_no_bss3  = sum(GNSDR_bss3.no.*GNSDR_bss3.len)/sum(GNSDR_bss3.len);
    GNSDR_binary_bss3 = sum(GNSDR_bss3.binary.*GNSDR_bss3.len)/sum(GNSDR_bss3.len);
    GNSDR_soft_bss3 = sum(GNSDR_bss3.soft.*GNSDR_bss3.len)/sum(GNSDR_bss3.len);

    % SIR
    GSIR_no  = sum(GSIR.no.*GNSDR.len)/sum(GNSDR.len);
    GSIR_binary = sum(GSIR.binary.*GNSDR.len)/sum(GNSDR.len);
    GSIR_soft = sum(GSIR.soft.*GNSDR.len)/sum(GNSDR.len);

    GSIR_no_bss3  = sum(GSIR_bss3.no.*GNSDR_bss3.len)/sum(GNSDR_bss3.len);
    GSIR_binary_bss3 = sum(GSIR_bss3.binary.*GNSDR_bss3.len)/sum(GNSDR_bss3.len);
    GSIR_soft_bss3 = sum(GSIR_bss3.soft.*GNSDR_bss3.len)/sum(GNSDR_bss3.len);

    % SAR
    GSAR_no  = sum(GSAR.no.*GNSDR.len)/sum(GNSDR.len);
    GSAR_binary = sum(GSAR.binary.*GNSDR.len)/sum(GNSDR.len);
    GSAR_soft = sum(GSAR.soft.*GNSDR.len)/sum(GNSDR.len);

    GSAR_no_bss3  = sum(GSAR_bss3.no.*GNSDR_bss3.len)/sum(GNSDR_bss3.len);
    GSAR_binary_bss3 = sum(GSAR_bss3.binary.*GNSDR_bss3.len)/sum(GNSDR_bss3.len);
    GSAR_soft_bss3 = sum(GSAR_bss3.soft.*GNSDR_bss3.len)/sum(GNSDR_bss3.len);


    fprintf('\n');
    if isfield(eI,'ioffset'),
        fprintf('%s %s ioffset:%d iter:%d - no mask - \tGNSDR:%.3f\tGSIR:%.3f\tGSAR:%.3f\n', ...
            modelname, stage, eI.ioffset, iter, GNSDR_no, GSIR_no, GSAR_no);
        fprintf('%s %s ioffset:%d iter:%d - binary mask - \tGNSDR:%.3f\tGSIR:%.3f\tGSAR:%.3f\n',...
            modelname, stage, eI.ioffset, iter, GNSDR_binary, GSIR_binary, GSAR_binary);
        fprintf('%s %s ioffset:%d iter:%d - soft mask - \tGNSDR:%.3f\tGSIR:%.3f\tGSAR:%.3f\n',...
            modelname, stage, eI.ioffset, iter, GNSDR_soft, GIR_soft, GSAR_soft);
    else % finish at once
        fprintf('%s %s iter:%d - no mask - \tGNSDR:%.3f\tGSIR:%.3f\tGSAR:%.3f\n', ...
            modelname, stage, iter, GNSDR_no, GSIR_no, GSAR_no);
        fprintf('%s %s iter:%d - binary mask - \tGNSDR:%.3f\tGSIR:%.3f\tGSAR:%.3f\n',...
            modelname, stage, iter, GNSDR_binary, GSIR_binary, GSAR_binary);
        fprintf('%s %s iter:%d - soft mask - \tGNSDR:%.3f\tGSIR:%.3f\tGSAR:%.3f\n',...
            modelname, stage, iter, GNSDR_soft, GSIR_soft, GSAR_soft);
    end

    if isfield(eI,'ioffset'),
        fprintf('%s %s ioffset:%d iter:%d - no mask bss3- \tGNSDR:%.3f\tGSIR:%.3f\tGSAR:%.3f\n', ...
            modelname, stage, eI.ioffset, iter, GNSDR_no_bss3, GSIR_no_bss3, GSAR_no_bss3);
        fprintf('%s %s ioffset:%d iter:%d - binary mask bss3- \tGNSDR:%.3f\tGSIR:%.3f\tGSAR:%.3f\n',...
            modelname, stage, eI.ioffset, iter, GNSDR_binary_bss3, GSIR_binary_bss3, GSAR_binary_bss3);
        fprintf('%s %s ioffset:%d iter:%d - soft mask bss3- \tGNSDR:%.3f\tGSIR:%.3f\tGSAR:%.3f\n',...
            modelname, stage, eI.ioffset, iter, GNSDR_soft_bss3, GSIR_soft_bss3, GSAR_soft_bss3);
    else % finish at once
        fprintf('%s %s iter:%d - no mask bss3- \tGNSDR:%.3f\tGSIR:%.3f\tGSAR:%.3f\n', ...
            modelname, stage, iter, GNSDR_no_bss3, GSIR_no_bss3, GSAR_no_bss3);
        fprintf('%s %s iter:%d - binary mask bss3- \tGNSDR:%.3f\tGSIR:%.3f\tGSAR:%.3f\n',...
            modelname, stage, iter, GNSDR_binary_bss3, GSIR_binary_bss3, GSAR_binary_bss3);
        fprintf('%s %s iter:%d - soft mask bss3- \tGNSDR:%.3f\tGSIR:%.3f\tGSAR:%.3f\n',...
            modelname, stage, iter, GNSDR_soft_bss3, GSIR_soft_bss3, GSAR_soft_bss3);
    end

   %% record max dev soft SDR: dev/test SDR, iter
    global SDR;
    global SDR_bss3;

    if ieval==1
        if GNSDR_soft > SDR.devmax
            SDR.devmax=GNSDR_soft;
            SDR.deviter=iter;

            SDR.devsar=GSAR_soft;
            SDR.devsir=GSIR_soft;
        end

        if GNSDR_soft_bss3 > SDR_bss3.devmax
            SDR_bss3.devmax=GNSDR_soft_bss3;
            SDR_bss3.deviter=iter;

            SDR_bss3.devsar=GSAR_soft_bss3;
            SDR_bss3.devsir=GSIR_soft_bss3;
        end
    else
        if SDR.deviter==iter
            SDR.testmax=GNSDR_soft;

            SDR.testsar=GSAR_soft;
            SDR.testsir=GSIR_soft;
        end
        if SDR_bss3.deviter==iter
            SDR_bss3.testmax=GNSDR_soft_bss3;

            SDR_bss3.testsar=GSAR_soft_bss3;
            SDR_bss3.testsir=GSIR_soft_bss3;
        end
    end
end % eval_type -dev test

return;

%% unit test
% (TODO) add
savedir='results';
iter=200;
modelname='model_test';
load([savedir,modelname, filesep, 'model_',num2str(iter),'.mat']);
test_mir1k_general_kl_bss3(modelname_in, theta, eI, stage, iter);
end
