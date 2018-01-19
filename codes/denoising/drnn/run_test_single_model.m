function run_test_single_model
% (TODO) add
% Given a model, evaluate the performance.
    baseDir = '../../';
    addpath([baseDir, filesep, 'codes']);
    addpath([baseDir, filesep, 'tools', filesep,'bss_eval']);
    addpath([baseDir, filesep, 'tools', filesep,'bss_eval_3']);
    addpath([baseDir, filesep, 'tools', filesep,'labrosa']);

    ModelPath=[baseDir, filesep, 'codes',filesep,'mir1k', filesep, 'model_demo'];
    

    global SDR;
    global SDR_bss3;

    SDR.deviter=0;   SDR.devmax=0;   SDR.testmax=0;
    SDR.devsar=0; SDR.devsir=0; SDR.testsar=0; SDR.testsir=0;
    SDR_bss3.deviter=0;   SDR_bss3.devmax=0;   SDR_bss3.testmax=0;
    SDR_bss3.devsar=0; SDR_bss3.devsir=0; SDR_bss3.testsar=0; SDR_bss3.testsir=0;

    j=400;

    % Load model
    load([ModelPath, filesep, 'model_', num2str(j),'.mat']);
    eI.writewav=1;
    eI.bss3=1;
    eI.DataPath=[baseDir, filesep, 'codes', filesep, 'mir1k', ...
        filesep, 'Wavfile', filesep];
    eI.saveDir = [baseDir, filesep, 'codes', filesep, 'mir1k', ...
        filesep, 'model_demo', filesep, 'results', filesep];
    test_mir1k_general_kl_bss3(eI.modelname, theta, eI, 'testall', info.iteration);

    fprintf('%s\tdevmaxiter:\t%d\tdevGNSDR:\t%.3f\ttestGNSDR:\t%.3f\t',...
        eI.modelname, SDR.deviter, SDR.devmax, SDR.testmax);
    fprintf('devGSIR:\t%.3f\tdevGSAR:\t%.3f\t',SDR.devsir, SDR.devsar);
    fprintf('testGSIR:\t%.3f\ttestGSAR:\t%.3f\n',SDR.testsir, SDR.testsar);

    fprintf('%s\tbss3 devmaxiter:\t%d\tdevGNSDR:\t%.3f\ttestGNSDR:\t%.3f\t',...
        eI.modelname, SDR_bss3.deviter, SDR_bss3.devmax, SDR_bss3.testmax);
    fprintf('devGSIR:\t%.3f\tdevGSAR:\t%.3f\t',SDR_bss3.devsir, SDR_bss3.devsar);
    fprintf('testGSIR:\t%.3f\ttestGSAR:\t%.3f\n',SDR_bss3.testsir, SDR_bss3.testsar);
end
