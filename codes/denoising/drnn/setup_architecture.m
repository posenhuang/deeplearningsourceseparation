% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
eI = [];
% 0- mfcc, 1- logmel, 2- spectrum
eI.MFCCorlogMelorSpectrum=MFCCorlogMelorSpectrum;
eI.CFGPath=CFGPath;
eI.seqLen = [1 10 25 50 100];

eI.framerate=framerate;
if eI.framerate==64
    eI.winsize = 1024;    eI.nFFT = 1024;    eI.hop =eI.winsize/4;    eI.scf=1;
%     eI.winsize = 1024;    eI.nFFT = 1024;    eI.hop =eI.winsize/2;    eI.scf=1;
else %32
    eI.winsize = 512;    eI.nFFT = 512;    eI.hop = eI.winsize/4;    eI.scf=1;
%     eI.winsize = 512;    eI.nFFT = 512;    eI.hop = eI.winsize/2;    eI.scf=1;
end
winsize=eI.winsize; nFFT= eI.nFFT; hop= eI.hop; scf=eI.scf;
% windows=sin(0:pi/winsize:pi-pi/winsize);

% single target or multiple targets
eI.cleanonly=iscleanonly;

% context window size of the input.
eI.num_contextwin = context_win;

% dimension of each input frame
if eI.MFCCorlogMelorSpectrum==0 %0 for mfcc 1 for logmel
    eI.featDim =39;
elseif eI.MFCCorlogMelorSpectrum==1 %0 for mfcc 1 for logmel
    eI.featDim =123;
else
    eI.featDim = (eI.nFFT/2+1);
end

eI.dropout = isdropout;

% weight tying in hidden layers
% if you want tied weights, must have odd number of *hidden* layers
eI.tieWeights = 0;

eI.const=const; eI.const2=const2;

hidden_units_set=[];

for il=1:num_layers
    hidden_units_set=[hidden_units_set, hidden_units];
end
% 2 hidden layers and output layer
if eI.cleanonly==1,
    eI.layerSizes = [hidden_units_set  nFFT/2+1 ];
else
    eI.layerSizes = [hidden_units_set  (nFFT/2+1)*2];
end
% highest hidden layer is temporal
eI.temporalLayer =isRNN;
% dim of network input at each timestep (final size after window & whiten)
eI.inputDim = eI.featDim * eI.num_contextwin;
% length of input sequence chunks.
% activation functionbfgs_iter
switch act
    case 0
        eI.activationFn = 'logistic';
    case 1
        eI.activationFn = 'tanh';
    case 2
        eI.activationFn = 'RELU';
end

% temporal initialization type
eI.temporalInit = 'rand';
% weight norm penaly
% eI.lambda = 0;
eI.lambda = wdecay;
% file containing whitening matrices for outputs
eI.outputL1=0;

eI.inputL1=isinputL1;

eI.r=pos_neg_r;

eI.isdiscrim=2;

if opt==0,
    eI.opt='softlinear';
elseif opt==1,
    eI.opt='softabs';
elseif opt==2,
    eI.opt='softquad';
elseif opt==3,
    eI.opt='softabs_const';
elseif opt==4,
    eI.opt='softabs_kl_const';
end

eI.train_mode= train_mode;
eI.outputnonlinear=outputnonlinear;

if exist('grad_clip','var'),
    eI.clip = grad_clip;
end
%% setup weight caching
if isRNN,
    if isRNN==num_layers+1
        modelname=['model_RNNall'];
        eI.fullRNN=1;
    else
        modelname=['model_RNN',num2str(isRNN)];
    end
else
    modelname='model_DNN';
end
modelname=[modelname,'_win',num2str(context_win),'_h', num2str(hidden_units),'_l',num2str(num_layers)];
if iscleanonly, modelname=[modelname,'_cleanonly']; end
if isdropout,  modelname=[modelname,'_dropout',num2str(isdropout)];    end
modelname=[modelname,['_r', num2str(eI.r)]];

modelname=[modelname,['_', num2str(eI.framerate),'ms']];
modelname=[modelname, '_', num2str(circular_step)];
eI.circular_step = circular_step;

modelname=[modelname,'_',eI.opt];
if outputnonlinear==0, modelname=[modelname,'_linearout']; end

modelname=[modelname, '_', eI.activationFn];

if eI.inputL1, modelname=[modelname, '_L',num2str(eI.inputL1)]; end

if eI.MFCCorlogMelorSpectrum==0
    modelname=[modelname,'_mfcc'];
elseif  eI.MFCCorlogMelorSpectrum==1
    modelname=[modelname,'_logmel'];
elseif  eI.MFCCorlogMelorSpectrum==2
    modelname=[modelname,'_spectrum'];
else
    modelname=[modelname,'_logpowspect'];
end

modelname= [modelname, '_trn', num2str(eI.train_mode)];
modelname=[modelname,'_c',num2str(const), '_c',num2str(const2)];

if ~isempty(batchsize) && ~isempty(lbfgs_iter)
  modelname=[modelname,'_bsz',num2str(batchsize), '_miter',num2str(max_iter), '_bf',num2str(lbfgs_iter)];   
end
if exist('grad_clip','var') && grad_clip~=0, 
   modelname = [modelname, '_c', num2str(grad_clip)]; 
end

if eI.lambda~=0,  modelname = [modelname, '_w', num2str(eI.lambda)]; end
    
eI.modelname=modelname;
disp(modelname);

eI.saveDir = [saveDir, filesep, modelname, filesep];
if ~exist(eI.saveDir,'dir'), mkdir(eI.saveDir); end
