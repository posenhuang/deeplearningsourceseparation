function n = drnn( x, y, z, context_win, hidden_units, num_layers, isdropout, ...
        isRNN, iscleanonly, circular_step , isinputL1, MFCCorlogMelorSpectrum, ...
        framerate, pos_neg_r, outputnonlinear, opt, act, train_mode, const,  ...
        const2, isGPU, max_iter, batchsize, lbfgs_iter, clip, wdecay)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
% Feed-forward regression neural net
%  function n = nn6r( x, y, l, sg, ep, h, pl, et)
%
% Training:
%  x    is the data input (wav)
%  y    is the desired output1 (wav)
%  z    is the desired output2 (wav)
%  l    is the number of units in the hidden layer
%  sg   are the activation functions to use
%  iter is the number of epochs to train
%  h    is the learnign rate setup ([init rate])
%  pl   is the plotting flag (default = false)
%  et   is the error type to use ('e', 'kl1', 'kl2', 'is')
%  n    is the classifier data structure
%
% Simulation:
%  x are the input data
%  y is the classifier structure
%  n are the network outputs


% Forward pass
if isstruct(y)
    % run test
    eI = y.eI;
    theta = y.theta;
    
    n = test_denoising_general_kl_bss3(...
        x, theta, eI, 'testall', 0);
elseif ~isempty(batchsize) && ~isempty(lbfgs_iter)
    n = train_denoising_mini(y,z,context_win, hidden_units, num_layers, isdropout, ...
        isRNN, iscleanonly, circular_step , isinputL1, MFCCorlogMelorSpectrum, ...
        framerate, pos_neg_r, outputnonlinear, opt, act, train_mode, const,  ...
        const2, isGPU, max_iter, batchsize, lbfgs_iter, clip, wdecay);
else  
    n = train_denoising(y,z,context_win, hidden_units, num_layers, isdropout, ...
        isRNN, iscleanonly, circular_step , isinputL1, MFCCorlogMelorSpectrum, ...
        framerate, pos_neg_r, outputnonlinear, opt, act, train_mode, const,  ...
        const2, isGPU, max_iter, batchsize, lbfgs_iter, clip);
end