function [ stack, W_t ] = initialize_weights( eI )
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
%INITIALIZE_WEIGHTS Random weight structures for a network architecture
%   eI describes an RNN via the fields layerSizes, inputDim and
%   temporalLayer
%
%   This uses Xavier's weight initialization tricks for better backprop
%   See: X. Glorot, Y. Bengio. Understanding the difficulty of training
%        deep feedforward neural networks. AISTATS 2010.

%% initialize hidden layers
stack = cell(1, numel(eI.layerSizes));
for l = 1 : numel(eI.layerSizes)
    if l > 1
        prevSize = eI.layerSizes(l-1);
    else
        prevSize = eI.inputDim;
    end;
    curSize = eI.layerSizes(l);
    % Xaxier's scaling factor
    s = sqrt(6) / sqrt(prevSize + curSize);
    % Ilya suggests smaller scaling for recurrent layer
    if (~isfield(eI, 'fullRNN') && l == eI.temporalLayer) ||...
        (isfield(eI, 'fullRNN') && eI.fullRNN==1 && l< numel(eI.layerSizes))
         s = sqrt(6) / sqrt(prevSize + 2*curSize);
    end;
    stack{l}.W = rand(curSize, prevSize)*2*s - s;
    stack{l}.b = zeros(curSize, 1);
end
%% weight tying
% default weight tying to false
if ~isfield(eI, 'tieWeights')
    eI.tieWeights = 0;
end;
% overwrite decoder layers for tied weights
if eI.tieWeights
    decList = [(numel(eI.layerSizes)/2)+1 : numel(eI.layerSizes)-1];
    for l = 1:numel(decList)
        lDec = decList(l);
        lEnc = decList(1) - l;
        assert( norm(size(stack{lEnc}.W') - size(stack{lDec}.W)) == 0, ...
            'Layersizes dont match for tied weights');
        stack{lDec}.W = stack{lEnc}.W';
    end;
end;
%% initialize temporal weights if they should exist
if isfield(eI, 'fullRNN') && eI.fullRNN==1
    W_t = cell(1, numel(eI.layerSizes)-1);
else
    W_t = [];
end
if eI.temporalLayer
    if isfield(eI, 'fullRNN') && eI.fullRNN==1
        for  l = 1 : numel(eI.layerSizes)-1
            % assuems temporal init type set
            if strcmpi(eI.temporalInit, 'zero')
                W_t{l}.W = zeros(eI.layerSizes(l));
            elseif strcmpi(eI.temporalInit, 'rand')
                % Ilya's modification to Xavier's update rule
                s = sqrt(6) / sqrt(3*eI.layerSizes(l));
                W_t{l}.W = rand(eI.layerSizes(l))*2*s - s;
            elseif strcmpi(eI.temporalInit, 'eye')
                W_t{l}.W = eye(eI.layerSizes(l));
            else
                error('unrecognized temporal initialization: %s', eI.temporalInit);
            end;
        end
    else
        % assuems temporal init type set
        if strcmpi(eI.temporalInit, 'zero')
            W_t = zeros(eI.layerSizes(eI.temporalLayer));
        elseif strcmpi(eI.temporalInit, 'rand')
            % Ilya's modification to Xavier's update rule
            s = sqrt(6) / sqrt(3*eI.layerSizes(eI.temporalLayer));
            W_t = rand(eI.layerSizes(eI.temporalLayer))*2*s - s;
        elseif strcmpi(eI.temporalInit, 'eye')
            W_t = eye(eI.layerSizes(eI.temporalLayer));
        else
            error('unrecognized temporal initialization: %s', eI.temporalInit);
        end;
    end

end;
%% init short circuit connections
% default short circuits to false
if ~isfield(eI, 'shortCircuit')
    eI.shortCircuit = 0;
end;
if eI.shortCircuit
    % use random init since input might contain noise estimate
    s = sqrt(6) / sqrt(eI.inputDim + eI.layerSizes(end));
    stack{end}.W_ss = rand(eI.inputDim, eI.layerSizes(end))*2*s - s;
end;



