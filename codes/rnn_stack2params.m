function [ params ] = rnn_stack2params( stack, eI, W_t, sum_tied )
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
%RNN_STACK2PARAMS converts stack structure of RNN weights to single vector
%   Takes a stack strcutre with stack{l}.W and stack{l}.b for each layer
%   Also takes single matrix of temporal weights W_t
%   The flag sum_tied will sum tied encoder and decoder weights.
%       This is useful for gradient aggregation
%   Verifies the stack structure conforms to their descriptions in eI
%   Namely checks eI.layerSizes, eI.inputDim, and eI.temporalLayer

%% assume no weight tieing if parameter unset
if ~isfield(eI, 'tieWeights')
    eI.tieWeights = 0;
end;
if ~exist('sum_tied','var')
    sum_tied = false;
end;
%% default short circuits to false
if ~isfield(eI, 'shortCircuit')
    eI.shortCircuit = 0;
end;  
% check short circuit consistency
assert( ~xor(eI.shortCircuit, isfield(stack{end},'W_ss')));
%% check first layer dimensions
assert( size(stack{1}.W,1) == eI.layerSizes(1));
assert( size(stack{1}.W,2) == eI.inputDim);
assert( size(stack{1}.b,1) == eI.layerSizes(1));
%% stack first layer
params = [ stack{1}.W(:); stack{1}.b(:)];
%% check and stack all layers. no special treatment of output layer
for l = 2 : numel(eI.layerSizes)
    assert( size(stack{l}.W,1) == eI.layerSizes(l));
    assert( size(stack{l}.W,2) == eI.layerSizes(l-1));
    assert( size(stack{l}.b,1) == eI.layerSizes(l));    
    if ~eI.tieWeights || (l <= numel(eI.layerSizes)/2 ...
            || l == numel(eI.layerSizes))
        % untied layer, save the weights
        if eI.tieWeights && sum_tied && l < numel(eI.layerSizes)
            % sum decoder weights if its a tied encoder layer
            lDec = numel(eI.layerSizes) - l + 1 ;
            params = [ params; reshape(stack{l}.W + stack{lDec}.W',[],1)];
        else
            params = [ params; stack{l}.W(:)];
        end;
    end;
    % always aggregate bias
    params = [ params; stack{l}.b(:)];
end
%% append temporal weight matrix
if ~isempty(W_t) || eI.temporalLayer
    if isfield(eI, 'fullRNN') && eI.fullRNN==1
        
        params = [ params; W_t{1}.W(:)];
    %% check and stack all layers. no special treatment of output layer
        for l = 2 : numel(eI.layerSizes)-1
            assert( size(W_t{l}.W,1) == eI.layerSizes(l));
            assert( size(W_t{l}.W,2) == eI.layerSizes(l));
            if ~eI.tieWeights || (l <= numel(eI.layerSizes)/2 ...
                    || l == numel(eI.layerSizes))
                % untied layer, save the weights
                if eI.tieWeights && sum_tied && l < numel(eI.layerSizes)
                    % sum decoder weights if its a tied encoder layer
                    lDec = numel(eI.layerSizes) - l + 1 ;
                    params = [ params; reshape(W_t{l}.W + W_t{lDec}.W',[],1)];
                else
                    params = [ params; W_t{l}.W(:)];
                end;
            end;
        end  
    else
        assert(size(W_t,1) == eI.layerSizes(eI.temporalLayer));
        assert(size(W_t,2) == eI.layerSizes(eI.temporalLayer));    
        params = [ params; W_t(:)];
    end    
       
end
%% append short circuit matrix
if eI.shortCircuit
    params = [params; stack{end}.W_ss(:)];
end;
