function [ cost, grad, numTotal, pred_cell ] = drdae_obj(theta, eI, data_cell,...
                                             targets_cell, fprop_only, pred_out)
% Copyright (c) 2014-present University of Illinois at Urbana-Champaign
% All rights reserved.
% 		
% Developed by:     Po-Sen Huang, Paris Smaragdis
%                   Department of Electrical and Computer Engineering
%                   Department of Computer Science
%
%  MSE obj, no discrim training, no joint masking
%
%PRNN_OBJ MinFunc style objective for Deep Recurrent Denoising Autoencoder
%   theta is the full parameter vector
%   eI contains experiment / network architecture
%   data_cell is a cell array of matrices. Each a distinct length is a cell
%             entry. Each matrix has a time series example in each column
%   targets_cell is parallel to data, but contains the labels for each time
%   fprop_only is a flag that only computes the cost, no gradient
%   eI.recurrentOnly is a flag for computing gradients for only the
%   recurrent layer. All other gradients set to 0
%   numTotal is total number of frames evaluated
%   pred_out is a binary flag for whether pred_cell is populated
%            pred_cell only filled properly when utterances one per cell


%% Debug: Turns this into an identity-function for debugging rest of system
if isfield(eI, 'objReturnsIdentity') && eI.objReturnsIdentity
    cost = 0; grad = 0; numTotal = 0;
    for l = 1:numel(data_cell)
        numUtterances = size(data_cell{l}, 2);
        original_vector = reshape(data_cell{l}, eI.winSize*eI.featDim, []);
        midPnt = ceil(eI.winSize/2);
        original_vector = original_vector((midPnt-1)*14+1 : midPnt*14, :);
        pred_cell{l} = reshape(original_vector, [], numUtterances);
    end
    return;
end

%if isempty(return_activation),
  return_activation = 0;
%end

%% Load data from globals if not passed in (happens when run on RPC slave)
global g_data_cell;
global g_targets_cell;
isSlave = false;
if isempty(data_cell)
    data_cell = g_data_cell;
    targets_cell = g_targets_cell;
    isSlave = true;
end;
pred_cell = cell(1,numel(data_cell));
act_cell = cell(1,numel(data_cell));
%% default short circuits to false
if ~isfield(eI, 'shortCircuit')
    eI.shortCircuit = 0;
end;

%% default dropout to false
if ~isfield(eI, 'dropout')
  eI.dropout = 0;
end;

%% setup weights and accumulators
[stack, W_t] = rnn_params2stack(theta, eI);
cost = 0; numTotal = 0;
outputDim = eI.layerSizes(end);
%% setup structures to aggregate gradients
stackGrad = cell(1,numel(eI.layerSizes));
W_t_grad = zeros(size(W_t));
for l = 1:numel(eI.layerSizes)
    stackGrad{l}.W = zeros(size(stack{l}.W));
    stackGrad{l}.b = zeros(size(stack{l}.b));
end
if eI.shortCircuit
    stackGrad{end}.W_ss = zeros(size(stack{end}.W_ss));
end;
%% check options
if ~exist('fprop_only','var')
    fprop_only = false;
end;
if ~exist('pred_out','var')
    pred_out = false;
end;

% DROPOUT: vector of length of hidden layers with 0 or 1
% (to drop or keep activation unit) with prob=0.5
hActToDrop = cell(numel(eI.layerSizes-1),1);
for i=1:numel(eI.layerSizes)-1
 if eI.dropout
   hActToDrop{i} = round(rand(eI.layerSizes(i),1));
 else
   hActToDrop{i} = ones(eI.layerSizes(i),1);
 end
end

%% loop over each distinct length
for c = 1:numel(data_cell)
    data = data_cell{c};
    targets = {};
    if ~isempty(targets_cell), targets = targets_cell{c}; end;
    uttPred = [];
    T =size(data,1) / eI.inputDim;
    % store hidden unit activations at each time instant
    hAct = cell(numel(eI.layerSizes)-1, T);
    for t = 1:T
        %% forward prop all hidden layers
        for l = 1:numel(eI.layerSizes)-1
            if l == 1
                hAct{1,t} = stack{1}.W * data((t-1)*eI.inputDim+1:t*eI.inputDim, :);
            else
                hAct{l,t} = stack{l}.W * hAct{l-1,t};
            end;
            hAct{l,t} = bsxfun(@plus, hAct{l,t}, stack{l}.b);
            % temporal recurrence. limited to single layer for now
            if l == eI.temporalLayer && t > 1
                hAct{l,t} = hAct{l,t} + W_t * hAct{l,t-1};
            end;
            % nonlinearity
            if strcmpi(eI.activationFn,'tanh')
                hAct{l,t} = tanh(hAct{l,t});
            elseif strcmpi(eI.activationFn,'logistic')
                hAct{l,t} = 1./(1+exp(-hAct{l,t}));
           elseif strcmpi(eI.activationFn,'RELU')
                hAct{l,t} = max(0, hAct{l,t});
            else
                error('unrecognized activation function: %s',eI.activationFn);
            end;
            %dropout (hActToDrop will be all ones if no dropout specified)
            hAct{1,t} = bsxfun(@times, hAct{1,t}, hActToDrop{l});
        end;
        % forward prop top layer not done here to avoid caching it
    end;
    %% compute cost and backprop through time
    if  eI.temporalLayer
        delta_t = zeros(eI.layerSizes(eI.temporalLayer),size(data,2));
    end;
    for t = T:-1:1
        l = numel(eI.layerSizes);
        %% forward prop output layer for this timestep

        curPred = bsxfun(@plus, stack{l}.W * hAct{l-1,t}, stack{l}.b);

        % add short circuit to regression prediction if model has it
        if eI.shortCircuit
            curPred = curPred + stack{end}.W_ss ...
                * data((t-1)*eI.inputDim+1:t*eI.inputDim, :);
        end;
        if pred_out, uttPred = [curPred; uttPred]; end;
        % skip loss computation if no targets given
        if isempty(targets), continue; end;
        curTargets = targets((t-1)*outputDim+1:t*outputDim, :);

        if isfield(eI,'outputL1') && eI.outputL1==1, % normalize output
%              curPred_norm = zeros( size( curPred ) );
%              act = exp(curPred( 1: outputDim/ 2, :));
%              curPred_norm( 1: outputDim/ 2, :)= bsxfun(@rdivide, act, sum(act, 1)+eps );
%
%              act = exp(curPred( outputDim/2+ 1: end, :));
%              curPred_norm( outputDim/2+ 1: end, :)= bsxfun(@rdivide, act, sum(act, 1)+eps );

             maxAct = max(curPred,[], 1);
             act = exp(curPred - repmat(maxAct,size(curPred,1),1));
             curPred_norm = bsxfun(@rdivide, act, sum(act, 1)+eps );

%             curPred( outputDim/2+ 1: end, :)= curPred( outputDim/2+ 1: end, :)./ sum (curPred( outputDim/2+ 1: end/ 2, :), 2);
%              act = feat_data{nLayers} * net{nLayers}.w + repmat(net{nLayers}.recBias,batchSize,1);
%             maxAct = max(act,[], 2);
%             act = exp(act - repmat(maxAct,1,nClasses));
%             feat_data{nLayers+1} = act ./ repmat(sum(act,2),1,nClasses);
        end

        %% compute cost. Squared L2 loss
        if isfield(eI,'outputL1') && eI.outputL1==1, % normalize output
            %%%%%%%%%%%%%%%%%%%%%%MSE + L1 not working %%%%%%%%%%%%%%%%%%%
% %             delta = (curPred_norm - curTargets);
% %             cost = cost + 0.5 * sum(delta(:).^2);       % MSE
% %             delta = delta.* curPred_norm.*(1-curPred_norm);    %linear in the last layer
            %%%%%%%%%%%%%%%%%%%%%%MSE + L1 not working %%%%%%%%%%%%%%%%%%%
            delta = -(curPred_norm - curTargets);
            cost = cost + sum(sum(curTargets.*log(curPred_norm + eps) ));      % cross entropy
        else
            delta = curPred - curTargets;
            cost = cost + 0.5 * sum(delta(:).^2);
        end
        %%
        if fprop_only, continue; end;
        %% regression layer gradient and delta
        stackGrad{l}.W = stackGrad{l}.W + delta * hAct{l-1,t}';
        stackGrad{l}.b = stackGrad{l}.b + sum(delta,2);
        % short circuit layer
        if eI.shortCircuit
            stackGrad{end}.W_ss = stackGrad{end}.W_ss + delta ...
                * data((t-1)*eI.inputDim+1:t*eI.inputDim, :)';
        end;
        delta = stack{l}.W' * delta;
        %% backprop through hidden layers
        for l = numel(eI.layerSizes)-1:-1:1
            % aggregate temporal delta term if this is the recurrent layer
            if l == eI.temporalLayer
                delta = delta + delta_t;
            end;
            % push delta through activation function for this layer
            % tanh unit choice assumed
            if strcmpi(eI.activationFn,'tanh')
                delta = delta .* (1 - hAct{l,t}.^2);
            elseif strcmpi(eI.activationFn,'logistic')
                delta = delta .* hAct{l,t} .* (1 - hAct{l,t});
            elseif strcmpi(eI.activationFn,'RELU')
                delta = delta .* double(hAct{l,t}>0);
            else
                error('unrecognized activation function: %s',eI.activationFn);
            end;

            % gradient of bottom-up connection for this layer
            if l > 1
                stackGrad{l}.W = stackGrad{l}.W + delta * hAct{l-1,t}';
            else
                stackGrad{l}.W = stackGrad{l}.W + delta * data((t-1)*eI.inputDim+1:t*eI.inputDim, :)';
            end;
            % gradient for bias
            stackGrad{l}.b = stackGrad{l}.b + sum(delta,2);

            % compute derivative and delta for temporal connections
            if l == eI.temporalLayer && t > 1
                 W_t_grad = W_t_grad + delta * hAct{l,t-1}';
                 % push delta through temporal weights
                 delta_t = W_t' * delta;
            end;
            % push delta through bottom-up weights
            if l > 1
                delta = stack{l}.W' * delta;
            end;
        end
        % reduces avg memory usage but doesn't reduce peak
        %hAct(:,t) = [];
    end
    pred_cell{c} = uttPred;
    % Return the activations for this utterance.
    if return_activation,
      act_cell{c} = cell2mat(hAct);
    end
    % keep track of how many examples seen in total
    numTotal = numTotal + T * size(targets,2);
end

%% stack gradients into single vector and compute weight cost
wCost = numTotal * eI.lambda * sum(theta.^2);
grad = rnn_stack2params(stackGrad, eI, W_t_grad, true);
grad = grad + 2 * numTotal * eI.lambda * theta;

%% clipping
if isfield(eI,'clip') && eI.clip~=0, % if eI.clip==0, no clip
  if eI.clip > 0 % method one -clip the whole
      norm_grad = norm(grad);  
      fprintf('norm_grad:%f\n', norm_grad);
      % avoid numerial problem
      if norm_grad <0 || norm_grad > 1e15 || isnan(norm_grad) || isinf(norm_grad),
          grad = zeros(size(grad));
          fprintf('set gradient to zeros\n');
      end  
      if norm_grad > eI.clip 
         grad = eI.clip * grad/ norm_grad;    
      end  
  else % method two - clip each entry
      clip_value = -1*eI.clip;
      grad(grad > clip_value)=clip_value;
      grad(grad < -clip_value)=-clip_value;     
  end
end

%%
avCost = cost/numTotal;
avWCost = wCost/numTotal;
cost = cost + wCost;

%% print output
if ~isSlave && ~isempty(targets_cell)
    fprintf('loss:  %f  wCost:  %f \t',avCost, avWCost);
    fprintf('wNorm: %f  rNorm: %f  oNorm: %f\n',sum(stack{1}.W(:).^2),...
        sum(W_t(:).^2), sum(stack{end}.W(:).^2));
% plot(theta,'kx');drawnow;
end;
