% Import values from backpropagation value file
run("SM_Backprop_Values.m")

% Backpropagate through softmax then penultimate layer, return gradient of cross-entropy loss w.r.t. data
% NOTE: label is either 0 or 1 for indexing...
function der_L_wrt_x = linear_layer_gradient (weight, bias, feature_vector, label, use_double)

% Toggle double, everything is naively in floats...
if (use_double)
 weight = double(weight);
 bias = double(bias);
 feature_vector = double(feature_vector);
endif

% Print fixed attributes
disp("Feature Vector: ")
disp(feature_vector)

disp("Dimension of weight: ")
disp(size(weight))

disp("Bias: ")
disp(bias)

% Compute output of linear layer by multiplying weight and adding bias 
z = feature_vector * weight' + bias

% Compute softmax 
denom = sum(exp(z));  
y_tilde = exp(z) / denom;
disp("Softmax Confidence: ")
disp(y_tilde);
disp("");

% Hard-set confidence as PyTorch returns it
%y_tilde = [1.0000e+00, 3.3631e-43];  % For zero-gradient vote swatch
%y_tilde = [1.0000e+00; 1.3056e-16];  % For non-zero gradient vote swatch 

% Assuming confidence vector is already processed through softmax, get difference 
% NOTE: To see that using soft labels (i.e. [0.9, 0.0] instead of [1.0, 0.0]) alleviates gradient masking, replace label matrix here 
if (label+1 == 1)
 label_matrix = [1.0, 0.0];
else 
 label_matrix = [0.0, 1.0];
endif

% Get loss with respect to example
L_wrt_x = -sum(label * y_tilde);

% Get gradient with respect to softmax logits
der_L_wrt_z = y_tilde - label_matrix; 
disp("Gradient of Loss w.r.t. softmax logits: ")
disp(der_L_wrt_z);
disp("")

% Gradient of feature neuron w.r.t. current example is just our weight matrix!
der_z_wrt_x = weight';

% Gradients with respect to weights
der_L_wrt_w = z' * der_L_wrt_z;

% Calculate loss w.r.t. example 
der_L_wrt_x = der_z_wrt_x .* der_L_wrt_z;
%der_L_wrt_x = der_L_wrt_z .* der_z_wrt_x;
disp("Gradient of Loss w.r.t. Feature Vector: ");
disp(size(der_L_wrt_x));
disp(der_L_wrt_x);

endfunction

% Compute gradient with respect to loss
der_L_wrt_x = linear_layer_gradient(lin_layer_weights, lin_layer_biases, vote_swatch_non_zero_grad_feature, 0, true);
