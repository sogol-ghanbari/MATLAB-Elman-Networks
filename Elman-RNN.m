%% Elman Recurrent Neural Network for Non-linear Dynamic System Identification
% Implementation from Scratch using Chain Rule and Gradient Descent
% Author: Sogol Ghanbari

clc;
close all;
clear all;

%% 1. Non-linear Dynamic System Dataset Generation
num_samples = 1003;
u = rand(1, num_samples) - 0.5;   % Control input signal: u(k) in [-0.5, 0.5]
y_plant = zeros(1, num_samples);

% Plant difference equation
for k = 3:num_samples - 1
    x1 = y_plant(k);
    x2 = y_plant(k-1);
    x3 = y_plant(k-2);
    x4 = u(k);
    x5 = u(k-1);
    
    y_plant(k+1) = (x1 * x2 * x3 * x5 * (x3 - 1) + x4) / (1 + x2^2 + x3^3);
end

% Input patterns and corresponding targets
inputs_all  = [y_plant(3:1002); y_plant(2:1001); y_plant(1:1000); u(3:1002); u(2:1001)];
targets_all = y_plant(4:1003);

%% 2. Network Architecture & Data Partitioning
num_train = 700;
num_test  = 200;
num_check = 100;

num_inputs = 5;
num_hidden = 20;
num_output = 1;

num_epochs    = 100;
learning_rate = 0.01;

% Training, Testing, and Checking Splits
target_train = targets_all(1:num_train);
input_train  = inputs_all(:, 1:num_train);

target_test  = targets_all(num_train + 1 : num_train + num_test);
input_test   = inputs_all(:, num_train + 1 : num_train + num_test);

target_check = targets_all(num_train + num_test + 1 : end);
input_check  = inputs_all(:, num_train + num_test + 1 : end);

%% 3. Weight & Bias Initialization
weights_input_hidden  = rand(num_hidden, num_inputs) * 2 - 1;
bias_hidden           = zeros(num_hidden, 1);

weights_hidden_output = rand(num_output, num_hidden) * 2 - 1;
bias_output           = zeros(num_output, 1);

weights_recurrent     = rand(num_hidden, num_hidden) * 2 - 1; % Recurrent (context) weights

%% 4. Training Phase
train_mse_history = [];
test_mse_history  = zeros(1, num_epochs);
train_errors_summary = [];
test_errors_summary  = [];

figure;

for epoch = 1:num_epochs
    train_errors_epoch = [];
    output_train_epoch = [];
    
    % Previous recurrent states: o^1(t-1) and o^1(t-2)
    hidden_out_prev  = zeros(num_hidden, 1);
    hidden_out_prev2 = zeros(num_hidden, 1);
    
    for s = 1:num_train
        %% Forward Pass
        sample_input = input_train(:, s);
        net_hidden   = weights_input_hidden * sample_input + weights_recurrent * hidden_out_prev + bias_hidden;
        hidden_out   = tanh(net_hidden);         % Activation function: tanh / tansig
        
        net_output   = weights_hidden_output * hidden_out + bias_output;
        y_hat        = net_output;               % Linear activation: purelin
        
        sample_error = target_train(s) - y_hat;
        train_errors_epoch = [train_errors_epoch, sample_error];
        output_train_epoch = [output_train_epoch, y_hat];
        
        %% Backpropagation via Chain Rule
        sigma       = (weights_hidden_output') * sample_error;
        fprime      = 1 - hidden_out.^2;         % Derivative f'(t)
        fprime_prev = 1 - hidden_out_prev.^2;    % Derivative f'(t-1)
        
        %% Output Layer Updates
        weights_hidden_output = weights_hidden_output + learning_rate * sample_error * hidden_out';
        bias_output           = bias_output + learning_rate * sample_error;
        
        %% Hidden Layer Updates
        for j = 1:num_hidden
            delta_j = sigma(j) * fprime(j);
            weights_input_hidden(j, :) = weights_input_hidden(j, :) + learning_rate * delta_j * sample_input';
            bias_hidden(j)             = bias_hidden(j) + learning_rate * delta_j;
        end
        
        %% Recurrent Layer Updates (Chain Rule expansion)
        for j = 1:num_hidden
            for c = 1:num_hidden
                d_jc = hidden_out_prev(c) + weights_recurrent(j, c) * fprime_prev(j) * hidden_out_prev2(c);
                weights_recurrent(j, c) = weights_recurrent(j, c) + learning_rate * sigma(j) * fprime(j) * d_jc;
            end
        end
        
        %% Shift Context States
        hidden_out_prev2 = hidden_out_prev;
        hidden_out_prev  = hidden_out;
    end
    
    % Record training MSE
    train_mse_history = [train_mse_history, mse(train_errors_epoch)];
    
    %% Validation / Test Phase for Current Epoch
    test_errors_epoch = [];
    output_test_epoch = [];
    hidden_out_prev_test = zeros(num_hidden, 1);
    
    for i = 1:length(input_test)
        sample_test_input = input_test(:, i);
        net_hidden_test   = weights_input_hidden * sample_test_input + weights_recurrent * hidden_out_prev_test + bias_hidden;
        hidden_out_test   = tansig(net_hidden_test);
        
        net_output_test   = weights_hidden_output * hidden_out_test + bias_output;
        output_test_val   = purelin(net_output_test);
        
        test_errors_epoch = [test_errors_epoch, target_test(i) - output_test_val];
        output_test_epoch = [output_test_epoch, output_test_val];
        
        hidden_out_prev_test = hidden_out_test;
    end
    
    test_mse_history(epoch) = mse(test_errors_epoch);
    
    %% Live Plotting
    subplot(2, 1, 1);
    plot(train_mse_history);
    title('MSE error for training data');
    
    subplot(2, 1, 2);
    plot(test_mse_history);
    title('MSE error for test data');
    drawnow;
end

% Final Error Evaluations
final_train_mse = mse(train_errors_epoch)
final_test_mse  = mse(test_errors_epoch)

%% 5. Checking Phase (Independent Check Set)
check_errors_list = [];
output_check_list = [];
hidden_out_prev_check = zeros(num_hidden, 1);

for i = 1:length(input_check)
    sample_check_input = input_check(:, i);
    net_hidden_check   = weights_input_hidden * sample_check_input + weights_recurrent * hidden_out_prev_check + bias_hidden;
    hidden_out_check   = tansig(net_hidden_check);
    
    net_output_check   = weights_hidden_output * hidden_out_check + bias_output;
    output_check_val   = purelin(net_output_check);
    
    check_errors_list = [check_errors_list, target_check(i) - output_check_val];
    output_check_list = [output_check_list, output_check_val];
    
    hidden_out_prev_check = hidden_out_check;
end

final_check_mse = mse(check_errors_list)

train_errors_summary = [train_errors_summary, final_train_mse];
test_errors_summary  = [test_errors_summary, final_test_mse];

%% 6. Final Result Plots
figure;
plot(output_train_epoch); hold on; plot(target_train, 'r');
title('Outputs for training data & target');

figure;
plot(output_test_epoch); hold on; plot(target_test, 'r');
title('Outputs for testing data & target');

figure;
plot(output_check_list); hold on; plot(target_check, 'r');
title('Outputs for checking data & target');
