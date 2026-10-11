%canUseGPU
%canUseParallelPool
%gpuDevice

% Set the number of iterations
N = 8; 

% Start a timer
tic;

% Run the parallel loop
parfor i = 1:N
    % Pause for 1 second to simulate a workload
    pause(1); 
    
    % Get the current worker's ID
    t = getCurrentTask();
    
    % If a worker exists (pool is running), print its ID. 
    % Otherwise, print that it ran sequentially.
    if ~isempty(t)
        fprintf('Iteration %d executed by Worker %d\n', i, t.ID);
    else
        fprintf('Iteration %d executed sequentially (no pool active).\n', i);
    end
end

% Stop the timer and display total time
elapsedTime = toc;
fprintf('Total execution time: %.2f seconds\n', elapsedTime);