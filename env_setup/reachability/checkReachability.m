function [isReachable, configSol] = checkReachability(robot, endeffectorName, ikSolver, ik_weights, initial_guesses, ee_pose)
        
    [configSol, solInfo] = ikSolver(endeffectorName, ee_pose, ik_weights, initial_guesses);
    
    [isColliding, ~, ~] = robot.checkCollision(configSol);
    
    %%% Target
    % tar_pos = ee_pose(1:3,4);
    % tar_eul = rotm2eul(ee_pose(1:3,1:3))';

    %%% Result
    % ee_pose_res = getTransform(robot, configSol, endeffectorName, robot.Base.Name);
    % res_pos = ee_pose_res(1:3,4);
    % res_eul = rotm2eul(ee_pose_res(1:3,1:3))';

    % ikSolver.SolverParameters.SolutionTolerance
    % ikSolver.SolverParameters.SolutionTolerance
    
    if (~isColliding(1) && solInfo.ExitFlag == 1) % ignore self-collision "~isColliding(1) &&" and collision with objects "~isColliding(2) &&"
        isReachable = 1;
        % fprintf("good\n")
        % solInfo.ExitFlag
    else
        isReachable = 0;
        % solInfo.ExitFlag
        % solInfo
        % getTransform(robot, configSol, robot.Base.Name ,robot.BodyNames{end})
        % ee_pose
    end

end