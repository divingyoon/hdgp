function [R_index] = getReachability(app)
    % Parameter
    robot = app.robot_urdf;
    R_mesh = app.R_mesh;
    theta_nums = app.ThetaSlicesSlider.Value;
    phi_nums = app.PhiSlicesSlider.Value;
    endeffectorName1 = app.endeffectorName1;
    endeffectorName2 = app.endeffectorName2;

    theta = linspace(0, pi  , theta_nums);
    phi   = linspace(0, 2*pi, phi_nums + 1);

    ik = inverseKinematics();
    ik.SolverParameters.MaxIterations = app.NumberofIKIteration;
    ik.SolverParameters.SolutionTolerance = 0.01;
    ik.RigidBodyTree = robot;
    % ik_weights = [0. 0. 0. 1 1 1];
    % ik_weights = [0.1 0.1 0.1 1 1 1];
    ik_weights = [0.2 0.2 0.2 1 1 1];
    % ik_weights = [0.25 0.25 0.25 1 1 1];
    % ik_weights = [0.5 0.5 0.5 1 1 1];
    %ik_weights = [0.7 0.7 0.7 1 1 1];
    
    initialguess = app.InitialConfiguration;
    
    [~, i_end, j_end, k_end] = size(R_mesh);
    R_index = zeros(size(R_mesh));

    R_mesh_total = R_mesh;
    R_mesh_total(isnan(R_mesh_total)) = 0;
    R_mesh_total = sum(R_mesh_total, 1);
    total_points_notnan = nnz(R_mesh_total);

    % Initialize joint pose
    pose_target = eye(4);
    pose_target(1:3,4) = [app.X_mesh(1,1,1); app.Y_mesh(1,1,1); app.Z_mesh(1,1,1)];
    [~, configSol] = checkReachability(robot, endeffectorName1, ik, ik_weights, initialguess, pose_target);
    initialguess(1:end/2) = configSol(1:end/2);

    % Main -- Get Reachability
    loop_counter = 0;
    clc;
    tic
    for i=1:i_end
        for j=1:j_end
            for k=1:k_end
                if (R_mesh_total(1,i,j,k))
                    loop_counter = loop_counter + 1;
                    pose_target = eye(4);
                    pose_target(1:3,4) = [app.X_mesh(i,j,k); app.Y_mesh(i,j,k); app.Z_mesh(i,j,k)];

                    fprintf("Point: %d out of %d -> ", loop_counter, total_points_notnan);
                    for i_theta=1:length(theta)
                        for j_phi=1:length(phi)-1
                            pose_target(1:3,1:3) = eul2rotm([theta(i_theta) 0 phi(j_phi)], 'XYZ');
                            if ~isnan(R_mesh(1, i, j, k))
                                [reachable, configSol] = checkReachability(robot, endeffectorName1, ik, ik_weights, initialguess, pose_target);
                                initialguess(1:end/2) = configSol(1:end/2);
                                if reachable
                                    R_index(1, i, j, k) = R_index(1, i, j, k) + 1;
                                end
                            end

                            if app.NumOfEndeffector == 2 && ~isnan(R_mesh(2, i, j, k))
                                [reachable, configSol] = checkReachability(robot, endeffectorName2, ik, ik_weights, initialguess, pose_target);
                                initialguess(end/2+1:end) = configSol(end/2+1:end);
                                if reachable
                                    R_index(2, i, j, k) = R_index(2, i, j, k) + 1;
                                end
                            end
                        end
                    end

                    fprintf("Reachability:");

                    if ~isnan(R_mesh(1, i, j, k))
                        fprintf("EE1 %d/%d \t", R_index(1, i, j, k), theta_nums * phi_nums);
                    end

                    if app.NumOfEndeffector == 2 && ~isnan(R_mesh(2, i, j, k))
                        fprintf("EE2 %d/%d", R_index(2, i, j, k), theta_nums * phi_nums);
                    end
                    fprintf("\n");

                    % fprintf("Reachability %d / %d\n", R_index(i, j, k), theta_nums * phi_nums);
                    % app.MessageTextArea.Value = sprintf("Point: %d out of %d // Reachability %d / %d \n", loop_counter, total_points_notnan, R_index(i, j, k), theta_nums * phi_nums);
                    % pause(0.00001)
                end
            end
        end
    end
    
    R_index(R_index==0) = NaN;
    toc
end