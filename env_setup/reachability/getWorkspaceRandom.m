function [X_ee_array, Y_ee_array, Z_ee_array] = getWorkspaceRandom(app)
    warning off;
    % Parameter
    robot = app.robot_urdf;
    lower_limits = app.lower_limits;
    upper_limits = app.upper_limits;
    total_points = app.NumberofRandomConfigurationsEditField.Value;

    app.endeffectorName1 = app.EndEffector1DropDown.Value;
    endeffectorName1 = app.endeffectorName1;
    app.endeffectorName2 = app.EndEffector2DropDown.Value;
    endeffectorName2 = app.endeffectorName2;
    
    
    X_ee_array = zeros([app.NumOfEndeffector, total_points]);
    Y_ee_array = zeros([app.NumOfEndeffector, total_points]);
    Z_ee_array = zeros([app.NumOfEndeffector, total_points]);
    
    for i=1:total_points
        joint_rand = rand(size(lower_limits(:)), "double")';
        joint_rand = lower_limits + joint_rand .* (upper_limits - lower_limits);
    
        ee_pose = getTransform(robot, joint_rand', endeffectorName1, robot.Base.Name);
        X_ee_array(1,i) = ee_pose(1, 4);
        Y_ee_array(1,i) = ee_pose(2, 4);
        Z_ee_array(1,i) = ee_pose(3, 4);
        
        if app.NumOfEndeffector == 2
            ee_pose = getTransform(robot, joint_rand', endeffectorName2, robot.Base.Name);
            X_ee_array(2,i) = ee_pose(1, 4);
            Y_ee_array(2,i) = ee_pose(2, 4);
            Z_ee_array(2,i) = ee_pose(3, 4);
        end

        fprintf("Point %d out of %d \n Progress %2.2f%%\n", i, total_points, (i/total_points)*100);
        % app.MessageTextArea.Value = sprintf("Point %d out of %d \n Progress %2.2f%%\n", i, total_points, (i/total_points)*100);
        % pause(0.00001)
    end
end