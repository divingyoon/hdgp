function [X_mesh, Y_mesh, Z_mesh, reach_mesh, min_limits, max_limits] = voxelizeWorkspace(app)
    warning off;    
    % Parameter    
    X_ee_array = app.X_ee_array;
    Y_ee_array = app.Y_ee_array;
    Z_ee_array = app.Z_ee_array;
    voxelsPerDim = app.VoxelsperDimensionEditField.Value;

    % Getting the lower and upper limits for axes based on the workspace
    X_lower_lim = min(X_ee_array(:))-0.1; % in meter
    X_upper_lim = max(X_ee_array(:))+0.1; % in meter
    
    Y_lower_lim = min(Y_ee_array(:))-0.1; % in meter
    Y_upper_lim = max(Y_ee_array(:))+0.1; % in meter
    
    Z_lower_lim = min(Z_ee_array(:))-0.1; % in meter
    Z_upper_lim = max(Z_ee_array(:))+0.1; % in meter
    
    min_limits = [X_lower_lim, Y_lower_lim, Z_lower_lim];
    max_limits = [X_upper_lim, Y_upper_lim, Z_upper_lim];
    
    [X_mesh, Y_mesh, Z_mesh] = ndgrid(linspace(X_lower_lim, X_upper_lim, voxelsPerDim), ...
                                        linspace(Y_lower_lim, Y_upper_lim, voxelsPerDim), ...
                                        linspace(Z_lower_lim, Z_upper_lim, voxelsPerDim));
    
    % Reshape the mesh grid points into a Nx3 matrix for vectorized processing
    grid_points = [X_mesh(:), Y_mesh(:), Z_mesh(:)]; % Flatten mesh grid to Nx3 array
    
    % Preallocate the reach_mesh array
    reach_mesh = zeros([app.NumOfEndeffector, size(X_mesh)]);
    
    % Loop over each end-effector
    for j = 1:app.NumOfEndeffector
        if j == 1
            % Convex Hull points
            hull_points = [app.X_ee_array(1, :)', app.Y_ee_array(1, :)', app.Z_ee_array(1, :)']; 

            % Vectorized `inhull` check for all points
            inside_hull = inhull(grid_points, hull_points, app.workspace1);
            
            % Reshape the result back to the 3D grid
            reach_mesh(1, :, :, :) = reshape(inside_hull, size(X_mesh));
        else
            % Convex Hull points
            hull_points = [app.X_ee_array(2, :)', app.Y_ee_array(2, :)', app.Z_ee_array(2, :)']; 
            
            % Vectorized `inhull` check for all points
            inside_hull = inhull(grid_points, hull_points, app.workspace2);
            
            % Reshape the result back to the 3D grid
            reach_mesh(2, :, :, :) = reshape(inside_hull, size(X_mesh));
        end
    end

    % R_mesh = reach_mesh;
    reach_mesh(reach_mesh~=0) = 1;
    reach_mesh(reach_mesh==0) = NaN;
end