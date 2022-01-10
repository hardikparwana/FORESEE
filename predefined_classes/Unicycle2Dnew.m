classdef Unicycle2Dnew <handle

        properties(Access = public)
           id = 1;
           X = [0;0;0];
           yaw = 0;
           
           G = [0;0]; %goal  
           status = 'nominal';  % nominal or adversary
           
           % Dynamcs matrices for x_dot = f(x) + g(x)u
           f;
           g;
           
           safe_dist = 0;          
           D = 2;
           
           % figure ploit handles
           p1;         % plot current location
           p2;         % plot safe distance for leader
           p3; % only for leader
           p4;         % plot whole trajectory
           Xt = [];
           
           observed_data = [];
           input_data = [];
           predicted_data = [];
           predicted_std = [];
           predicted_normal_data = [];
           predicted_normal_std = [];
           inputs = [];
           
           particles;
           weights;
           
           max_D;
           min_D;
           FOV;
           
           
        end
        
        properties(Access = private)
            iter = 0;
        end
        
        methods(Access = public)
           
            function quad = Unicycle2Dnew(ID,x,y,yaw,r_safe,D,status,min_D, max_D, FOV)
               
                quad.X(1) = x;
                quad.X(2) = y;
                quad.X(3) = yaw;
                quad.yaw = yaw;
                quad.safe_dist = r_safe;
                quad.id=ID;                
                quad.D = D;
                quad.status = status;                
                quad = plot_update(quad); 
                quad.min_D = min_D;
                quad.max_D = max_D;
                quad.FOV = FOV;
                
                % Dynamics
                quad.f = [0;
                         0;
                         0];
                quad.g = [cos(yaw) 0;
                         sin(yaw) 0;
                         0 1];
                     
            end
            
            function d = plot_update(d)
                        
                center = [d.X(1) d.X(2)];
                radius = d.safe_dist;
                
                d.Xt = [d.Xt;center ];
                
                if strcmp('nominal',d.status)
                    color = 'g';
                else
                    color = 'b';
                end
                
                if (d.iter<1)
                    
                   figure(1)
                   if (d.id>0) % Follower
                       d.p1 = scatter(d.X(1),d.X(2),50,color,'filled');  
                       % Display the safe distance circle.
                       %d.p2 = viscircles(center,radius,'Color',color,'LineStyle','--');
                       d.p4 = plot( d.Xt(:,1),d.Xt(:,2) );
                       d.iter = 1;
                   else        % Leader
                       d.p1 = scatter(d.X(1),d.X(2),50,'r','filled');  
                       % Display the safe distance circle.
                       d.p2 = viscircles(center,radius,'Color','r','LineStyle','--');
                       d.p3 = viscircles(center,d.D,'Color','r','LineStyle','--');
                       d.iter = 1;
                   end
                   
                else
                    
                    set(d.p1,'XData',d.X(1),'YData',d.X(2));
                    set(d.p4,'XData',d.Xt(:,1),'YData',d.Xt(:,2));
                    delete(d.p2);
                    delete(d.p3);
                    
                    if (d.id>0) % Follower
                        %d.p2 = viscircles(center,radius,'Color',color,'LineStyle','--');                 
                    else        % Leader
                        %d.p2 = viscircles(center,radius,'Color','r','LineStyle','--'); 
                        d.p3 = viscircles(center,d.D,'Color','r','LineStyle','--');
                    end
           
                end
                      
            end
            
            
            function out = control_state(d,U,dt)
                
                % Euler update with Dynamics
                
                d.X = d.X + [ U(1)*cos(d.X(3));U(1)*sin(d.X(3)); U(2)]*dt;
%                 cur_sigma = cur_sigma + U(2)*dt;
                d.X(3) = wrap_pi(d.X(3));
                d.yaw = d.X(3);
                
                d.g =[cos(d.X(3)) 0;
                     sin(d.X(3)) 0;
                     0 1];
                
                d = plot_update(d);
                out = d.X;
            
            end
            
            function out = forward_state(X,U,dt)
                
                % Euler update with Dynamics
                
                out = X + [ U(1)*cos(X(3));U(1)*sin(X(3)); U(2)]*dt;
%                 cur_sigma = cur_sigma + U(2)*dt;
                out(3) = wrap_pi(out(3));
            
            end
            
            function out = fgx(d,x)
               
                out = zeros(3,3);
                out(:,2:3) = [ cos(x(3)) 0;
                            sin(x(3)) 0;
                            0      1];
                
            end
            
            function [h, dh_dxi, dh_dxj] = agent_barrier(d,agent)
                
                global d_min
                %barrier
                h = d_min^2 - norm(d.X(1:2)-agent.X(1:2))^2;
                dh_dxi = [-2*(d.X(1:2)-agent.X(1:2))' 0];    % 0 because robot state is x,y,theta
                dh_dxj = [2*(d.X(1:2)-agent.X(1:2))' 0];                
                
            end
            
            function [V, dV_dx_agent, dV_dx_target] = goal_lyapunov(d,X,G)
               
                % Lyapunov
                X = X(1:2,1);
                V = norm(X-G)^2;
                dV_dx_agent = [2*(X-G)' 0];  % 0 because robot state is x,y,theta
                dV_dx_target = -2*( X - G )';
                
            end
                       
                  
            function uni_input = nominal_controller(d,X,G,u_min,u_max)
                
                psi = X(3);
                X = X(1:2,1);
                dx = X - G;
                kw = 0.7*u_max(2)/pi; %0.5*u_max(2)/pi;
                phi_des = atan2( -dx(2),-dx(1) );
                delta_phi = wrap_pi( phi_des - psi );

                w0 = kw*delta_phi;
                kv = 1.0;%0.1;
                v0 = kv*norm(dx)*max(0.1,cos(delta_phi)^2);                

                uni_input = [v0;w0];      
                
            end
            
            function [h, dh_dx_agent, dh_dx_target] = cbf1_loss(d,X,G)
               
                % Max distance
                X = X(1:2,1);
                ratio = d.max_D^2 - d.min_D^2;
                h = d.max_D^2 - norm( X - G )^2;
                dh_dx_agent = -2*( X - G )';
                dh_dx_target = 2*( X - G )';
                
                h = h/ratio;
                dh_dx_agent = [dh_dx_agent/ratio 0];
                dh_dx_target = dh_dx_target/ratio;                
                
            end
            
            function [h, dh_dx_agent, dh_dx_target] = cbf2_loss(d,X,G)
               
                X = X(1:2,1);
                % Min distance
                ratio = d.max_D^2 - d.min_D^2;
                h = norm( X - G )^2 - d.min_D^2
                dh_dx_agent = 2*( X - G )';
                dh_dx_target = -2*( X - G )';
                
                h = h/ratio;
                dh_dx_agent = [dh_dx_agent/ratio 0];
                dh_dx_target = dh_dx_target/ratio;                
                
            end
            
            function [h, dh_dx_agent, dh_dx_target] = cbf3_loss(d,X,G)
               
                % Min angle
                psi = X(3);
                X = X(1:2);
                
                dir_vector = [cos(psi);sin(psi)];
                bearing_angle = dir_vector' * ( G - X ) / norm( G - X );
                h = (bearing_angle - cos(d.FOV/2))/(1-cos(d.FOV/2));
                
                p = G - X;
                dh_dx = dir_vector'/norm(p) - (dir_vector' * p * p')/( norm(p)^3 ); % 1 x 2
                
                dh_dTheta = [-sin(psi) cos(psi)] * p / norm(p);
                
                dh_dx_agent = [-dh_dx dh_dTheta] /(1-cos(d.FOV/2));
                dh_dx_target = dh_dx / (1-cos(d.FOV/2));               
                          
                
            end
           
            
        end



end