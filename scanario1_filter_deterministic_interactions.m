clear all;
close all;

warning('off');
num = 2;
% Display
figure(1+num)
hold on
% set(gca,'DataAspectRatio',[1 1 1])
axis([ 0 5 0 8 ])

% Parameters
global r d_min
r=0.25;  % goal display radius
d_min=0.1;   % inter agent collision minimum distance

% robot dynamics
inputs_per_robot = 2;

% Robots 
n_robots = 3;
robot(1) = SingleIntegrator2D(1,3.0,1,'reactive');   %ID,x,y,yaw,r_safe,D,status
robot(2) = SingleIntegrator2D(2,2.5,0,'reactive'); 
robot(3) = SingleIntegrator2D(3,3.5,0,'reactive');

robot_nominal(1) = SingleIntegrator2D(1,3.0,1,'nominal');   %ID,x,y,yaw,r_safe,D,status
robot_nominal(2) = SingleIntegrator2D(2,2.5,0,'nominal'); 
robot_nominal(3) = SingleIntegrator2D(3,3.5,0,'nominal');

num_robots = size(robot,2);

% Humans
human(1) = SingleIntegrator2D(1,0.0,4,'human');

dt = 0.05;
tf = 6.0;
alpha_cbf = 0.8;%1.0;

% Record Video
myVideo = VideoWriter('always_cbf'); %open video file
myVideo.FrameRate = 10;  %can adjust this, 5 - 10 works well for me
open(myVideo)

% Nominal simulation
% for t=0:dt:tf
%     
%    % Human movement: straight line
%    u_human = [1.0;0.0];
%    human(1) = control_state(human(1),u_human,dt);
%    
%    % Opne Loop Robots:
%    u_robot = [0;1.0];
%    robot(1) = control_state(robot(1),u_robot,dt);
%    robot(2) = control_state(robot(2),u_robot,dt);
%    robot(3) = control_state(robot(3),u_robot,dt);
%    
%    pause(0.01)
%     
% end
% keyboard
% Safety Critical un-cooperative behaviour

for t=0:dt:tf
    
   % Human movement: straight line. Therefore uncooperative
   u_human = [1.0;0.0];
   human(1) = control_state(human(1),u_human,dt);
   
   % Open Loop Robots:
   u_robot = [0;1.0];
   
   for i=1:1:num_robots
       
       robot_nominal(i) = control_state(robot_nominal(i),u_robot,dt);
       
       % Human
       [h, dh_dxi, dh_dxj] = agent_barrier(robot(i),human(1));
       contribution = dh_dxj * u_human;
       
       [V, dV_dx] = lyapunov(robot(i),robot_nominal(i).X);
       V_max = 0.2;
       k = 2.0;
       
       % nominal control input
%        cvx_begin quiet        
%             variable u(2,1)
%             minimize( norm(u) ); 
%             subject to
%                 dV_dx * u <= -k*V;%+ k * V_max;
%        cvx_end
       
       u_nominal = -k*V*dV_dx'/norm(dV_dx);
       
       if 1%contribution>0 % uncooperative
          robot(i).trust_humans = 0;  
          
          cvx_begin quiet        
            variable u(2,1)
            variable delta
%             minimize( norm(u-u_robot) ); 
            minimize( 1.01*norm(u-u_nominal) )% + 100*norm(delta) ); 
            subject to
                dh_dxi*( robot(i).f + robot(i).g*u) + dh_dxj*( human(1).f + human(1).g*u_human ) <= -alpha_cbf * h;
%                 dV_dx * u <= -k*V + delta ;%+ k * V_max;
          cvx_end
       else
          robot(i).trust_humans = 1;
          u = u_nominal;
       end
%        if V>V_max
%            keyboard
%        end
        % Know future movement so see if human is contributing or not        
        robot(i) = control_state(robot(i),u,dt);
        robot(i).inputs(:,end+1) = u;
        
        frame = getframe(gcf); %get frame
        writeVideo(myVideo, frame);
        
   end
   
   pause(0.05)
    
end

close(myVideo)

figure(2+num)
hold on
plot(robot(1).inputs(2,:),'r')
plot(robot(2).inputs(2,:),'g')
plot(robot(3).inputs(2,:),'k')
legend('1','2','3')
