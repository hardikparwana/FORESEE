import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import gpr_lib.Utils.Parameters_covariance_functions as cov_func
model_optimization_opt_dict['criterion'] = Likelihood.Marginal_log_likelihood # Optimize marginal likelihood


self.model_learning.set_training_mode()


# train GPs on observed interaction data
            print('\n\n----- REINFORCE THE MODEL -----')
            self.model_learning.reinforce_model(optimization_opt_list = model_optimization_opt_list)
            
            with torch.no_grad():
                if self.log_path is not None:
                    print('Save log file...')
                    self.log_dict['parameters_gp_'+str(trial_index)] = [copy.deepcopy(self.model_learning.gp_list[k].state_dict()) for k in range(0, self.model_learning.num_gp)]
                    self.log_dict['gp_inputs_'+str(trial_index)] = self.model_learning.gp_inputs
                    self.log_dict['gp_output_list_'+str(trial_index)] = self.model_learning.gp_output_list
                    self.log_dict['state_samples_history'] = self.state_samples_history
                    self.log_dict['input_samples_history'] = self.input_samples_history
                    self.log_dict['noiseless_states_history'] = self.noiseless_states_history
                    pkl.dump(self.log_dict,open(self.log_path+'/log.pkl','wb'))


self.model_learning.set_eval_mode()


# get gp predictions
gp_inputs, gp_outputs_target_list,\
gp_output_mean_list, gp_output_var_list = self.model_learning.get_gp_estimate_from_data(states = torch.tensor(self.state_samples_history[data_collection_index],
                                                                                                            dtype = self.dtype, device = self.device),
                                                                                        inputs = torch.tensor(self.input_samples_history[data_collection_index],
                                                                                                            dtype = self.dtype, device = self.device),
                                                                                        flg_pretrain = flg_pretrain)
for i in range(self.model_learning.num_gp):
    gp_output_var_list[i] = gp_output_var_list[i]*self.model_learning.norm_list[i]**2
    
    
    # move data to numpy
gp_outputs_target_list = [gp_outputs_target_list[i].detach().cpu().numpy()
                            for i in range(0, self.model_learning.num_gp)]
gp_output_mean_list = [gp_output_mean_list[i].detach().cpu().numpy()
                        for i in range(0, self.model_learning.num_gp)]

 # get gp performance
        for gp_index in range(0,self.model_learning.num_gp):

            print('MSE gp'+str(gp_index)+': ',
                  ((gp_outputs_target_list[gp_index]-gp_output_mean_list[gp_index])**2).mean())
        #     # uncomment to plot model learning performance
        #     plt.figure()
        #     plt.plot(gp_outputs_target_list[gp_index], label = 'y '+str(gp_index))
        #     plt.plot(gp_output_mean_list[gp_index], label = 'y '+str(gp_index)+' hat')
        #     plt.grid()
        #     plt.legend()
        #     # plt.savefig('results_tmp/'+'gp'+str(gp_index)+'_trial'+str(data_collection_index)+'.pdf')
        #     # plt.close()
        # plt.show()
        
        