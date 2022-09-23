def getGrad(param, l_bound = -2, u_bound = 2):
            if param.grad==None:
                # print("Grad NONE")
                try: 
                    return np.zeros(( param.shape[0], param.shape[1] ))
                except:
                    try:
                        return np.zeros(param.shape[0])
                    except:
                        return 0.0
            value = param.grad.detach().numpy()
            param.grad = None
            value = np.clip( value, l_bound, u_bound )
            return value