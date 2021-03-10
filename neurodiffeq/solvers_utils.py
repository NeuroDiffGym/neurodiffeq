
class PretrainedSolver(ABC):

    @classmethod
    def load(cls, solver_name_or_path, additional_params=...):
        print("Load.....")
        # Instantiate solver
        solver = cls(*model_args, **model_kwargs)
        return solver
    
    
    def save(self, solver_name_or_path, save_to_hub=False):
	    # Save the solver here
        print("Save.....")
