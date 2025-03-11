# code to constrain train the slope constrained 1D model.

# SHOULD I CREATE THE CLASS WHERE I CAN ALSO LOAD THE MODEL AND ADD TRAINING AS ONE OF ITS  METHOD. I THINK SO YES. NEED TO LOOK INTO IT

import torch
import torch.optim as optim
from .nll_loss import  nll_loss
from .tv2_regul import TV2   
from tqdm import tqdm
import matplotlib.pyplot as plt
from .training_analysis import analyse_training
from .plot_1d_case import  plot_multiple_histogram, plot_with_annotations
from activations.linearspline_slope_constraint import LinearSplineSlopeConstrained

class ModelTrainer:
    def __init__(self, size, range_, start_val, end_val, init_type='manual', **kwargs):
        # Initialize parameters
        self.SIZE = size
        self.RANGE = range_
        self.START_VAL = start_val
        self.END_VAL = end_val

        # Initialize model
        if 'GRID_VALS' in kwargs and 'FN_INIT' in kwargs:
            self.GRID_VALS = kwargs['GRID_VALS']
            self.FN_INIT = kwargs['FN_INIT']

        # GRID_VALS, FN_INIT = transformation_laplace_to_std_normal(
        #     mu=self.mean, b=self.scale, x_range=(self.START_VAL, self.END_VAL), num_points=self.SIZE
        # )
        noise = torch.randn_like(torch.tensor(self.FN_INIT)) * 2e-1

        #initialise the  model
        self.model = LinearSplineSlopeConstrained(
            mode='fc',
            num_activations=1,
            size=self.SIZE,
            range_=self.RANGE,
            grid_values=torch.tensor(self.GRID_VALS),
            init=init_type,
            smin=0.001,
            smax=10,
            slope_constrained=1,
            manual_init_fn_tensor=torch.tensor(self.FN_INIT) + noise
        )
        if 'mu' in kwargs and 'b' in kwargs:
            self.mu = kwargs.get('mu', 0)
            self.b = kwargs.get('b', 1)

    def before_training(self, train_data, **kwargs):
        # visualise the code space before the training
        input_tensor = torch.tensor(train_data)
        output_tensor2=self.model(input_tensor)
        plot_multiple_histogram([input_tensor.numpy(), output_tensor2.detach().numpy()
                                ],
                                labels_list=["data_space (x)","code_space(z)"])
        
        x2pos, y2pos = self.model.nodal_val_loc_tensor.detach().numpy(), self.model.slope_constrained_coefficients_vect.detach().numpy()
        plt.figure()
        if 'mu' in kwargs and 'b' in kwargs:# for the laplace case
            plot_with_annotations(x2pos[0][:], y2pos[:],
                            title="splines (without TV2)", xlabel="x", ylabel="y",
                            annotate=0, style='-o', mu=self.mu, b=self.b)
        else:
            plot_with_annotations(x2pos[0][:], y2pos[:],
                            title="splines (without TV2)", xlabel="x", ylabel="y",
                            annotate=0, style='-o')
        plt.legend()
        plt.show()


    def train_and_evaluate(
        self, train_loader, val_loader, 
        lambda_tv2=1e-4, num_epochs=10, 
        lr=0.001, print_after=1, tv2_regulation=False, 
        scheduler_type="StepLR", step_size=5, gamma=0.1,
        alpha_nll=1,
        track_coefficients=False, type_model="ls"  # Default to "ls"
    ):
        """
        Train the RealNVP model and evaluate on a validation dataset.

        Args:
        - train_loader (DataLoader): DataLoader for the training dataset.
        - val_loader (DataLoader): DataLoader for the validation dataset.
        - num_epochs (int): Number of training epochs.
        - lr (float): Learning rate for the optimizer.
        - print_after (int): Number of epochs after which to print the training and validation loss.
        - scheduler_type (str): Type of scheduler to use ("StepLR", "ExponentialLR", etc.).
        - step_size (int): Step size for the StepLR scheduler (if applicable).
        - gamma (float): Multiplicative factor for learning rate decay.

        Returns:
        - train_losses (list): List of training losses for each epoch.
        - val_losses (list): List of validation losses for each epoch.
        """
        
        # Define the optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Define the scheduler
        if scheduler_type == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif scheduler_type == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=2)
        else:
            raise ValueError("Unsupported scheduler type. Choose from 'StepLR', 'ExponentialLR', or 'ReduceLROnPlateau'.")

        train_losses = []  # List to store training losses
        val_losses = []  # List to store validation losses
        model_params_history = []  # To store model state_dicts for each epoch
        model_params_history.append({k: v.clone() for k, v in self.model.state_dict().items()})

        if track_coefficients:
            coeffs_evol = []
            slope_const_coeffs_evol = []

        for epoch in tqdm(range(num_epochs), desc="Training Progress"):
            total_train_loss = 0.0
            total_train_and_regul_loss = 0.0
            total_loss_1 = 0.0
            total_loss_2 = 0.0

            if track_coefficients:
                coeffs = self.model.coefficients_vect.detach().numpy()
                if type_model == "ls":
                    slope_const_coeffs = self.model.lipschitz_coefficients_vect.detach().numpy()
                elif type_model == "scls":
                    slope_const_coeffs = self.model.slope_constrained_coefficients_vect.detach().numpy()
                coeffs_evol.append(list(coeffs))
                slope_const_coeffs_evol.append(list(slope_const_coeffs))

            # Training phase
            self.model.train()  # Set the model to training mode
            for data in train_loader:
                inputs = data[0]  # data is a list containing the tensor [tensor()]
                optimizer.zero_grad()

                # Forward pass (encoding)
                encoded = self.model(inputs)

                # Loss calculation
                train_loss, loss_normal, loss_1, loss_2 = nll_loss(encoded, self.model.grad_x_temp,
                                                                    alpha_nll=alpha_nll, return_indiv_loss=1)

                # TV2 regularisation term
                if tv2_regulation:
                    tv2_regul = TV2(self.model, self.model.coefficients_vect.device)

                # Total loss
                if tv2_regulation:
                    total_loss = train_loss + lambda_tv2 * tv2_regul
                else:
                    total_loss = train_loss

                total_loss.backward()  # Backward pass
                optimizer.step()  # Update weights
                total_train_and_regul_loss += total_loss.item()
                total_train_loss += train_loss.item()
                total_loss_1 += loss_1.item()
                total_loss_2 += loss_2.item()

            # Step the scheduler
            if scheduler_type != "ReduceLROnPlateau":
                scheduler.step()
            else:
                scheduler.step(total_train_loss / len(train_loader))

            # Save a copy of the model's state_dict
            model_params_history.append({k: v.clone() for k, v in self.model.state_dict().items()})

            # Average training loss for the epoch
            average_train_loss = total_train_loss / len(train_loader)
            avg_train_and_regul_loss = total_train_and_regul_loss / len(train_loader)
            avg_loss_1 = total_loss_1 / len(train_loader)
            avg_loss_2 = total_loss_2 / len(train_loader)

            # Validation phase
            if val_loader is not None:
                self.model.eval()  # Set the model to evaluation mode
                total_val_loss = 0.0
                with torch.no_grad():
                    for val_data in val_loader:
                        val_inputs = val_data[0]
                        val_encoded = self.model(val_inputs)
                        val_loss, _ = nll_loss(val_encoded, self.model.grad_x_temp)
                        total_val_loss += val_loss.item()

                average_val_loss = total_val_loss / len(val_loader)

                if (epoch + 1) % print_after == 0:
                    print(
                        f"Epoch {epoch + 1}/{num_epochs}, "
                        f"train NLL+{lambda_tv2 * int(tv2_regulation)}XTV2: {avg_train_and_regul_loss}, "
                        f"data avg loss-1: {avg_loss_1}, "
                        f"data avg loss-2: {avg_loss_2}, "
                        f"Training NLL Loss: {average_train_loss}, "
                        f"Validation NLL Loss: {average_val_loss}, "
                        f"LR: {scheduler.get_last_lr()[0]}"
                    )

                train_losses.append(average_train_loss)
                val_losses.append(average_val_loss)

            self.model.train()  # Set the model back to training mode

        print("Training complete")

        if track_coefficients:
            return train_losses, val_losses, model_params_history, coeffs_evol, slope_const_coeffs_evol
        return train_losses, val_losses, model_params_history

    def plot_training_dynamics(self, train_loss, val_loss, epoch_wise_param_list, test_data, figsize=(20, 20), plot_freq=1):
        # Visualize the training dynamics
        if self.mu and self.b:
            analyse_training(
                model_in=self.model,
                train_loss=train_loss,
                val_loss=val_loss,
                test_data=test_data,
                figsize=figsize,
                model_params_list=epoch_wise_param_list,
                num_coeffs=10,
                size=self.SIZE,
                range=self.RANGE,
                print_model_params=0,
                plot_freq=plot_freq,
                mu=self.mu,
                b=self.b
            )
        else:
            analyse_training(
                model_in=self.model,
                train_loss=train_loss,
                val_loss=val_loss,
                test_data=test_data,
                figsize=figsize,
                model_params_list=epoch_wise_param_list,
                num_coeffs=10,
                size=self.SIZE,
                range=self.RANGE,
                print_model_params=0,
                plot_freq=plot_freq,
            )



def generate_unequally_spaced_sorted_numbers(start_val, end_val, N):
    """
    Generate N unequally spaced but sorted numbers between start_val and end_val.

    Parameters:
        start_val (float): The starting value of the range.
        end_val (float): The ending value of the range.
        N (int): Number of numbers to generate.

    Returns:
        torch.Tensor: A tensor of N sorted, unequally spaced numbers.
    """
    assert N > 1, "N must be greater than 1 to create a range."
    assert start_val < end_val, "start_val must be less than end_val."
    
    # Generate N random values between 0 and 1
    random_values = torch.rand(N)
    
    # Sort the random values to ensure monotonicity
    sorted_values = torch.sort(random_values).values
    
    # Scale and shift the values to fit into the range [start_val, end_val]
    scaled_values = start_val + (end_val - start_val) * sorted_values
    
    return scaled_values

# train and evaluate fn.
def train_and_evaluate(
    model, train_loader, val_loader, 
    lambda_tv2=1e-4, num_epochs=10, 
    lr=0.001, print_after=1, tv2_regulation=False, 
    scheduler_type="StepLR", step_size=5, gamma=0.1,
    alpha_nll=1,
    track_coefficients = False, type_model="ls"#"scls"
):
    """
    Train the RealNVP model and evaluate on a validation dataset.

    Args:
    - model: The NF model to be trained.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    - num_epochs (int): Number of training epochs.
    - lr (float): Learning rate for the optimizer.
    - print_after (int): Number of epochs after which to print the training and validation loss.
    - scheduler_type (str): Type of scheduler to use ("StepLR", "ExponentialLR", etc.).
    - step_size (int): Step size for the StepLR scheduler (if applicable).
    - gamma (float): Multiplicative factor for learning rate decay.

    Returns:
    - train_losses (list): List of training losses for each epoch.
    - val_losses (list): List of validation losses for each epoch.
    """

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Define the scheduler
    if scheduler_type == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=2)
    else:
        raise ValueError("Unsupported scheduler type. Choose from 'StepLR', 'ExponentialLR', or 'ReduceLROnPlateau'.")

    train_losses = []  # List to store training losses
    loss_1_list=[]
    loss_2_list=[]
    val_losses = []  # List to store validation losses
    model_params_history = []  # To store model state_dicts for each epoch
    # Save a copy of the model's state_dict
    model_params_history.append({k: v.clone() for k, v in model.state_dict().items()})

    if track_coefficients:
        coeffs_evol=[]
        slope_const_coeffs_evol=[]

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        total_train_loss = 0.0
        total_train_and_regul_loss=0.0
        total_loss_1=0.0; total_loss_2=0.0

        if track_coefficients:
            # do I really need to change them to numpy?
            coeffs=model.coefficients_vect.detach().numpy()
            if type_model=="ls":  #activations.linearspline_slope_constraint.LinearSplineSlopeConstrained
                slope_const_coeffs = model.lipschitz_coefficients_vect.detach().numpy()
            elif type_model== "scls":
                slope_const_coeffs = model.slope_constrained_coefficients_vect.detach().numpy()
            coeffs_evol.append(list(coeffs))
            slope_const_coeffs_evol.append(list(slope_const_coeffs))

        # Training phase
        model.train()  # Set the model to training mode
        for data in train_loader:
            inputs = data[0]  # data is a list containing the tensor [tensor()]
            # print(f"shape of inputs: {inputs.size()}")
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass (encoding)
            encoded = model(inputs)

            # Loss calculation
            ## Normal loss term
            train_loss, loss_normal, loss_1, loss_2 = nll_loss(encoded, model.grad_x_temp,
                                            alpha_nll=alpha_nll,return_indiv_loss=1)

            ### TV2 regularisation term
            if tv2_regulation:
                tv2_regul = TV2(model, model.coefficients_vect.device)
            # print(f"tv2 regul: {tv2_regul}")
            ## Total loss
            if tv2_regulation:
                total_loss = train_loss + lambda_tv2 * tv2_regul
            else:
                total_loss = train_loss
            # print(f"lambda_tv2 {(lambda_tv2 )} x tv2_regul: {lambda_tv2 * tv2_regul}")
            # Backward pass (gradient computation)
            total_loss.backward()

            # Update weights
            optimizer.step()
            total_train_and_regul_loss += total_loss.item()
            total_train_loss += train_loss.item()#loss_normal.item()#train_loss.item()
            total_loss_1 +=loss_1.item()
            total_loss_2 += loss_2.item()# initially this + was missing
        # Step the scheduler
        if scheduler_type != "ReduceLROnPlateau":
            scheduler.step()
        else:
            scheduler.step(total_train_loss / len(train_loader))
        
        # Save a copy of the model's state_dict
        model_params_history.append({k: v.clone() for k, v in model.state_dict().items()})

        # i commented it because i also want to see how coefficients look before the training begins
        # if track_coefficients:
        #     # do I really need to change them to numpy?
        #     coeffs=model.coefficients_vect.detach().numpy()
        #     slope_const_coeffs = model.slope_constrained_coefficients_vect.detach().numpy()
        #     coeffs_evol.append(list(coeffs))
        #     slope_const_coeffs_evol.append(list(slope_const_coeffs))

        # Average training loss for the epoch
        average_train_loss = total_train_loss / len(train_loader)
        avg_train_and_regul_loss = total_train_and_regul_loss / len(train_loader)
        avg_loss_1 = total_loss_1/len(train_loader)
        avg_loss_2 = total_loss_2/len(train_loader)
        # Validation phase
        if val_loader is not None:
            model.eval()  # Set the model to evaluation mode
            total_val_loss = 0.0
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs = val_data[0]

                    # Forward pass (encoding) for validation
                    val_encoded = model(val_inputs)

                    # Loss calculation for validation
                    val_loss,_ = nll_loss(val_encoded, model.grad_x_temp)

                    total_val_loss += val_loss.item()

            # Average validation loss for the epoch
            average_val_loss = total_val_loss / len(val_loader)

            # Print training and validation losses together
            if (epoch + 1) % print_after == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, "
                    f" train NLL+{lambda_tv2 * int(tv2_regulation)}XTV2: {avg_train_and_regul_loss}, "
                    f"data avg loss-1: {avg_loss_1} ,"
                    f"data avg loss-2: {avg_loss_2} ,"
                    f"Training NLL Loss: {average_train_loss}, "
                    f"Validation NLL Loss: {average_val_loss}, "
                    f"LR: {scheduler.get_last_lr()[0]}"
                )

            # Append losses to the lists
            train_losses.append(average_train_loss)
            val_losses.append(average_val_loss)

        # Set the model back to training mode
        model.train()

    print("Training complete")

    if track_coefficients:
        return train_losses, val_losses,model_params_history, coeffs_evol, slope_const_coeffs_evol
    return train_losses, val_losses,model_params_history
