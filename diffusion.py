import numpy as np
import torch
import torch.nn.functional as F

from losses import normal_kl, discretized_gaussian_log_likelihood
from model import WrappedModel

LAMBDA = 0.001

def extract(a, t, x_shape):
    """ A helper function to extract values from a tensor given a tensor of timesteps/indices

    Args:
        a (tensor): The values that we want to sample from
        t (tensor): The indices that we want to sample according to
        x_shape (tuple): The shape of our output

    Returns:
        tensor: The values from tensor a sampled according to the indices in tensor t. Shape: N x 1 x 1 x 1
    """

    # Gather the values from a according to the t tensor
    out = a.gather(dim=-1, index=t.cpu())

    # Reshapes the tensor to be compatible with x_shape, new tensor will be N x 1 x 1 x 1
    batch_size = t.shape[0]
    out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    return out





class GaussianDiffusion:
    """This class is given a variance schedule and handles all operations related to forward or reverse diffusion"""
    def __init__(self, betas, vartype, T, sampling_steps = None):

        self.vartype = vartype  # Whether variance is fixed or learned
        self.T = T              # Number of timesteps in diffusion process

        assert self.vartype in ['fixed', 'learned'], "variance type must either be fixed or learned"

        # This calculates all of the constants necessary for the diffusion operations
        self.construct_constants(betas)

        if sampling_steps:
            self.space_betas(sampling_steps)

    def construct_constants(self, betas):
        """Calculates all of the beta/alpha/sqrt constants that are needed for forward and reverse diffusion"""

        self.betas = betas              # The variance schedule
        self.alphas = 1.0 - betas       # Complement of variance schedules
        self.sqrt_recip_alphas = 1.0 / torch.sqrt(self.alphas)      # Reciprocal of the sqrt of alphasa
        self.alpha_cumprods = torch.cumprod(self.alphas, dim=0)     # Cumulative products of the alpha values
        self.sqrt_alpha_cumprods = torch.sqrt(self.alpha_cumprods)  # Square root of cumulative products
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprods)

        # Shifts alpha cumulative products right one and pads beginning with a 1.0
        self.alpha_cumprods_prev = torch.cat([torch.tensor([1]), self.alpha_cumprods[1:]]) 
        
        # Used for posterior variance calculations
        self.posterior_variance = (1 - self.alpha_cumprods_prev) * self.betas / (1 - self.alpha_cumprods)
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))

    def space_betas(self, timesteps):
        """ Re-calculates all of the constants in the class to use a subset, S of total timesteps T.
        This allows for sampling with fewer timesteps based on a reduced variance schedule

        Args:
            timesteps (list[int]): A list of integers of which timesteps we will be sampling from
        """
        # Define empty beta list and starting alpha cumulative product value
        new_betas = []
        self.timestep_map = []
        last_alpha_cumprod = 1.0

        # Iterate over each value in the alpha cumulative products
        for i, alpha_cumprod in enumerate(self.alpha_cumprods):
            if i in timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        
        self.construct_constants(torch.tensor(new_betas))

    def wrap_model(self, model):
        """ Wrap the model so that it converts from subsequence S samplings steps to original T"""
        return WrappedModel(model, self.timestep_map, self.T)

    def forward_diffuse(self, images, t, noise=None):
        """Samples from a forward diffusion process at specified timesteps.
        Samples q(x_t | x{0})

        Args:
            images (tensor): A batch of images to be noised in the shape N x C x H x W
            t (tensor): A batch of timesteps to sample from the forward process in the shape of N
            noise (tensor, optional): Gaussian noise in the shape of the images batch. Defaults to None.

        Returns:
            tensor: The images sampled from a forward diffusion process at timesteps t.
        """

        # If noise is not given, create random noise
        if noise is None:
            noise = torch.randn_like(images)

        # Extract constants for calculation
        sqrt_alpha_cumprod = extract(self.sqrt_alpha_cumprods, t, images.shape)
        sqrt_one_minus_alpha_cumprod = extract(self.sqrt_one_minus_alpha_cumprod, t, images.shape)

        # The new (noised) image will have a mean centered around the original image times the alpha value
        mean = sqrt_alpha_cumprod * images

        # The variance will be controlled by the multiplication of the beta schedules
        variance = sqrt_one_minus_alpha_cumprod

        # Push the gaussian distribution to center on our image with the appropriate variance
        noised_image = mean + noise * variance

        return noised_image

    def get_posterior_mean_variance(self, x_0, x_t, t):
        """ Calculates the mean and variance of the posterior distribution, given some noised distribution and the
        starting distribution. Returns q{x_{t-1} | x_t, x_0}

        Args:
            x_0 (tensor): A batch of the starting images. N x C x H x W
            x_t (tensor): A batch of the noised images. N x C x H x W
            t (_type_): A batch of the timesteps of how noisy the noised images are. N

        Returns:
           (tensor, tensor): The mean and the variance of the posterior distribution. Will be in the same shape
           as the images.
        """        


        assert x_0.shape == x_t.shape, "Starting and ending image shape must be the same."

        # Extract the constants necessary for the operation
        betas = extract(self.betas, t, x_t.shape)
        alphas = extract(self.alphas, t, x_t.shape)
        alpha_cumprods = extract(self.alpha_cumprods, t, x_t.shape)
        alpha_cumprods_prev = extract(self.alpha_cumprods_prev, t, x_t.shape)
    
        # The mean of the posterior given x_0 and x_t
        mean = alpha_cumprods_prev * betas * x_0 / (1 - alpha_cumprods)\
            + torch.sqrt(alphas) * (1 - alpha_cumprods_prev) * x_t / (1 - alpha_cumprods)

        # Posterior variance equation
        variance = extract(self.posterior_variance, t, x_t.shape)
        log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return mean, variance, log_variance

    def p_mean_variance(self, model, x, t):
        
        # Extract each constant and reshape to shape of image
        betas = extract(self.betas, t, x.shape)
        sqrt_recip_alphas = extract(self.sqrt_recip_alphas, t, x.shape)
        sqrt_one_minus_alpha_cumprod = extract(self.sqrt_one_minus_alpha_cumprod, t, x.shape)

        # Run the images and timestep through the model to get the model output
        model_output = model(x, t)

        # If we are learning the variance, split the model output into mean and variance
        if self.vartype == 'learned':
            model_output, pred_variance = torch.chunk(model_output, 2, dim=1)

        # Calculate the mean of the new img
        model_mean = sqrt_recip_alphas * (x - (betas * model_output / sqrt_one_minus_alpha_cumprod))
        
        # If variance is fixed, simply extract the constants for variance and log variance
        if self.vartype == 'fixed':
            model_variance = extract(self.posterior_variance, t, x.shape)
            model_log_variance = extract(self.posterior_log_variance_clipped, t, x.shape)

        # Otherwise interpolate the learned variance between beta and beta prime
        else:
            betas_prime = extract(self.posterior_log_variance_clipped, t, x.shape)
            betas = torch.log(extract(self.betas, t, x.shape))

            # Convert variance from [-1, 1] to [0, 1]
            frac = (pred_variance + 1) / 2
            model_log_variance = frac * betas + (1 - frac) * betas_prime
            model_variance = torch.exp(pred_variance)

        return model_mean, model_variance, model_log_variance


    def p_sample(self, model, x, t, t_index):
        """Samples from the reverse diffusion process given some x_t and a t. 
        Returns p( x_{t-1} | x_t )

        Args:
            model (nn.Module): The UNet that we will use to make a prediction
            x  (tensor): A set of images or x_t. Shape: N x C x H x W
            t (tensor): A tensor containing the timesteps, t that we are starting from
            t_index (int): A single index of the timestep we are starting on

        Returns:
            tensor: A set of denoised images of p(x_{t-1}|x{t}). Same shape as x
        """

        # Obtain the predicted mean and log variance fomr the odel
        model_mean, _, model_log_variance = self.p_mean_variance(model, x, t)

        # If at the last timestep, just return mean without any noise
        if t_index == 0:
            return model_mean
        else:
            # Construct original gaussian distribution and shift it towards model mean with given variance
            noise = torch.randn_like(x)     

            return model_mean + torch.exp(0.5 * model_log_variance) * noise

    def get_vlb_term(self, model, x_0, x_t, t):

        
        # Get the model mean and log variance as well as the real mean and log variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_0, t)
        true_mean, _, true_log_variance = self.get_posterior_mean_variance(x_0, x_t, t)

        # Calculate the kl divergence, take it's mean and divide it by the log of 2
        kl = normal_kl(true_mean, true_log_variance, model_mean, model_log_variance)
        kl = torch.mean(kl, dim=tuple(range(1, len(kl.shape)))) / np.log(2.0)

        # Compute negative log likelihood
        decoder_nll = -discretized_gaussian_log_likelihood(x_0, model_mean, model_log_variance * 0.5)
        decoder_nll = torch.mean(decoder_nll, dim=tuple(range(1, len(decoder_nll.shape)))) / np.log(2.0) 

        # At timestep 0, VLB will be the NLL, otherwise it will be KL
        l_vlb = torch.where((t == 0), decoder_nll, kl)
        
        return l_vlb


    def compute_losses(self, model, images, device):
        """Given the model, and x_0, or a set of starting images, computes the appropriate loss value.
        Either is MSE or a combination of that and a scaled variational lower bound

        Args:
            model (nn.Module): The diffusion model
            images (tensor): A batch of images from the dataset
            device (str, int): An integer corresponding to the device that these are stored on

        Returns:
           int: A single value that is the loss to be optimized for this batch
        """
    
        # Define gaussian noise to be applied to this image
        noises = torch.randn_like(images)

        # Define a series of random timesteps to sample noise from
        timesteps = torch.randint(0, self.T, (images.shape[0], ), device=device, dtype=torch.long)
        
        # Apply noise to each image at each timestep with each gaussian noise
        noised_images = self.forward_diffuse(images, timesteps, noises)
        
        # Predict epsilon, the noise used to noise each image
        model_output = model(noised_images, timesteps)
    
        # If we are learning the variance, then split output into predicted noise and predicted variance
        if self.vartype == "learned":
            model_output, pred_variance = torch.chunk(model_output, 2, dim=1)

        # Calculates simple MSE between pred noise and actual noises
        l_simple = F.mse_loss(model_output, noises)

        # If variance is fixed, loss is just the l_simple MSE term
        if self.vartype == "fixed":
            loss = l_simple

        # Otherwise loss is a hybrid combination of l_simple and variational lower bound
        else:

            # Freeze the mean so it is not updated from VLB loss term
            frozen_out = torch.cat([model_output.detach(), pred_variance], dim=1)

            # Make model a function that just returns the frozen output and get the variational lower bound
            l_vlb = self.get_vlb_term(lambda *args: frozen_out, images, noised_images, timesteps)

            # Scale variational lower bound with timesteps and combine it into hybrid loss
            l_vlb *= self.T / 1000.0
            l_vlb = l_vlb.mean() * LAMBDA
            loss = l_simple + l_vlb
        return loss
        


if __name__ == "__main__":
    diffuser = GaussianDiffusion(torch.linspace(.0001, .02, 1000), vartype='fixed')


