import math
import random
import os

import numpy as np
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from mnist_data import MNISTDataset
from celeb_data import CelebDataset
from diffusion import GaussianDiffusion
from model import UNet



# This is the amount of timesteps we can sample from
T = 4000

# A fixed hyperparameter for the cosine scheduler
S = 0.008

def linear_schedule(timesteps):

    # Scale beta start and end to work with any number of timesteps
    scale = 1000.0 / timesteps
    beta_start = 0.0001 * scale
    beta_end = 0.02 * scale

    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_schedule(timesteps):

    # The cosine schedule formula
    formula = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = []
    for i in range(timesteps):
        t1 = i / timesteps
        t2 = (i + 1) / timesteps
        betas.append(1-formula(t2) / formula(t1))

    betas = torch.tensor(betas)
    return torch.clip(betas, 0.0001, 0.9999)

def reverse_transform(image, switch_dims=True):
    """ Takes in a tensor image and converts it to a uint8 numpy array"""

    image = (image + 1) / 2                 # Range [-1, 1] -> [0, 1)
    image *= 255                            # Range [0, 1) -> [0, 255)

    if len(image.shape) == 3:
        image = image.permute(1, 2, 0)
    image = image.numpy().astype(np.uint8)  # Cast image to numpy and make it an unsigned integer type (no negatives)


    return image

@torch.no_grad()
def reverse_diff(model, diffuser, sampling_steps, shape):
    """Constructs a sequence of frames of the denoising process"""

    device = next(model.parameters()).device

    b = shape[0]

    # Start imgs out with random noise
    img = torch.randn(shape, device=device)

    imgs = []

    # Construct an even sampling range between 0 and T according to n_sampling_steps
    sampling_range = range(0, sampling_steps)

    # Wrap the model so that it converts timesteps from S to timesteps from original T sequence
    model = diffuser.wrap_model(model)

    # Loops through each timestep to get to the generated image
    for i in tqdm(reversed(sampling_range), desc='sampling loop time step', total=sampling_steps):

        # Samples from the specific timestep
        img = diffuser.p_sample(model, img, torch.full((b,), i, device=device), i)
        imgs.append(img.cpu())

    return imgs




def imshow(image):
    """Helper function for displaying images"""
    plt.imshow(image, cmap='gray')
    plt.show()

def extract(a, t, x_shape):
    """ Helper function to extract indices from a tensor and reshape them"""

    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def construct_image_grid(model, device, image_size, image_channels, num_imgs):
    """Constructs a 3x3 grid of images using the diffusion model"""

    imgs = reverse_diff(model, device, image_size, image_channels, num_imgs)[-1]

    return make_grid(imgs, nrow=int(math.sqrt(num_imgs)))


def main():
    world_size = 1
    if world_size > 1:
        
        mp.spawn(setup,
                args=(world_size, ),
                nprocs=world_size,
                join=True)
    else:
        # If world size is 1, set rank to first device and call function normally
        rank = 0
        train_model(rank, world_size)

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    train_model(rank, world_size)


def train_model(rank, world_size):

    # Define Hyperparameters
    batch_size = 8
    grad_iters = 16
    epochs = 100
    lr = 3e-4
    device = rank
    channel_space = 128

    image_size = 64
    image_channels = 3
    dim_mults = (1, 2, 4, 8)
    attn_resos = (2, 4, 8)
    dropout=0.1
    vartype = 'learned'
    n_log_images = 9        # How many images to log to wandb each epoch
    sampling_steps = 1000

    model_checkpoint = None
    data_path = "/home/bassets/diffusion_models/data/celebHQ/"

    # Define a diffusion process for training and one for sampling
    train_diffuser = GaussianDiffusion(linear_schedule(T), vartype, T)
    sampler_diffuser = GaussianDiffusion(linear_schedule(T), vartype, T, sampling_steps=list(range(0, T, T // sampling_steps)))

    # Define the dataset
    train_set = CelebDataset(data_path)

    # If the world size is greater than 1, initialize a distributed sampler to split dataset up
    sampler = None if world_size <= 1 else DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, sampler=sampler)

    # Define model, optimizer, and loss function
    model = UNet(img_start_channels = image_channels, channel_space=channel_space, dim_mults=dim_mults, 
    vartype=vartype, attn_resolutions=attn_resos, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()
    # If there is a model checkpoint set, then load weights from that model
    if model_checkpoint:
        model.load_state_dict(torch.load(model_checkpoint))


    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)


    # Initialize wandb project if we are at first rank
    if rank == 0 or world_size == 1:
        wandb.init(project='diffusion_testing', settings=wandb.Settings(start_method='fork'))


    # Keep track of training iterations
    count = 0
    for epoch in range(epochs):
        

        #train_loader.sampler.set_epoch(epoch)
        # Before each epoch, make sure model is in training mode
        model.train()

        # Initialize statistics to keep track of loss
        running_loss = 0

        # If rank is 0 then apply tqdm progress bar to this
        tqdm_train_loader = tqdm(train_loader, desc=f'Epoch {epoch}') if rank == 0 else train_loader
        
        # Zero out the gradients before starting batch
        optimizer.zero_grad()

        # Iterate over batch of images and random timesteps
        for batch_idx, images in enumerate(tqdm_train_loader):
            
            
            # Cast image to device
            images = images.to(device)

            # Use automatic mixed precision for loss calculation
            loss = train_diffuser.compute_losses(model, images, device)

            # Apply loss to each prediction and update count
            running_loss += loss.item()

            loss = loss / grad_iters

            # Update model parameters
            loss.backward()         # Send loss backwards (compute gradients)

            if (batch_idx + 1) % grad_iters == 0:
                optimizer.step()        # Update model weights
                count += 1              # Keep track of num of updates
                optimizer.zero_grad()
        

        # If rank is 0 then evaluate model
        if rank == 0:
            # Set model to evaluation mode before doing validation steps
            model.eval()

            # Grab random samples from the training set and convert them into wandb image for logging
            real_images = torch.stack([train_set[random.randint(0, len(train_set)-1)] for _ in range(n_log_images)])

            real_img_grid = make_grid(real_images, nrow=3)

            # Generate a batch of images
            gen_imgs = reverse_diff(model, sampler_diffuser, sampling_steps, (n_log_images, image_channels, image_size, image_size))

            # Make the last (t=0) slice of images into a grid
            gen_img_grid = make_grid(gen_imgs[-1], nrow=3)

            # Take all of the frames from the reverse diffusion process for an image and convert it into numpy array
            gif = reverse_transform(torch.stack(gen_imgs)[:, 0], switch_dims=False)

            # Log all of the statistics to wandb
            wandb.log({'Loss': running_loss / len(train_loader),
                        'Training Iters': count * 4,
                        'Generated Images': wandb.Image(gen_img_grid, caption="Generated Images"),
                        'Real Images': wandb.Image(real_img_grid, caption='Real Images'),
                        'Gif': wandb.Video(gif, fps=60, format='gif')})

            # Save the model checkpoint somewhere
            #torch.save(model.state_dict(), 'checkpoints/weights_1.pt')


if __name__ == "__main__":
    main()


