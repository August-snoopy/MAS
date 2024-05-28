import argparse
import os, sys
import json
from typing import Tuple

import numpy as np

from config import config
from model import siMLPe as Model
from amass import AMASSDataset
from utils.logger import get_logger, print_and_log_info
from utils.pyt_utils import link_file, ensure_dir

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def generate_dct_and_idct_matrix(signal_length: int) -> tuple:
    """
    Generate Discrete Cosine Transform (DCT) matrix and its inverse.

    Args:
        signal_length: int, length of the input signal

    Returns:
        dct_matrix: np.ndarray, DCT matrix
        idct_matrix: np.ndarray, inverse DCT matrix
    """
    # Initialize the DCT matrix as an identity matrix
    dct_matrix = np.eye(signal_length)

    # Calculate the DCT matrix
    for freq in np.arange(signal_length):
        for time in np.arange(signal_length):
            # Calculate the weight
            weight = np.sqrt(2 / signal_length) if freq != 0 else np.sqrt(1 / signal_length)
            # Update the DCT matrix
            dct_matrix[freq, time] = weight * np.cos(np.pi * (time + 0.5) * freq / signal_length)

    # Calculate the inverse DCT matrix
    idct_matrix = np.linalg.inv(dct_matrix)

    return dct_matrix, idct_matrix


def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer):
    if nb_iter > 100000:
        current_lr = 1e-5
    else:
        current_lr = 3e-4

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


def calculate_velocity(motion_matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculate the velocity from the given motion matrix.

    The velocity is calculated as the difference between each consecutive pair of elements in the motion matrix.
    This is done by subtracting the matrix with its own shifted version (shifted by one time step).

    Args:
        motion_matrix: torch.Tensor, the motion matrix

    Returns:
        velocity_matrix: torch.Tensor, the velocity matrix
    """
    # Shift the motion matrix by one time step
    shifted_motion_matrix = motion_matrix[:, 1:]

    # Calculate the velocity matrix as the difference between the motion matrix and its shifted version
    velocity_matrix = shifted_motion_matrix - motion_matrix[:, :-1]

    return velocity_matrix


def train_step(amass_motion_input: torch.Tensor,
               amass_motion_target: torch.Tensor,
               dct_m: torch.Tensor,
               idct_m: torch.Tensor,
               writer: SummaryWriter,
               model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               nb_iter: int,
               total_iter: int,
               max_lr: float,
               min_lr: float) -> Tuple[float, torch.optim.Optimizer, float]:
    """
    Perform a training step.

    Args:
        amass_motion_input: torch.Tensor, the input motion data
        amass_motion_target: torch.Tensor, the target motion data
        dct_m: torch.Tensor, the DCT matrix
        idct_m: torch.Tensor, the inverse DCT matrix
        writer: SummaryWriter, the writer for logging
        model: torch.nn.Module, the model to be trained
        optimizer: torch.optim.Optimizer, the optimizer for training
        nb_iter: int, the current iteration number
        total_iter: int, the total number of iterations
        max_lr: float, the maximum learning rate
        min_lr: float, the minimum learning rate

    Returns:
        loss_value: float, the loss value of this training step
        optimizer: torch.optim.Optimizer, the updated optimizer
        current_lr: float, the current learning rate
    """
    # If derivative input is enabled in the configuration, apply DCT to the input motion data
    if config.deriv_input:
        amass_motion_input = torch.matmul(dct_m, amass_motion_input.cuda())

    # Forward pass through the model
    motion_pred = model(amass_motion_input.cuda())
    motion_pred = torch.matmul(idct_m, motion_pred)

    # If derivative output is enabled in the configuration, add the last frame of the input to the prediction
    if config.deriv_output:
        offset = amass_motion_input[:, -1:].cuda()
        motion_pred = motion_pred[:, :config.motion.amass_target_length] + offset

    # Calculate the loss as the mean of the L2 norm between the prediction and the target
    motion_pred = motion_pred.reshape(-1, 3)
    amass_motion_target = amass_motion_target.cuda().reshape(-1, 3)
    loss = torch.mean(torch.norm(motion_pred - amass_motion_target, 2, 1))

    # If relative loss is enabled in the configuration, add the derivative loss to the total loss
    if config.use_relative_loss:
        motion_pred = motion_pred.reshape(-1, 18, 3)
        dmotion_pred = calculate_velocity(motion_pred)
        motion_gt = amass_motion_target.reshape(-1, 18, 3)
        dmotion_gt = calculate_velocity(motion_gt)
        dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1, 3), 2, 1))
        loss = loss + dloss

    # Log the loss value
    writer.add_scalar('Loss/angle', loss.detach().cpu().numpy(), nb_iter)

    # Perform backpropagation and update the model parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update the learning rate
    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer)
    writer.add_scalar('LR/train', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr


def main(config, model_path=None):
    """
    The main function for training the model.

    Args:
        config: The configuration object containing all the necessary parameters.
        model_path: str, optional, the path to the pre-trained model. If provided, the model will be loaded from this path.

    Returns:
        None
    """
    torch.manual_seed(config.seed)
    writer = SummaryWriter()

    # Generate DCT and IDCT matrices
    dct_m, idct_m = generate_dct_and_idct_matrix(config.motion.amass_input_length_dct)

    # The shape of dct_m is (config.motion.amass_input_length_dct, config.motion.amass_input_length_dct) after generate

    # We use unsqueeze to add an extra dimension at the beginning of the tensor
    # Match the expected input shape for the matrix multiplication operation that will be performed later in the code
    # The shape of dct_m becomes (1, config.motion.amass_input_length_dct, config.motion.amass_input_length_dct)
    dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)

    # Similar operation is performed on idct_m
    idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

    model = Model(config)
    model.train()
    model.cuda()

    config.motion.amass_target_length = config.motion.amass_target_length_train
    dataset = AMASSDataset(config, 'train', config.data_aug)

    dataloader = DataLoader(dataset, batch_size=config.batch_size,
                            num_workers=config.num_workers, drop_last=True,
                            shuffle=True, pin_memory=True)

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.cos_lr_max,
                                 weight_decay=config.weight_decay)

    ensure_dir(config.snapshot_dir)
    logger = get_logger(config.log_file, 'train')
    link_file(config.log_file, config.link_log_file)

    print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))

    if model_path is not None:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict, strict=True)
        print_and_log_info(logger, "Loading model path from {} ".format(model_path))

    # Training
    nb_iter = 0
    avg_loss = 0.
    avg_lr = 0.

    while (nb_iter + 1) < config.cos_lr_total_iters:
        for (amass_motion_input, amass_motion_target) in dataloader:
            loss, optimizer, current_lr = train_step(amass_motion_input,
                                                     amass_motion_target,
                                                     dct_m,
                                                     idct_m,
                                                     writer,
                                                     model,
                                                     optimizer,
                                                     nb_iter,
                                                     config.cos_lr_total_iters,
                                                     config.cos_lr_max,
                                                     config.cos_lr_min)
            avg_loss += loss
            avg_lr += current_lr

            if (nb_iter + 1) % config.print_every == 0:
                avg_loss = avg_loss / config.print_every
                avg_lr = avg_lr / config.print_every

                print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
                print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
                avg_loss = 0
                avg_lr = 0

            if (nb_iter + 1) % config.save_every == 0:
                torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')

            if (nb_iter + 1) == config.cos_lr_total_iters:
                break
            nb_iter += 1

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train siMLPe')
    parser.add_argument('--config', type=str, default='config', help='config file name')
    parser.add_argument('--model_path', type=str, default=None, help='pretrained model path')
    args = parser.parse_args()

    config.merge_from_file(f'./exps/{args.config}.py')
    main(config, args.model_path)
