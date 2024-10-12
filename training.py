import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None):

    # Initialize optimizer
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    scaler = torch.amp.GradScaler('cuda')  # For mixed precision training

    # Handle existing model directory
    if os.path.exists(model_dir):
        val = input(f"The model directory {model_dir} exists. Overwrite? (y/n) ")
        if val == 'y':
            shutil.rmtree(model_dir)

    os.makedirs(model_dir, exist_ok=True)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    start_epoch = 0
    # Load the latest checkpoint if available
    if os.path.exists(checkpoints_dir):
        checkpoints = [f for f in os.listdir(checkpoints_dir) if 'model_epoch_' in f]
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            model.load_state_dict(torch.load(os.path.join(checkpoints_dir, latest_checkpoint)))
            start_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])
            print(f"Resuming training from epoch {start_epoch}")

    total_steps = 0
    with tqdm(total=len(train_dataloader) * (epochs - start_epoch)) as pbar:
        train_losses = []
        for epoch in range(start_epoch, epochs):
            if epoch > 0 and epoch % epochs_til_checkpoint == 0:
                torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'model_epoch_{epoch:04d}.pth'))
                np.savetxt(os.path.join(checkpoints_dir, f'train_losses_epoch_{epoch:04d}.txt'), np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
                
                # Move data to GPU
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                optim.zero_grad()

                with torch.amp.autocast(device_type='cuda'):  # Mixed precision context
                    model_output = model(model_input)
                    losses = loss_fn(model_output, gt)
                    train_loss = sum(loss.mean() for loss in losses.values())

                # Scale the loss and call backward
                scaler.scale(train_loss).backward()

                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optim)
                scaler.update()

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                if total_steps % steps_til_summary == 0:
                    torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'model_current.pth'))
                    summary_fn(model, model_input, gt, model_output, writer, total_steps)

                pbar.update(1)

                if total_steps % steps_til_summary == 0:
                    tqdm.write(f"Epoch {epoch}, Total loss {train_loss:.6f}, iteration time {time.time() - start_time:.6f}")

                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for (model_input_val, gt_val) in val_dataloader:
                                model_input_val = {key: value.cuda() for key, value in model_input_val.items()}
                                gt_val = {key: value.cuda() for key, value in gt_val.items()}
                                model_output_val = model(model_input_val)
                                val_loss = loss_fn(model_output_val, gt_val)
                                val_losses.append(val_loss)

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        model.train()

                total_steps += 1

                # Clear unused variables
                del model_output, losses
                torch.cuda.empty_cache()  # Free up memory

        torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'), np.array(train_losses))


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.0)

