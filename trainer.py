import os
import time
import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision
import matplotlib
from meta import Meta

import network
import train_dataset
import utils

def WGAN_trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    # Build networks
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    if opt.perceptual_loss:
        perceptualnet = utils.create_perceptualnet()
    else:
        perceptualnet = None


    # Loss functions
    L1Loss = nn.L1Loss()
    MSELoss = nn.MSELoss()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(lr_in, optimizer, epoch, opt):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = lr_in * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the two-stage generator model
    def save_model_generator(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'deepfillv2_WGAN_G_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
                
    # Save the dicriminator model
    def save_model_discriminator(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'deepfillv2_WGAN_D_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
                
    # load the model
    def load_model(net, epoch, opt, type='G'):
        """Save the model at "checkpoint_interval" and its multiple"""
        if type == 'G':
            model_name = 'deepfillv2_WGAN_G_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        else:
            model_name = 'deepfillv2_WGAN_D_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        pretrained_dict = torch.load(model_name)
        net.load_state_dict(pretrained_dict)

    if opt.resume:
        load_model(generator, opt.resume_epoch, opt, type='G')
        load_model(discriminator, opt.resume_epoch, opt, type='D')
        print('--------------------Pretrained Models are Loaded--------------------')
        
    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        if opt.perceptual_loss:
            perceptualnet = nn.DataParallel(perceptualnet)
            perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        if opt.perceptual_loss:
            perceptualnet = perceptualnet.cuda()
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = train_dataset.InpaintDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True, drop_last=True)
    
    # ----------------------------------------
    #            Training
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()
    
    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # Training loop
    for epoch in range(opt.resume_epoch, opt.epochs):
        for batch_idx, (img, height, width) in enumerate(dataloader):

            img = img.cuda()
            # set the same free form masks for each batch
            mask = torch.empty(img.shape[0], 1, img.shape[2], img.shape[3]).cuda()
            for i in range(opt.batch_size):
                mask[i] = torch.from_numpy(train_dataset.InpaintDataset.random_ff_mask(
                                                shape=(height[0], width[0])).astype(np.float32)).cuda()
            
            # LSGAN vectors
            valid = Tensor(np.ones((img.shape[0], 1, height[0]//32, width[0]//32)))
            fake = Tensor(np.zeros((img.shape[0], 1, height[0]//32, width[0]//32)))
            zero = Tensor(np.zeros((img.shape[0], 1, height[0]//32, width[0]//32)))

            ### Train Discriminator
            optimizer_d.zero_grad()

            # Generator output
            first_out, second_out = generator(img, mask)

            # forward propagation
            first_out_wholeimg = img * (1 - mask) + first_out * mask        # in range [0, 1]
            second_out_wholeimg = img * (1 - mask) + second_out * mask      # in range [0, 1]

            # Fake samples
            fake_scalar = discriminator(second_out_wholeimg.detach(), mask)
            # True samples
            true_scalar = discriminator(img, mask)
            
            # Loss and optimize
            loss_fake = -torch.mean(torch.min(zero, -valid-fake_scalar))
            loss_true = -torch.mean(torch.min(zero, -valid+true_scalar))
            # Overall Loss and optimize
            loss_D = 0.5 * (loss_fake + loss_true)
            loss_D.backward()
            optimizer_d.step()

            ### Train Generator
            optimizer_g.zero_grad()

            # L1 Loss
            first_L1Loss = (first_out - img).abs().mean()
            second_L1Loss = (second_out - img).abs().mean()
            
            # GAN Loss
            fake_scalar = discriminator(second_out_wholeimg, mask)
            GAN_Loss = -torch.mean(fake_scalar)

            # Compute losses
            loss = opt.lambda_l1 * first_L1Loss + opt.lambda_l1 * second_L1Loss + \
                 + opt.lambda_gan * GAN_Loss

            if opt.perceptual_loss:
                # Get the deep semantic feature maps, and compute Perceptual Loss
                img_featuremaps = perceptualnet(img)                          # feature maps
                second_out_featuremaps = perceptualnet(second_out)
                second_PerceptualLoss = L1Loss(second_out_featuremaps, img_featuremaps)

                loss = loss + opt.lambda_perceptual * second_PerceptualLoss

            loss.backward()
            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()
            
            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [first Mask L1 Loss: %.5f] [second Mask L1 Loss: %.5f]" %
                ((epoch + 1), opt.epochs, batch_idx, len(dataloader), first_L1Loss.item(), second_L1Loss.item()))
            print("\r[D Loss: %.5f] [G Loss: %.5f] time_left: %s" %
                (loss_D.item(), GAN_Loss.item(), time_left))
                
            masked_img = img * (1 - mask) + mask
            mask = torch.cat((mask, mask, mask), 1)
            # Summary
            if (batch_idx + 1) % 40 == 0:
                summary = SummaryWriter(opt.logs_dir_path)

                img_list = [img, masked_img, first_out, second_out]
                # img_list = [x[0,:,:,:].copy_(x.data.squeeze()) for x in img_list]
                # print(img_list.shape())
                image_tensor = torch.cat([images[:1] for images in img_list], 0)
                img_grid = torchvision.utils.make_grid(image_tensor.data, nrow=4, padding=0, normalize=False)
                #img_grid = torchvision.utils.make_grid(img_list)
                summary.add_image('img masked_img first_out second_out', img_grid, batches_done)

                summary.add_scalar('first Mask L1 Loss', first_L1Loss.item(), batches_done)
                summary.add_scalar('second Mask L1 Loss', second_L1Loss.item(), batches_done)
                summary.add_scalar('D Loss', loss_D.item(), batches_done)
                summary.add_scalar('G Loss', GAN_Loss.item(), batches_done)
                summary.add_scalar('Total G Loss', loss.item(), batches_done)
                if opt.perceptual_loss:
                    summary.add_scalar('Perceptual Loss', second_PerceptualLoss.item(), batches_done)

                summary.add_scalar('psnr', utils.psnr(second_out_wholeimg, img), batches_done)
                summary.add_scalar('ssim', utils.ssim(second_out_wholeimg, img), batches_done)
                

        # Learning rate decrease
        adjust_learning_rate(opt.lr_g, optimizer_g, (epoch + 1), opt)
        adjust_learning_rate(opt.lr_d, optimizer_d, (epoch + 1), opt)

        # Save the model
        save_model_generator(generator, (epoch + 1), opt)
        save_model_discriminator(discriminator, (epoch + 1), opt)

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [img, mask, masked_img, first_out, second_out]
            name_list = ['gt', 'mask', 'masked_img', 'first_out', 'second_out']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)





def Meta_trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    # Build networks
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    if opt.perceptual_loss:
        perceptualnet = utils.create_perceptualnet()
    else:
        perceptualnet = None

    # Loss functions
    L1Loss = nn.L1Loss()
    MSELoss = nn.MSELoss()

    generator.train()
    discriminator.train()
    maml = Meta(opt, generator, discriminator)

    # Learning rate decrease
    def adjust_learning_rate(lr_in, optimizer, epoch, opt):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = lr_in * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the two-stage generator model
    def save_model_generator(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'deepfillv2_WGAN_G_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
                
    # Save the dicriminator model
    def save_model_discriminator(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'deepfillv2_WGAN_D_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
                
    # load the model
    def load_model(net, epoch, opt, type='G'):
        """Save the model at "checkpoint_interval" and its multiple"""
        if type == 'G':
            model_name = 'deepfillv2_WGAN_G_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        else:
            model_name = 'deepfillv2_WGAN_D_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        pretrained_dict = torch.load(model_name)
        net.load_state_dict(pretrained_dict)

    if opt.resume:
        load_model(generator, opt.resume_epoch, opt, type='G')
        load_model(discriminator, opt.resume_epoch, opt, type='D')
        print('--------------------Pretrained Models are Loaded--------------------')
        
    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        if opt.perceptual_loss:
            perceptualnet = nn.DataParallel(perceptualnet)
            perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        if opt.perceptual_loss:
            perceptualnet = perceptualnet.cuda()
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = train_dataset.InpaintDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True, drop_last=True)
    
    # ----------------------------------------
    #            Training
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()
    
    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # Training loop
    for epoch in range(opt.resume_epoch, opt.epochs):
        for batch_idx, (img, height, width) in enumerate(dataloader):

            img = img.cuda()
            # set the same free form masks for each batch
            mask = torch.empty(img.shape[0], 1, img.shape[2], img.shape[3]).cuda()
            for i in range(opt.batch_size):
                mask[i] = torch.from_numpy(train_dataset.InpaintDataset.random_ff_mask(
                                                shape=(height[0], width[0])).astype(np.float32)).cuda()
            
            # Meta-train step
            loss_D, loss_G, outputs = maml(img, mask)

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()
            
            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [first Mask L1 Loss: %.5f] [second Mask L1 Loss: %.5f]" %
                ((epoch + 1), opt.epochs, batch_idx, len(dataloader), 0, 0))
                #((epoch + 1), opt.epochs, batch_idx, len(dataloader), first_L1Loss.item(), second_L1Loss.item()))
            print("\r[D Loss: %.5f] [G Loss: %.5f] time_left: %s" %
                (loss_D.item(), loss_G.item(), time_left))
                
            masked_img = img * (1 - mask) + mask
            mask = torch.cat((mask, mask, mask), 1)
            '''
            # Summary
            if (batch_idx + 1) % 40 == 0:
                summary = SummaryWriter('models/tmp')

                img_list = [img, masked_img, first_out, second_out]
                # img_list = [x[0,:,:,:].copy_(x.data.squeeze()) for x in img_list]
                # print(img_list.shape())
                image_tensor = torch.cat([images[:1] for images in img_list], 0)
                img_grid = torchvision.utils.make_grid(image_tensor.data, nrow=4, padding=0, normalize=False)
                #img_grid = torchvision.utils.make_grid(img_list)
                summary.add_image('img masked_img first_out second_out', img_grid, batches_done)

                summary.add_scalar('first Mask L1 Loss', first_L1Loss.item(), batches_done)
                summary.add_scalar('second Mask L1 Loss', second_L1Loss.item(), batches_done)
                summary.add_scalar('D Loss', loss_D.item(), batches_done)
                summary.add_scalar('G Loss', GAN_Loss.item(), batches_done)
                if opt.perceptual_loss:
                    summary.add_scalar('Perceptual Loss', second_PerceptualLoss.item(), batches_done)

                summary.add_scalar('psnr', utils.psnr(second_out, img), batches_done)
                summary.add_scalar('ssim', utils.ssim(second_out, img), batches_done)
            '''

        '''
        # Learning rate decrease
        adjust_learning_rate(opt.lr_g, optimizer_g, (epoch + 1), opt)
        adjust_learning_rate(opt.lr_d, optimizer_d, (epoch + 1), opt)
        '''

        # Save the model
        save_model_generator(generator, (epoch + 1), opt)
        save_model_discriminator(discriminator, (epoch + 1), opt)

        ### Sample data every epoch
        '''
        if (epoch + 1) % 1 == 0:
            img_list = [img, mask, masked_img, first_out, second_out]
            name_list = ['gt', 'mask', 'masked_img', 'first_out', 'second_out']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
        '''




