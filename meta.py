import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
Tensor = torch.cuda.FloatTensor

from network import GatedGenerator, PatchDiscriminator



class Meta_for_validating(nn.Module):
    """
    Meta Learner

    [*] code from https://github.com/dragen1860/MAML-Pytorch/blob/master/meta.py
    """
    def __init__(self, args, generator, discriminator):
        super(Meta, self).__init__()
        
        self.args = args

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step

        self.G = generator
        self.D = discriminator
        self.meta_g_optim = optim.Adam(self.G.parameters(), lr=self.meta_lr)
        self.meta_d_optim = optim.Adam(self.D.parameters(), lr=self.meta_lr)


    def d_loss_func(self, fake_scalar, true_scalar, img):
        valid = Tensor(np.ones((img.shape[0], 1, img.shape[2]//32, img.shape[3]//32)))
        zero = Tensor(np.zeros((img.shape[0], 1, img.shape[2]//32, img.shape[3]//32)))

        loss_fake = -torch.mean(torch.min(zero, -valid-fake_scalar))
        loss_true = -torch.mean(torch.min(zero, -valid+true_scalar))
        d_loss = 0.5 * (loss_fake + loss_true)
        return d_loss


    def g_loss_func(self, x1, x2, fake_scalar, img):
        valid = Tensor(np.ones((img.shape[0], 1, img.shape[2]//32, img.shape[3]//32)))

        first_L1Loss = (x1 - img).abs().mean()
        second_L1Loss = (x2 - img).abs().mean()
        gan_loss = -torch.mean(fake_scalar)

        g_loss = self.args.lambda_l1 * first_L1Loss + self.args.lambda_l1 * second_L1Loss + \
                + self.args.lambda_gan * gan_loss
        return g_loss


    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, img_tasks, mask_tasks):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        img_tasks = img_tasks.unsqueeze(1)
        mask_tasks = mask_tasks.unsqueeze(1)
        task_num, setsz, c_, h, w = img_tasks.size()

        losses_g = [0 for _ in range(self.update_step + 1)]
        losses_d = [0 for _ in range(self.update_step + 1)]
        outputs = []

        for i in range(task_num):

            img = img_tasks[i]
            mask = mask_tasks[i]
            
            ## Discriminator update
            first_out, second_out = self.G(img, mask, vars=None)
            if(i == 0): outputs.append(second_out.detach())
            second_out_wholeimg = img * (1 - mask) + second_out * mask
            fake_scalar = self.D(second_out_wholeimg.detach(), mask)
            true_scalar = self.D(img, mask)
            d_loss = self.d_loss_func(fake_scalar, true_scalar, img)            
            d_grad = torch.autograd.grad(d_loss, self.D.parameters())
            d_fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(d_grad, self.D.parameters())))

            ## Generator update
            first_out, second_out = self.G(img, mask, vars=None)
            second_out_wholeimg = img * (1 - mask) + second_out * mask
            fake_scalar = self.D(second_out_wholeimg, mask)
            g_loss = self.g_loss_func(first_out, second_out, fake_scalar, img)
            g_grad = torch.autograd.grad(g_loss, self.G.parameters())
            g_fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(g_grad, self.G.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                first_out, second_out = self.G(img, mask)
                second_out_wholeimg = img * (1 - mask) + second_out * mask
                fake_scalar = self.D(second_out_wholeimg.detach(), mask)
                true_scalar = self.D(img, mask)
                d_loss = self.d_loss_func(fake_scalar, true_scalar, img)  
                losses_d[0] += d_loss

                first_out, second_out = self.G(img, mask)
                second_out_wholeimg = img * (1 - mask) + second_out * mask
                fake_scalar = self.D(second_out_wholeimg, mask)
                g_loss = self.g_loss_func(first_out, second_out, fake_scalar, img)
                losses_g[0] += g_loss


            # this is the loss and accuracy after the first update
            with torch.no_grad():
                first_out, second_out = self.G(img, mask, vars=g_fast_weights)
                second_out_wholeimg = img * (1 - mask) + second_out * mask
                fake_scalar = self.D(second_out_wholeimg.detach(), mask, vars=d_fast_weights)
                true_scalar = self.D(img, mask, vars=d_fast_weights)
                d_loss = self.d_loss_func(fake_scalar, true_scalar, img)  
                losses_d[0] += d_loss

                first_out, second_out = self.G(img, mask, vars=g_fast_weights)
                second_out_wholeimg = img * (1 - mask) + second_out * mask
                fake_scalar = self.D(second_out_wholeimg, mask, vars=d_fast_weights)
                g_loss = self.g_loss_func(first_out, second_out, fake_scalar, img)
                losses_g[0] += g_loss


            for k in range(self.update_step-1):
                first_out, second_out = self.G(img, mask, vars=g_fast_weights)
                if(i == 0): outputs.append(second_out.detach())
                second_out_wholeimg = img * (1 - mask) + second_out * mask
                fake_scalar = self.D(second_out_wholeimg.detach(), mask, vars=d_fast_weights)
                true_scalar = self.D(img, mask, vars=d_fast_weights)
                d_loss = self.d_loss_func(fake_scalar, true_scalar, img)  
                d_grad = torch.autograd.grad(d_loss, self.D.parameters())
                d_fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(d_grad, self.D.parameters())))
                losses_d[k+1] += d_loss

                first_out, second_out = self.G(img, mask, vars=g_fast_weights)
                second_out_wholeimg = img * (1 - mask) + second_out * mask
                fake_scalar = self.D(second_out_wholeimg, mask, vars=d_fast_weights)
                g_loss = self.g_loss_func(first_out, second_out, fake_scalar, img)
                g_grad = torch.autograd.grad(g_loss, self.G.parameters())
                g_fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(g_grad, self.G.parameters())))
                losses_g[k+1] += g_loss

            first_out, second_out = self.G(img, mask, vars=g_fast_weights)
            if(i == 0): outputs.append(second_out.detach())
            second_out_wholeimg = img * (1 - mask) + second_out * mask
            fake_scalar = self.D(second_out_wholeimg.detach(), mask, vars=d_fast_weights)
            true_scalar = self.D(img, mask, vars=d_fast_weights)
            d_loss = self.d_loss_func(fake_scalar, true_scalar, img)  
            losses_d[self.update_step] += d_loss

            first_out, second_out = self.G(img, mask, vars=g_fast_weights)
            second_out_wholeimg = img * (1 - mask) + second_out * mask
            fake_scalar = self.D(second_out_wholeimg, mask, vars=d_fast_weights)
            g_loss = self.g_loss_func(first_out, second_out, fake_scalar, img)
            losses_g[self.update_step] += g_loss

            
        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_d = losses_d[-1] / task_num
        loss_g = losses_g[-1] / task_num

        print(loss_d, loss_g)

        # optimize theta parameters
        self.meta_d_optim.zero_grad()
        loss_d.backward()
        self.meta_d_optim.step()

        self.meta_g_optim.zero_grad()
        loss_g.backward()
        self.meta_g_optim.step()

        return loss_d, loss_g, outputs




class Meta(nn.Module):
    """
    Meta Learner

    [*] code from https://github.com/dragen1860/MAML-Pytorch/blob/master/meta.py
    """
    def __init__(self, args, generator, discriminator):
        super(Meta, self).__init__()
        
        self.args = args

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step

        self.G = generator
        self.D = discriminator
        self.meta_g_optim = optim.Adam(self.G.parameters(), lr=self.meta_lr)
        self.meta_d_optim = optim.Adam(self.D.parameters(), lr=self.meta_lr)


    def d_loss_func(self, fake_scalar, true_scalar, img):
        valid = Tensor(np.ones((img.shape[0], 1, img.shape[2]//32, img.shape[3]//32)))
        zero = Tensor(np.zeros((img.shape[0], 1, img.shape[2]//32, img.shape[3]//32)))

        loss_fake = -torch.mean(torch.min(zero, -valid-fake_scalar))
        loss_true = -torch.mean(torch.min(zero, -valid+true_scalar))
        d_loss = 0.5 * (loss_fake + loss_true)
        return d_loss


    def g_loss_func(self, x1, x2, fake_scalar, img):
        valid = Tensor(np.ones((img.shape[0], 1, img.shape[2]//32, img.shape[3]//32)))

        first_L1Loss = (x1 - img).abs().mean()
        second_L1Loss = (x2 - img).abs().mean()
        gan_loss = -torch.mean(fake_scalar)

        g_loss = self.args.lambda_l1 * first_L1Loss + self.args.lambda_l1 * second_L1Loss + \
                + self.args.lambda_gan * gan_loss
        return g_loss


    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, img_tasks, mask_tasks):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        img_tasks = img_tasks.unsqueeze(1)
        mask_tasks = mask_tasks.unsqueeze(2)
        task_num, setsz, c_, h, w = img_tasks.size()

        losses_g = [0 for _ in range(self.update_step + 1)]
        losses_d = [0 for _ in range(self.update_step + 1)]
        outputs = []

        for i in range(task_num):

            img = img_tasks[i]
            mask = mask_tasks[i]
            
            ## Discriminator update
            first_out, second_out = self.G(img, mask[1], vars=None)
            second_out_wholeimg = img_proxy * (1 - mask[1]) + second_out * mask[1]
            img_proxy = second_out_wholeimg.detach()
            fake_scalar = self.D(second_out_wholeimg.detach(), mask[1])
            true_scalar = self.D(img_proxy, mask[1])
            d_loss = self.d_loss_func(fake_scalar, true_scalar, img)            
            d_grad = torch.autograd.grad(d_loss, self.D.parameters())
            d_fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(d_grad, self.D.parameters())))

            ## Generator update
            first_out, second_out = self.G(img, mask[1], vars=None)
            second_out_wholeimg = img_proxy * (1 - mask[1]) + second_out * mask[1]
            fake_scalar = self.D(second_out_wholeimg, mask[1])
            g_loss = self.g_loss_func(first_out, second_out, fake_scalar, img_proxy)
            g_grad = torch.autograd.grad(g_loss, self.G.parameters())
            g_fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(g_grad, self.G.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                first_out, second_out = self.G(img, mask[0])
                second_out_wholeimg = img * (1 - mask[0]) + second_out * mask[0]
                fake_scalar = self.D(second_out_wholeimg.detach(), mask[0])
                true_scalar = self.D(img, mask[0])
                d_loss = self.d_loss_func(fake_scalar, true_scalar, img)  
                losses_d[0] += d_loss

                first_out, second_out = self.G(img, mask[0])
                second_out_wholeimg = img * (1 - mask[0]) + second_out * mask[0]
                fake_scalar = self.D(second_out_wholeimg, mask[0])
                g_loss = self.g_loss_func(first_out, second_out, fake_scalar, img)
                losses_g[0] += g_loss

                if(i == 0): outputs.append(second_out_wholeimg.detach())

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                first_out, second_out = self.G(img, mask[0], vars=g_fast_weights)
                second_out_wholeimg = img * (1 - mask[0]) + second_out * mask[0]
                fake_scalar = self.D(second_out_wholeimg.detach(), mask[0], vars=d_fast_weights)
                true_scalar = self.D(img, mask[0], vars=d_fast_weights)
                d_loss = self.d_loss_func(fake_scalar, true_scalar, img)  
                losses_d[1] += d_loss

                first_out, second_out = self.G(img, mask[0], vars=g_fast_weights)
                second_out_wholeimg = img * (1 - mask[0]) + second_out * mask[0]
                fake_scalar = self.D(second_out_wholeimg, mask[0], vars=d_fast_weights)
                g_loss = self.g_loss_func(first_out, second_out, fake_scalar, img)
                losses_g[1] += g_loss

                if(i == 0): outputs.append(second_out_wholeimg.detach())


            for k in range(1, self.update_step):
                # Inner Loop losses
                first_out, second_out = self.G(img, mask[k+1], vars=g_fast_weights)
                second_out_wholeimg = img_proxy * (1 - mask[k+1]) + second_out * mask[k+1]
                fake_scalar = self.D(second_out_wholeimg.detach(), mask[k+1], vars=d_fast_weights)
                true_scalar = self.D(img_proxy, mask[k+1], vars=d_fast_weights)
                d_loss = self.d_loss_func(fake_scalar, true_scalar, img_proxy)  
                d_grad = torch.autograd.grad(d_loss, self.D.parameters())
                d_fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(d_grad, self.D.parameters())))

                first_out, second_out = self.G(img, mask[k+1], vars=g_fast_weights)
                second_out_wholeimg = img_proxy * (1 - mask[k+1]) + second_out * mask[k+1]
                fake_scalar = self.D(second_out_wholeimg, mask[k+1], vars=d_fast_weights)
                g_loss = self.g_loss_func(first_out, second_out, fake_scalar, img_proxy)
                g_grad = torch.autograd.grad(g_loss, self.G.parameters())
                g_fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(g_grad, self.G.parameters())))

                # Outer Loop losses
                first_out, second_out = self.G(img, mask[0], vars=g_fast_weights)
                second_out_wholeimg = img * (1 - mask[0]) + second_out * mask[0]
                fake_scalar = self.D(second_out_wholeimg.detach(), mask[0], vars=d_fast_weights)
                true_scalar = self.D(img, mask[0], vars=d_fast_weights)
                d_loss = self.d_loss_func(fake_scalar, true_scalar, img)  
                losses_d[k+1] += d_loss

                first_out, second_out = self.G(img, mask[0], vars=g_fast_weights)
                second_out_wholeimg = img * (1 - mask[0]) + second_out * mask[0]
                fake_scalar = self.D(second_out_wholeimg, mask[0], vars=d_fast_weights)
                g_loss = self.g_loss_func(first_out, second_out, fake_scalar, img)
                losses_g[k+1] += g_loss

                if(i == 0): outputs.append(second_out_wholeimg.detach())

            
        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_d = losses_d[-1] / task_num
        loss_g = losses_g[-1] / task_num

        print(loss_d, loss_g)

        # optimize theta parameters
        self.meta_d_optim.zero_grad()
        loss_d.backward()
        self.meta_d_optim.step()

        self.meta_g_optim.zero_grad()
        loss_g.backward()
        self.meta_g_optim.step()

        return loss_d, loss_g, outputs











def main():
    pass


if __name__ == '__main__':
    main()
