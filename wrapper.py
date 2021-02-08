import tensorflow as tf


#batch랑 mask랑 param 어케가져오는지
class MetaLearnerWrapper():
    '''
    [*] Wrapper class for MAML
    Need to produce MAML optimizer
    '''
    def __init__(self, input_in, input_meta, mask0, generator, discriminator, alpha=1e-4, beta=1e-4):
        '''
        inner loop input : input_in(initial_rs), mask
        outer loop input : gt, mask
        initial_rs=G(gt)는 일단 trainer에서 가져온다고 치고??
        a: data for inner update (Task batch size, data batch size, ...) => dimension ???
        b: data for meta update
        network: class, not the object
        '''
        self.input_in = input_in
        self.input_meta = input_meta
        self.mask0 = mask0

        # initialize network
        self.G = generator()
        self.D = discriminator()
        # inner loop 몇 번 돌릴건지??
        self.gradient_number = 5
        
        self.alpha = tf.get_variable(name='alpha', initializer=tf.constant(alpha), dtype=tf.float32, trainable=False)
        self.beta = tf.get_variable(name='beta', initializer=tf.constant(beta), dtype=tf.float32, trainable=False)
        self.stop_grad = False
        self.batch_size = 16
        self.grad_clip_range = 1.

    '''
    G
    input: batch_pos, mask(생성 or 불러온)
    output: x1, x2, 사용된 mask, batch_complete

    D
    input: real_in, batch_complete, 사용된 mask(model에서 batch_pos_neg으로 조합)
    output: pos_neg

    g_loss
    input: pos, neg, batch_pos, x1, x2

    d_loss
    input: pos_neg
    '''
    def d_loss_func(self, pos_neg):
        pos, neg = tf.split(pos_neg, 2)

        with tf.variable_scope('gan_hinge_loss'):
            hinge_pos = tf.reduce_mean(tf.nn.relu(1-pos))
            hinge_neg = tf.reduce_mean(tf.nn.relu(1+neg))
            d_loss = tf.add(.5 * hinge_pos, .5 * hinge_neg)

        return d_loss, neg  # neg for g_loss


    def g_loss_func(self, x1, x2, real, neg):
        # ae_loss
        ae_loss = self.l1_loss_alpha * tf.reduce_mean(tf.abs(real - x1))
        ae_loss += self.l1_loss_alpha * tf.reduce_mean(tf.abs(real - x2))
        g_loss = ae_loss + (-tf.reduce_mean(neg))
        return g_loss


    def build_maml(self):
        # define present parameter
        g_params = self.G.params
        d_params = self.D.params

        def task_metalearn(arg_in, reuse=True):
            input_in, input_meta, mask0 = arg_in
            task_outputs_meta, task_d_losses_meta, task_g_losses_meta = [], [], []

            # input data processing
            real_in = input_in / 127.5 - 1.
            real_meta = input_meta / 127.5 - 1.
            # present G result
            x1, x2, task_mask, fake = self.G(real_in, input_params=g_params)
            task_output_in = fake
            # training discriminator
            pos_neg = self.D(real_in, task_output_in, task_mask)
            task_d_loss_in, d_output_neg = self.d_loss_func(pos_neg)
            # 현재 파라미터값으로 loss를 gradient 계산하는 식(?)
            d_grads = tf.gradients(task_d_loss_in, list(d_params.values()))
            if(self.stop_grad): # stop backpropagation of grads
                d_grads = [tf.stop_gradient(d_grad) for d_grad in d_grads]
            # {현재 파라미터 이름 : 그 파라미터를 gradient 계산한 값}
            d_gradients = dict(zip(d_params.keys(), d_grads))
            # {현재 파라미터 이름 : 현재 파라미터에서 gradient계산한값 뺀 것}
            # => 이 task에 대해서 파라미터 갱신(inner loop에서 할 일 끝)
            d_fast_params = dict(zip(d_params.keys(),
                [d_params[key] - self.alpha*d_gradients[key] for key in d_params.keys()]))

            # training generator
            task_g_loss_in = self.g_loss_func(x1, x2, real_in, d_output_neg)
            g_grads = tf.gradients(task_g_loss_in, list(g_params.values()))
            if(self.stop_grad):
                g_grads = [tf.stop_gradient(g_grad) for g_grad in g_grads]           
            g_gradients = dict(zip(g_params.keys(), g_grads))
            g_fast_params = dict(zip(g_params.keys(),
                [g_params[key] - self.alpha*g_gradients[key] for key in g_params.keys()]))
                
            # meta output, loss
            # 정해진 mask; mask0으로 inpainting
            x1, x2, _, fake = self.G(real_meta, mask0, input_params=g_fast_params)
            meta_output = fake
            pos_neg = self.D(real_meta, meta_output, mask0)
            task_d_loss_meta, d_output_neg = self.d_loss_func(pos_neg)
            task_g_loss_meta = self.g_loss_func(x1, x2, real_meta, d_output_neg)
            
            task_outputs_meta.append(meta_output)
            task_d_losses_meta.append(task_d_loss_meta)
            task_g_losses_meta.append(task_g_loss_meta)
            
            for j in range(self.gradient_number-1):
                # model에 fast_param갖고 한 번 더 gradient 계산   
                x1, x2, task_mask, fake = self.G(real_in, input_params=g_fast_params)
                pos_neg = self.D(real_in, fake, task_mask)
                _task_d_loss_in, d_output_neg = self.d_loss_func(pos_neg)
                d_grads = tf.gradients(_task_d_loss_in, list(d_fast_params.values()))
                if(self.stop_grad): # stop backpropagation of grads
                    d_grads = [tf.stop_gradient(d_grad) for d_grad in d_grads]
                d_gradients = dict(zip(d_fast_params.keys(), d_grads))
                d_fast_params = dict(zip(d_fast_params.keys(),
                    [d_fast_params[key] - self.alpha*d_gradients[key] for key in d_fast_params.keys()]))
                    
                _task_g_loss_in = self.g_loss_func(x1, x2, real_in, d_output_neg)
                g_grads = tf.gradients(_task_g_loss_in, list(g_fast_params.values()))
                if(self.stop_grad):
                    g_grads = [tf.stop_gradient(g_grad) for g_grad in g_grads]
                g_gradients = dict(zip(g_fast_params.keys(), g_grads))
                g_fast_params = dict(zip(g_fast_params.keys(),
                    [g_fast_params[key] - self.alpha*g_gradients[key] for key in g_fast_params.keys()]))

                # meta output, loss
                x1, x2, _, fake = self.G(real_meta, mask0, input_params=g_fast_params)
                meta_output = fake
                pos_neg = self.D(real_meta, meta_output, mask0)
                task_d_loss_meta, d_output_neg = self.d_loss_func(pos_neg)
                task_g_loss_meta = self.g_loss_func(x1, x2, real_meta, d_output_neg)
                task_outputs_meta.append(meta_output)
                task_d_losses_meta.append(task_d_loss_meta)
                task_g_losses_meta.append(task_g_loss_meta)

            task_output = [task_output_in, task_outputs_meta, task_d_loss_in\
                , task_g_loss_in, task_d_losses_meta, task_g_losses_meta]

            return task_output

        out_dtype = [tf.float32, [tf.float32]*self.gradient_number, tf.float32, \
            tf.float32, [tf.float32]*self.gradient_number, [tf.float32]*self.gradient_number]
        result = tf.map_fn(task_metalearn, elems=(self.input_in, self.input_meta, self.mask0), \
            dtype=out_dtype, parallel_iterations=self.batch_size)
        outputs_in, outputs_meta, d_losses_in, g_losses_in, d_losses_meta, g_losses_meta = result
        self.outputs_in = outputs_in
        self.outputs_meta = outputs_meta
        
        # optimizer
        with tf.variable_scope('d_optimizer'):
            d_total_loss = d_losses_meta[-1]

            d_optimizer = tf.train.AdamOptimizer(self.beta)
            self.d_gvs = d_optimizer.compute_gradients(d_total_loss)
            self.d_gvs = [(tf.clip_by_value(grad, -self.grad_clip_range, self.grad_clip_range), var) \
                for grad, var in self.d_gvs]
            self.d_metatrain_op = d_optimizer.apply_gradients(self.d_gvs)
            self.d_losses_meta = d_losses_meta

            self.d_pretrain_op = tf.train.AdamOptimizer(self.beta).minimize(tf.reduce_mean(d_losses_in))
            # self.d_losses_in = self.d_loss_func(self.D(self.input_meta[:, 0, :, :, :]), self.gt_output_meta[:, 0, :, :, :])
            # 얘 왜 새로 계산하는건지 모르겠음

        with tf.variable_scope('g_optimizer'):
            g_total_loss = g_losses_meta[-1]    # 왜 -1?

            g_optimizer = tf.train.AdamOptimizer(self.beta)
            self.g_gvs = g_optimizer.compute_gradients(g_total_loss)    # var 추가해야하나 그런데 어떻게...
            self.g_gvs = [(tf.clip_by_value(grad, -self.grad_clip_range, self.grad_clip_range), var) \
                for grad, var in self.g_gvs]    # 이거 var_list는 trainable_variables에서 가져온다네...
            self.g_metatrain_op = g_optimizer.apply_gradients(self.g_gvs)
            self.g_losses_meta = g_losses_meta

            self.g_pretrain_op = tf.train.AdamOptimizer(self.beta).minimize(tf.reduce_mean(g_losses_in))
            # self.g_losses_in = self.g_loss_func(self.G(self.input_meta[:, 0, :, :, :]), self.gt_output_meta[:, 0, :, :, :])
