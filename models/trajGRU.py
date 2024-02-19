import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

IN_LEN = 10
OUT_LEN= 5
STRIDE = IN_LEN



class activation():
    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError


def warp(input, flow):
    # gamma function
    B, C, H, W = input.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)#.to(cfg.GLOBAL.DEVICE)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)#.to(cfg.GLOBAL.DEVICE)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(input, vgrid)
    return output

class BaseConvRNN(nn.Module):
    def __init__(self, num_filter, b_h_w,
                 h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                 i2h_kernel=(3, 3), i2h_stride=(1, 1),
                 i2h_pad=(1, 1), i2h_dilate=(1, 1),
                 act_type=torch.tanh,
                 prefix='BaseConvRNN'):
        super(BaseConvRNN, self).__init__()
        self._prefix = prefix
        self._num_filter = num_filter
        self._h2h_kernel = h2h_kernel
        assert (self._h2h_kernel[0] % 2 == 1) and (self._h2h_kernel[1] % 2 == 1), \
            "Only support odd number, get h2h_kernel= %s" % str(h2h_kernel)
        self._h2h_pad = (h2h_dilate[0] * (h2h_kernel[0] - 1) // 2,
                         h2h_dilate[1] * (h2h_kernel[1] - 1) // 2)
        self._h2h_dilate = h2h_dilate
        self._i2h_kernel = i2h_kernel
        self._i2h_stride = i2h_stride
        self._i2h_pad = i2h_pad
        self._i2h_dilate = i2h_dilate
        self._act_type = act_type
        assert len(b_h_w) == 3
        i2h_dilate_ksize_h = 1 + (self._i2h_kernel[0] - 1) * self._i2h_dilate[0]
        i2h_dilate_ksize_w = 1 + (self._i2h_kernel[1] - 1) * self._i2h_dilate[1]
        self._batch_size, self._height, self._width = b_h_w
        self._state_height = (self._height + 2 * self._i2h_pad[0] - i2h_dilate_ksize_h)\
                             // self._i2h_stride[0] + 1
        self._state_width = (self._width + 2 * self._i2h_pad[1] - i2h_dilate_ksize_w) \
                             // self._i2h_stride[1] + 1
        self._curr_states = None
        self._counter = 0

class TrajGRU(BaseConvRNN):
    # b_h_w: input feature map size
    def __init__(self, input_channel, num_filter, b_h_w, zoneout=0.0, L=5,
                 i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                 h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                 act_type=activation('leaky', negative_slope=0.2, inplace=True)):
        super(TrajGRU, self).__init__(num_filter=num_filter,
                                      b_h_w=b_h_w,
                                      h2h_kernel=h2h_kernel,
                                      h2h_dilate=h2h_dilate,
                                      i2h_kernel=i2h_kernel,
                                      i2h_pad=i2h_pad,
                                      i2h_stride=i2h_stride,
                                      act_type=act_type,
                                      prefix='TrajGRU')
        self._L = L
        self._zoneout = zoneout

        # wxz, wxr, wxh
        # reset_gate, update_gate, new_mem
        self.i2h = nn.Conv2d(in_channels=input_channel,
                            out_channels=self._num_filter*3,
                            kernel_size=self._i2h_kernel,
                            stride=self._i2h_stride,
                            padding=self._i2h_pad,
                            dilation=self._i2h_dilate)

        # inputs to flow
        self.i2f_conv1 = nn.Conv2d(in_channels=input_channel,
                                out_channels=32,
                                kernel_size=(5, 5),
                                stride=1,
                                padding=(2, 2),
                                dilation=(1, 1))

        # hidden to flow
        self.h2f_conv1 = nn.Conv2d(in_channels=self._num_filter,
                                   out_channels=32,
                                   kernel_size=(5, 5),
                                   stride=1,
                                   padding=(2, 2),
                                   dilation=(1, 1))

        # generate flow
        self.flows_conv = nn.Conv2d(in_channels=32,
                                   out_channels=self._L * 2,
                                   kernel_size=(5, 5),
                                   stride=1,
                                   padding=(2, 2))


        # hh, hz, hr
        self.ret = nn.Conv2d(in_channels=self._num_filter*self._L,
                                   out_channels=self._num_filter*3,
                                   kernel_size=(1, 1),
                                   stride=1)



    # inputs: B*C*H*W
    def _flow_generator(self, inputs, states):
        if inputs is not None:
            i2f_conv1 = self.i2f_conv1(inputs)
        else:
            i2f_conv1 = None
        h2f_conv1 = self.h2f_conv1(states)
        f_conv1 = i2f_conv1 + h2f_conv1 if i2f_conv1 is not None else h2f_conv1
        f_conv1 = self._act_type(f_conv1)

        flows = self.flows_conv(f_conv1)
        flows = torch.split(flows, 2, dim=1)
        return flows

    # inputs & states
    # inputs: S*B*C*H*W
    def forward(self, inputs=None, states=None, seq_len=IN_LEN):
        if states is None:
            states = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                                  self._state_width), dtype=torch.float)#.to(cfg.GLOBAL.DEVICE)
        if inputs is not None:
            S, B, C, H, W = inputs.size()
            i2h = self.i2h(torch.reshape(inputs, (-1, C, H, W)))
            i2h = torch.reshape(i2h, (S, B, i2h.size(1), i2h.size(2), i2h.size(3)))
            i2h_slice = torch.split(i2h, self._num_filter, dim=2)

        else:
            i2h_slice = None

        prev_h = states
        outputs = []
        for i in range(seq_len):
            if inputs is not None:
                flows = self._flow_generator(inputs[i, ...], prev_h)
            else:
                flows = self._flow_generator(None, prev_h)
            warpped_data = []
            for j in range(len(flows)):
                flow = flows[j]
                warpped_data.append(warp(prev_h, -flow))
            warpped_data = torch.cat(warpped_data, dim=1)
            h2h = self.ret(warpped_data)
            h2h_slice = torch.split(h2h, self._num_filter, dim=1)
            if i2h_slice is not None:
                reset_gate = torch.sigmoid(i2h_slice[0][i, ...] + h2h_slice[0])
                update_gate = torch.sigmoid(i2h_slice[1][i, ...] + h2h_slice[1])
                new_mem = self._act_type(i2h_slice[2][i, ...] + reset_gate * h2h_slice[2])
            else:
                reset_gate = torch.sigmoid(h2h_slice[0])
                update_gate = torch.sigmoid(h2h_slice[1])
                new_mem = self._act_type(reset_gate * h2h_slice[2])
            next_h = update_gate * prev_h + (1 - update_gate) * new_mem
            if self._zoneout > 0.0:
                mask = F.dropout2d(torch.zeros_like(prev_h), p=self._zoneout)
                next_h = torch.where(mask, next_h, prev_h)
            outputs.append(next_h)
            prev_h = next_h

        # return torch.cat(outputs), next_h
        return torch.stack(outputs), next_h
    
def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))

class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets)==len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, 'stage'+str(index), make_layers(params))
            setattr(self, 'rnn'+str(index), rnn)

    def forward_by_stage(self, input, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))
        # print("after conv:", input.size())
        outputs_stage, state_stage = rnn(input, None)
        # print("after rnn:", outputs_stage.size(), state_stage.size())
        
        return outputs_stage, state_stage

    # input: 5D S*B*I*H*W
    def forward(self, input):
        hidden_states = []
        for i in range(1, self.blocks+1):
            input, state_stage = self.forward_by_stage(input, getattr(self, 'stage'+str(i)), getattr(self, 'rnn'+str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)

class Forecaster(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks-index), rnn)
            setattr(self, 'stage' + str(self.blocks-index), make_layers(params))

    def forward_by_stage(self, input, state, subnet, rnn):
        #print("before rnn:", "---", state.size())
        
        input, state_stage = rnn(input, state, seq_len=OUT_LEN)
        #print("after rnn:", input.size(), "---", state_stage.size())
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        
        #print("before conv:", input.size())
        input = subnet(input)
        #print("after conv:", input.size())
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))

        return input

        # input: 5D S*B*I*H*W

    def forward(self, hidden_states):
        #                                                       change to stage_i & rnn_i
        #                                                         with i=num of stage
        input = self.forward_by_stage(None, hidden_states[-1], getattr(self, 'stage3'),
                                      getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            input = self.forward_by_stage(input, hidden_states[i-1], getattr(self, 'stage' + str(i)),
                                                       getattr(self, 'rnn' + str(i)))
        return input
    
class EF(nn.Module):
    def __init__(self, encoder, forecaster):
        super().__init__()
        self.encoder = encoder
        self.forecaster = forecaster

    def forward(self, input):
        state = self.encoder(input)
        output = self.forecaster(state)
        return output