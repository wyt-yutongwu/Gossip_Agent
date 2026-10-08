import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np



class DQN_Module(nn.Module):

    def __init__(self, env, conv_filters, fc_filters, action_shape,img_shape,use_cnn = False, channels_last=False, verbose=False,):
        super(DQN_Module, self).__init__()
        self.conv_filters = conv_filters
        self.fc_filters = fc_filters
        self.img_shape = img_shape
        self.action_shape = action_shape
        self.use_cnn = use_cnn
        self.channels_last = channels_last
        if self.use_cnn:
            self._build_cnn_layers()
        else:
            self.conv_layers = nn.Identity()
            self.cnn_out_size = (
                int(np.prod(self.img_shape)) if self.img_shape is not None else 0
            )
        if verbose:
            print(f"[DQN_Module] img_shape={img_shape}"
                  f"action_shape={action_shape} use_cnn={use_cnn}")
        self.flatten = nn.Flatten()

        self._build_fc_layers()
        self.output_layer = nn.Linear(self.fc_out_size, self.action_shape)
        if self.use_cnn and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        if self.channels_last and self.use_cnn:
            self.to(memory_format=torch.channels_last)
 
        # if compile_model:
        #     try:
        #         self.forward = torch.compile(self.forward, mode="reduce-overhead")
        #     except Exception as e:
        #         if verbose:
        #             print(f"[DQN_Module] torch.compile unavailable: {e}")

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, image):
        if self.use_cnn:
            # NHWC -> NCHW. SPEEDUP 5: .contiguous() so cuDNN's fast path is hit.
            mem_fmt = (
                torch.channels_last if self.channels_last else torch.contiguous_format
            )
            x = image.permute(0, 3, 1, 2).contiguous(memory_format=mem_fmt)
            x = self.conv_layers(x)
            x = self.flatten(x)
        else:
            x = self.flatten(image)
        x = self.fc_layers(x)
        return self.output_layer(x)
    
    @staticmethod
    def _conv_out_dim(in_size, kernel, stride, padding=0):
        return (in_size + 2 * padding - kernel) // stride + 1
        

    def _build_cnn_layers(self):
        H, W, C = self.img_shape
        layers = []
        in_chan = C
        h, w = H, W
        for out_chan, kernel, stride in self.conv_filters:
            kh, kw = kernel
            layers.append(
                nn.Conv2d(
                    in_channels=in_chan,
                    out_channels=out_chan,
                    kernel_size=(kh, kw),
                    stride=stride,
                    padding=0,
                )
            )
                    # SPEEDUP 3: in-place ReLU (no extra activation tensor).
            layers.append(nn.ReLU(inplace=True))
            h = self._conv_out_dim(h, kh, stride)
            w = self._conv_out_dim(w, kw, stride)
            in_chan = out_chan
        self.conv_layers = nn.Sequential(*layers)
        self.cnn_out_size = in_chan * h * w
    
    def _build_fc_layers(self):
        layers = []
        prev_in = self.cnn_out_size
        for size in self.fc_filters:
            layers.append(nn.Linear(prev_in, size))
            layers.append(nn.ReLU(inplace=True))   # SPEEDUP 3
            prev_in = size
        self.fc_layers = nn.Sequential(*layers)
        self.fc_out_size = prev_in
 