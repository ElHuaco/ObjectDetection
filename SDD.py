class SSD(nn.Module):
    def __init__(self, base='vgg'):
        super(SSD, self).__init__()
        if base == 'vgg':
            self.base_network = VGG()
        elif base == 'inception':
            self.base_network = Inception()
        else:
            raise ValueError('SSD base network')
        #self.feature_maps =
        pass

    def _init_weights(self, module):
        pass

    def forward(self, x):
        x_base = self.base_network(x)
        # output de cada feature_map
        # flatten y cat en dim 1
        pass
        