"""
A mean-squared loss VAE that allows the user to specify the architecture instead of having it hardcoded in. 
"""

from torch import nn
from torch.autograd import Variable

class VAE(nn.Module):
    """
    @ encoder: A nn.Sequential object that determines the architecture of encoder
    @ decoder: A nn.Sequential object that determines the architecture of decoder
    @ encoded_dims: An integer specifying dimensionality of encoded images
    """
    def __init__(self, encoder, decoder, encoded_dims):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mu_fc = nn.Linear(encoded_dims, encoded_dims)
        self.logvar_fc = nn.Linear(encoded_dims, encoded_dims)
        self.encoded_dims = encoded_dims
    def encode(self, x):
        f = self.encoder.forward(x)
        return self.mu_fc(f), self.logvar_fc(f)
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return self.decoder.forward(z)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
