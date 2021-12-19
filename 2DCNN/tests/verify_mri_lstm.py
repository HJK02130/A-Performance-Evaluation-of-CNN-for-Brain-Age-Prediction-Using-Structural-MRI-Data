import torch
from torch import nn

"""
Code to test LSTM implementation with Lam et.al. 
Our implementation use vectorization and should be faster... but need to be verified.  
"""


def encoder_blk(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1),
        nn.InstanceNorm2d(out_channels),
        nn.MaxPool2d(2, stride=2),
        nn.ReLU()
    )


class MRI_LSTM(nn.Module):

    def __init__(self, lstm_feat_dim, lstm_latent_dim, *args, **kwargs):
        super(MRI_LSTM, self).__init__()

        self.input_dim = (1, 109, 91)

        self.feat_embed_dim = lstm_feat_dim
        self.latent_dim = lstm_latent_dim

        # Build Encoder
        encoder_blocks = [
                encoder_blk(1, 32),
                encoder_blk(32, 64),
                encoder_blk(64, 128),
                encoder_blk(128, 256),
                encoder_blk(256, 256)
        ]
        self.encoder = nn.Sequential(*encoder_blocks)

        # Post processing
        self.post_proc = nn.Sequential(
            nn.Conv2d(256, 64, 1, stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d([3, 2]),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, self.feat_embed_dim, 1)
        )

        # Connect w/ LSTM
        self.n_layers = 1
        self.lstm = nn.LSTM(self.feat_embed_dim, self.latent_dim, self.n_layers, batch_first=True)

        # Build regressor
        self.lstm_post = nn.Linear(self.latent_dim, 64)
        self.regressor = nn.Sequential(nn.ReLU(), nn.Linear(64, 1))

        self.init_weights()

    def init_weights(self):
        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and "regressor" in k:
                m.bias.data.fill_(62.68)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def init_hidden(self, x):
        h_0 = torch.zeros(self.n_layers, x.size(0), self.latent_dim, device=x.device)
        c_0 = torch.zeros(self.n_layers, x.size(0), self.latent_dim, device=x.device)
        h_0.requires_grad = True
        c_0.requires_grad = True
        return h_0, c_0

    def encode_old(self, x, ):

        B, C, H, W, D = x.size()
        h_t, c_t = self.init_hidden(x)
        for i in range(H):
            out = self.encoder(x[:, :, i, :, :])
            out = self.post_proc(out)
            out = out.view(B, 1, self.feat_embed_dim)
            h_t = h_t.view(1, B, self.latent_dim)
            c_t = c_t.view(1, B, self.latent_dim)
            h_t, (_, c_t) = self.lstm(out, (h_t, c_t))
        encoding = h_t.view(B, self.latent_dim)
        return encoding

    def encode_new(self, x):

        h_0, c_0 = self.init_hidden(x)
        B, C, H, W, D = x.size()
        # convert to 2D images, apply encoder and then reshape for lstm
        new_input = torch.cat([x[:, :, i, :, :] for i in range(H)], dim=0)
        encoding = self.encoder(new_input)
        encoding = self.post_proc(encoding)
        # (BxH) X C_out X W_out X D_out
        encoding = torch.stack(torch.split(encoding, B, dim=0), dim=2)
        # B X C_out X H X W_out X D_out
        encoding = encoding.squeeze(4).squeeze(3)
        # lstm take  batch x seq_len x dim
        encoding = encoding.permute(0, 2, 1)

        _, (encoding, _) = self.lstm(encoding)
        # output is 1 X batch x hidden
        encoding = encoding.squeeze(0)
        # pass it to lstm and get encoding
        return encoding

    def forward(self, x):
        embedding_old = self.encode_old(x)
        embedding_new = self.encode_new(x)

        return embedding_new, embedding_old


if __name__ == "__main__":
    B = 4
    new_model = MRI_LSTM(lstm_feat_dim=2, lstm_latent_dim=128)
    new_model.eval()
    inp = torch.rand(4, 1, 91, 109, 91)
    output = new_model(inp)
    print(torch.allclose(output[0], output[1]))
    # breakpoint()
