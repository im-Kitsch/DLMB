from torchvision.transforms import transforms
from dataloader.skin_lesion_dataset import SkinLesionDataset
from models.dcgan import Generator, Discriminator

import torch

def main():
   # potentially normalize
   dataset = SkinLesionDataset(transform=transforms.Compose([transforms.ToTensor()]))
   img, meta = dataset[0]
   print(meta)
   test = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)

   for t in test:
      x = 5
      break
   """
   imgs = torch.stack([img_t for img_t, _ in dataset], dim=3)
   m = imgs.view(3, -1).mean(dim=1)
   std = imgs.view(3, -1).std(dim=1)
   print(m)
   print(std)
   """

# move to other file maybe?
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    LEARNING_RATE = 3e-4
    NOISE_DIM = 64                                                            # dimension of input noise vector
    IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS = (600, 450, 3)                 # input image dimension
    BATCH_SIZE = 32
    EPOCHS = 100

    FEATURES_DISC = 64
    FEATURES_GEN = 64

    dataset = SkinLesionDataset(transform=transforms.Compose([transforms.ToTensor()]))
    data = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    disc = Discriminator(IMAGE_CHANNELS, FEATURES_DISC).to(device)
    gen = Generator(NOISE_DIM, IMAGE_CHANNELS, FEATURES_GEN).to(device)

    noise_vector = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)  # simple normal noise distribution

    # Init weights here

    disc.train()
    gen.train()

    # Training loop here
    for epoch in range(EPOCHS):
        print("Training nothing...")




if __name__ == '__main__':
    main()

