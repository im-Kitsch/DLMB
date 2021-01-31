from PIL import Image
import torchvision

img = Image.open('/home/yuan/Documents/datas/HAM10000/HAM10000_img/ISIC_0024684.jpg')

img2 = img.crop((75, 0, 600-75, 450))
img2 = img2.resize((256, 256))

img.show()
img2.show()

tr = torchvision.transforms.ToTensor()
img_tr = tr(img2)

