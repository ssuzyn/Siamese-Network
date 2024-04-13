import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image

from src.siamese import SiameseNetwork
from src.siameseDataset import SiameseNetworkDataset

net = None

def load_model():
    global net  
    net = SiameseNetwork().cuda()
    net.load_state_dict(torch.load('model/test.pt'))
    net.eval()

def load_dataset():
    dataset_folder = dset.ImageFolder(root='data/faces/training/')
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=dataset_folder,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)
    return DataLoader(siamese_dataset, num_workers=6, batch_size=1, shuffle=False)

def preprocessing(img_path):
    preprocess = transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()])
    img = Image.open(img_path)
    img = img.convert("L")
    img = preprocess(img)
    return img.view([1,1,100,100])

def inference(img, dataloader):
    person = (None, 100)
    for i in range(len(dataloader)):
        data_label = dataloader.dataset.imageFolderDataset.imgs[i][0].split('/')[-2]
        data = preprocessing(dataloader.dataset.imageFolderDataset.imgs[i][0])

        output1,output2 = net(Variable(img).cuda(),Variable(data).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        if person[1] > euclidean_distance.item():
            person = (data_label, euclidean_distance.item())
        
        print(person[0] + " : " + str(person[1]))
        if euclidean_distance.item() < 1.0:
            print(data_label + " : " + str(euclidean_distance.item()))
            # 유클리드 거리가 이전에 발견된 가장 짧은 거리보다 작을 경우에 person 튜플을 업데이트
    return person

def run(img_path):
    load_model()
    dataloader = load_dataset()
    img = preprocessing(img_path)
    who, _ = inference(img, dataloader)
    return who
