from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

from models.models import Network
from data.utils import get_train_dataset, get_test_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_one_hot(target,num_class):
    one_hot = torch.zeros(target.shape[0],num_class).to(device)
    one_hot = one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot


class iCaRLmodel:

    def __init__(self, args, task_classes, feature_extractor):
        super(iCaRLmodel, self).__init__()
        self.epochs = args["epochs"]
        self.learning_rate = args["lr"]
        self.batchsize = args["batchsize"]
        self.memory_size = args["memorysize"]
        
        self.task = 0 # Current task id
        self.task_classes = task_classes # List of sequential task sizes, e.g. [50,10,10,10,10]
        self.numclass = sum(self.task_classes[: self.task + 1]) # Classes seen so far
        self.task_size = self.task_classes[self.task] # Number of classes in current task

        self.exemplar_set = []
        self.class_mean_set = []

        # Data
        self.transform = transforms.Compose([#transforms.Resize(img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.classify_transform=transforms.Compose([transforms.RandomHorizontalFlip(p=1.), #TODO: Remove augmentation for classification 
                                                #transforms.Resize(img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.train_dataset = get_train_dataset(data_path="/opt/datasets")
        self.test_dataset = get_test_dataset(data_path="/opt/datasets")
        self.train_loader=None
        self.test_loader=None

        # Models
        self.model = Network(self.numclass, feature_extractor())
        self.old_model = None

    # get incremental train data
    # incremental
    def beforeTrain(self):
        self.model.eval()
        #classes=[self.numclass-self.task_size,self.numclass]
        classes = [self.numclass - self.task_size, self.numclass]
        self.train_loader,self.test_loader=self._get_train_and_test_dataloader(classes)
        if self.numclass>self.task_size:
            self.model.Incremental_learning(self.numclass)
        self.model.train()
        self.model.to(device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes, self.exemplar_set)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.batchsize)

        return train_loader, test_loader
    
    '''
    def _get_old_model_output(self, dataloader):
        x = {}
        for step, (indexs, imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                old_model_output = torch.sigmoid(self.old_model(imgs))
            for i in range(len(indexs)):
                x[indexs[i].item()] = old_model_output[i].cpu().numpy()
        return x
    '''

    # train model
    # compute loss
    # evaluate model
    def train(self):
        accuracy = 0
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)
        for epoch in range(self.epochs):
            if epoch == 48:
                if self.numclass==self.task_size:
                    opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 5, weight_decay=0.00001)
                else:
                    for p in opt.param_groups:
                        p['lr'] =self.learning_rate/ 5
                    #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 5,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                print(f"Change learning rate: {self.learning_rate / 5}")
            elif epoch == 62:
                if self.numclass>self.task_size:
                    for p in opt.param_groups:
                        p['lr'] =self.learning_rate/ 25
                    #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 25,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                else:
                    opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 25, weight_decay=0.00001)
                print(f"Change learning rate: {self.learning_rate / 25}")
            elif epoch == 80:
                if self.numclass==self.task_size:
                    opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125,weight_decay=0.00001)
                else:
                    for p in opt.param_groups:
                        p['lr'] =self.learning_rate/ 125
                    #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                print(f"Change learning rate: {self.learning_rate / 125}")
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(device), target.to(device)
                #output = self.model(images)
                loss_value = self._compute_loss(indexs, images, target)
                opt.zero_grad()
                loss_value.backward()
                opt.step()
                # print('epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss_value.item()))
            accuracy = self._test(self.test_loader, 1)
            print('epoch:%d,accuracy:%.3f' % (epoch, accuracy))
        return accuracy

    def _test(self, testloader, mode):
        if mode==0:
            print("compute NMS")
        self.model.eval()
        correct, total = 0, 0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = self.model(imgs) if mode == 1 else self.classify(imgs)
            predicts = torch.max(outputs, dim=1)[1] if mode == 1 else outputs
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct / total
        self.model.train()
        return accuracy

    def augment_with_memory(self, imgs, target):
        pass

    # TODO: Break loss into new_loss + old_loss
    # TODO: Train with simple replay
    def _compute_loss(self, indexs, imgs, target):
        output=self.model(imgs)
        target = get_one_hot(target, self.numclass)
        output, target = output.to(device), target.to(device)
        if self.old_model == None:
            return F.binary_cross_entropy_with_logits(output, target)
        else:
            #old_target = torch.tensor(np.array([self.old_model_output[index.item()] for index in indexs]))
            old_target=torch.sigmoid(self.old_model(imgs))
            old_task_size = old_target.shape[1]
            target[..., :old_task_size] = old_target
            return F.binary_cross_entropy_with_logits(output, target)

    # Separate distillation of base classes from learning new samples
    def _compute_loss_separate(self, indexs, imgs, target):
        output=self.model(imgs)
        target_onehot = get_one_hot(target, self.numclass)
        output, target_onehot = output.to(device), target_onehot.to(device)
        if self.old_model == None:
            return F.binary_cross_entropy_with_logits(output, target_onehot)
        else:
            # Find index of old task samples
            idx = (target < self.numclass - self.task_size).squeeze().nonzero().squeeze()
            old_imgs = imgs[idx]

            #old_target = torch.tensor(np.array([self.old_model_output[index.item()] for index in indexs]))
            old_target=torch.sigmoid(self.old_model(old_imgs))
            #old_task_size = old_target.shape[1]

            #target_onehot[..., :old_task_size] = old_target
            target_onehot[idx] = old_target # Replace base class labels with old predictions
            return F.binary_cross_entropy_with_logits(output, target_onehot)


    # change the size of examplar
    def afterTrain(self,accuracy):
        self.model.eval()
        m = int(self.memory_size / self.numclass)
        self._reduce_exemplar_sets(m)

        # for i in range(self.task + 1):
        #     print(f'Construct class {i} examplar:',end='')
        #     images = self.train_dataset.get_image_class(self.task_classes[i])
        #     self._construct_exemplar_set(images, m)

        for i in range(self.numclass - self.task_size, self.numclass):
            print(f'Construct class {i} examplar:',end='')
            images = self.train_dataset.get_image_class(i)
            self._construct_exemplar_set(images,m)

        # Update number of seen classes
        self.task += 1
        #self.numclass += self.task_size
        self.numclass = sum(self.task_classes[: self.task + 1]) # Classes seen so far
        self.task_size = self.task_classes[self.task] # Number of classes in current task
 

        self.compute_exemplar_class_mean()
        self.model.train()
        KNN_accuracy = self._test(self.test_loader,0)
        print("NMS accuracy："+str(KNN_accuracy.item()))
        filename = './snapshots/accuracy:%.3f_KNN_accuracy:%.3f_increment:%d_net.pkl' % (accuracy, KNN_accuracy, i + 10)
        torch.save(self.model,filename)
        self.old_model=torch.load(filename)
        self.old_model.to(device)
        self.old_model.eval()
        


    def _construct_exemplar_set(self, images, m):
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, 512))
     
        for i in range(m):
            # shape：batch_size*512
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])

        print("the size of exemplar :%s" % (str(len(exemplar))))
        self.exemplar_set.append(exemplar)
        #self.exemplar_set.append(images)

    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
            print('Size of class %d examplar: %s' % (index, str(len(self.exemplar_set[index]))))

    def Image_transform(self, images, transform):
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data

    def compute_class_mean(self, images, transform):
        x = self.Image_transform(images, transform).to(device)
        feature_extractor_output = F.normalize(self.model.feature_extractor(x).detach()).cpu().numpy()
        #feature_extractor_output = self.model.feature_extractor(x).detach().cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output

    def compute_exemplar_class_mean(self):
        self.class_mean_set = []
        for index in range(len(self.exemplar_set)):
            print("compute the class mean of %s"%(str(index)))
            exemplar = self.exemplar_set[index]
            #exemplar=self.train_dataset.get_image_class(index)
            class_mean, _  = self.compute_class_mean(exemplar, self.transform)
            class_mean_,_ = self.compute_class_mean(exemplar,self.classify_transform)
            class_mean = (class_mean/np.linalg.norm(class_mean)+class_mean_/np.linalg.norm(class_mean_))/2
            self.class_mean_set.append(class_mean)

    def classify(self, test):
        result = []
        test = F.normalize(self.model.feature_extractor(test).detach()).cpu().numpy()
        #test = self.model.feature_extractor(test).detach().cpu().numpy()
        class_mean_set = np.array(self.class_mean_set)
        for target in test:
            x = target - class_mean_set
            x = np.linalg.norm(x, ord=2, axis=1)
            x = np.argmin(x)
            result.append(x)
        return torch.tensor(result)