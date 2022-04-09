from food_classification import Classifier,train_set
import torch
import matplotlib.pyplot as plt

device='cuda:0' if torch.cuda.is_available() else 'cpu'
model=Classifier().to(device)
model.load_state_dict(torch.load('./ckpt.pth'))
'''
Saliency map
fed an image to model, after forward process we get the batch_loss,So related to loss are: 
image pixels, model parameter and label. 
Usually, we want to change the model parameters to fit image and label. 
Therefore, when loss calculates backward, we only care about the partial differential value of loss to model parameter. 
But mathematically, the image itself is also a continuous tensor,and we can calculate the partial differential value of 
the loss to the image pixels. This partial differential value represents "what changes will occur to the loss by 
slightly changing a certain pixel value of the image when the model parameter and label are fixed." People are used to 
interpreting the magnitude of this change as the importance of that pixel (each pixel has its own partial differential 
value). Therefore, by drawing the partial differential value of loss for each pixel in the same picture, you can see 
which positions in the picture are the important basis for the model to judge.
'''


# avoid a black image
def normalize(image):
    img_norm = (image - image.min()) / (image.max() - image.min())
    return img_norm


def saliency_map(x, y, model):
    model.eval()
    x = x.to(device)

    x.requires_grad_()  # add pixel information to do gradient
    y_pred = model(x)

    cri = torch.nn.CrossEntropyLoss()
    loss = cri(y_pred, y.to(device))
    loss.backward()

    saliency = x.grad.abs().detach().cpu()  # get gradient of x, saliencies dim: [batch,c,h,w]
    saliency = torch.stack([normalize(i) for i in saliency])
    # normalize img one by one, avoid an img with big value, another with small value
    return saliency


if __name__=='__main__':
    indices = [0,1,2,3]
    imgs, labels = train_set.getbatch(indices)
    saliencies = saliency_map(imgs, labels, model)
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for row, target in enumerate([imgs, saliencies]):  # 1st row: real img, 2nd row: saliency map
        for column, img in enumerate(target):
            # in pytorch,img_dim=[c=3,h=128,w=128].
            # in matplotlib, img_dim=[h=128,w=128,c=3],use permute to change dim
            # imshow only numpy.narray type
            axes[row][column].imshow(img.permute(1, 2, 0).numpy())
    plt.show()
    plt.close()