import torch
from food_classification import train_set,Classifier
import matplotlib.pyplot as plt
'''
Use Gradient Ascent method to find the image that activates the selected filter the most and plot them
(start from white noise).
'''
device='cuda:0' if torch.cuda.is_available() else 'cpu'
model=Classifier().to(device)
def normalize(image):
    return (image-image.min())/(image.max()-image.min())
layer_activations=None
def filter_explanation(x,model,cnnid,filterid,iteration=100,lr=1):
    #x: img
    model.eval()
    def hook(model,input,output):
        global layer_activations
        layer_activations=output
    hook_handle=model.cnn[cnnid].register_forward_hook(hook)
    #This line is telling pytorch that when forward "passes" the cnn of the cnnid layer,
    #it is necessary to call hook, before continuing to forward the next layer of cnn
    model(x.to(device))
    filter_activations=layer_activations[:,filterid,:,:].detach().cpu() #x经过被指定filter的activationmap
    x=x.to(device)
    x.requires_grad_()
    optimizer=torch.optim.Adam([x],lr=lr)
    for i in range(iteration):
        optimizer.zero_grad()
        model(x)
        objective=-layer_activations[:,filterid,:,:].sum()
        objective.backward()
        optimizer.step()
    filter_visualization=x.detach().cpu().squeeze()[0] #找到可以最大限度Activate这个filter的图片
    hook_handle.remove()
    return filter_activations,filter_visualization


if __name__=='__main__':
    indices = [0, 1, 2, 3]
    images, labels = train_set.getbatch(indices)
    # 1st layer
    filter_activation1, filter_visualization1 = filter_explanation(images, model, cnnid=0, filterid=30, iteration=100,
                                                                   lr=0.1)
    # 2nd layer
    filter_activation4, filter_visualization4 = filter_explanation(images, model, cnnid=4, filterid=30, iteration=100,
                                                                   lr=0.1)
    # 3rd layer
    filter_activation8, filter_visualization8 = filter_explanation(images, model, cnnid=8, filterid=30, iteration=100,
                                                                   lr=0.1)
    # 4th layer
    filter_activation12, filter_visualization12 = filter_explanation(images, model, cnnid=12, filterid=30,
                                                                     iteration=100, lr=0.1)
    # plot activation maps for each layer
    fig, axes = plt.subplots(5, 4, figsize=(10, 8))
    for i in range(4):
        axes[0][i].imshow(images[i].permute(1, 2, 0))
        axes[1][i].imshow(normalize(filter_activation1[i]))
        axes[2][i].imshow(normalize(filter_activation4[i]))
        axes[3][i].imshow(normalize(filter_activation8[i]))
        axes[4][i].imshow(normalize(filter_activation12[i]))

    plt.show()
    plt.close()

    # plot visualization map, they are same
    fig, axes = plt.subplots(1, 4, figsize=(10, 8))
    for i in range(4):
        axes[i].imshow(normalize(filter_visualization1.permute(1, 2, 0)))
        axes[i].imshow(normalize(filter_visualization4.permute(1, 2, 0)))
        axes[i].imshow(normalize(filter_visualization8.permute(1, 2, 0)))
        axes[i].imshow(normalize(filter_visualization12.permute(1, 2, 0)))
    plt.show()
    plt.close()