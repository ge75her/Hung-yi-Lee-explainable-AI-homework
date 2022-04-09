from food_classification import train_set,Classifier
import matplotlib.pyplot as plt
import torch
import numpy as np
from skimage.segmentation import slic,mark_boundaries
from lime import lime_image

'''
! pip install Lime
Local Interpretable Model-Agnostic Explanations To approximate a black-box model by a simple model locally
'''
device='cuda:0' if torch.cuda.is_available() else 'cpu'
model=Classifier().to(device)
def predict(input):
    #input: numpy array (batch,height,width,channels)
    model.eval()
    input=torch.FloatTensor(input).permute(0,3,1,2) #tensor [batch,c,h,w]
    output=model(input.to(device))
    return output.detach().cpu().numpy()
def segmentation(input):
    #skimage segmente img to 100 components
    return slic(input,n_segments=100,compactness=1,sigma=1)
indices=[0,1]
images,labels=train_set.getbatch(indices)
fig,axes=plt.subplots(1,2,figsize=(15,15))
np.random.seed(16)

for idx,(image,label) in enumerate(zip(images.permute(0,2,3,1).numpy(),labels)):
    x=image.astype(np.double)
    print(label.item())
    explainer=lime_image.LimeImageExplainer()
    explanation=explainer.explain_instance(image=x,classifier_fn=predict,segmentation_fn=segmentation,
                                          top_labels=11,hide_color=0,num_samples=1000)
    #classifier_fn define how img through model to prediction
    #segmentation_fn define how to segmente an img
    lime_img,mask=explanation.get_image_and_mask(label.item(),
                                                positive_only=False,
                                                hide_rest=False,
                                                num_features=11,
                                                min_weight=0.05)
    axes[idx].imshow(mark_boundaries(lime_img / 2 + 0.5,mask))
plt.show()
plt.close()