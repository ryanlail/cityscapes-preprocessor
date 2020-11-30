import torch
from PIL import Image
from torchvision import transforms
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
from sklearn.cluster import KMeans

model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)

########################################################### generate feature vector
def generate_feature_vector(filename):
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    #print(output[0].shape)
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    return (torch.nn.functional.softmax(output[0], dim=0))


####################################################### TSNE

paths = []
features = np.empty([2000,1000])

for batch in range(1, 2000):
    filename = "../../rois2/" + str(batch) + ".png"
    feature_vector = generate_feature_vector(filename)
    features[batch] = feature_vector.cpu().numpy()
    paths.append(filename)
    print(filename)

tsne = TSNE(n_components=2).fit_transform(features)

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
 
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range
 
# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]
 
tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

########################### K-Means
kmeans = KMeans(n_clusters=3, random_state=0).fit(np.column_stack((tx,ty)))

print(kmeans.labels_)


def getImage(path):
    return OffsetImage(cv2.resize(plt.imread(path), (50, 50)))

fig, ax = plt.subplots()
ax.scatter(tx, ty) 

for x0, y0, path, label in zip(tx, ty,paths, kmeans.labels_):
    if label == 0:
        colour = "red"
    elif label == 1:
        colour = "blue"
    else:
        colour = "green"
    ab = AnnotationBbox(getImage(path), (x0, y0), bboxprops = dict(edgecolor=colour))
    ax.add_artist(ab)

plt.show()


