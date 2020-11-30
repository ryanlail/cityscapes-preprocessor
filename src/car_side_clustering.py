import torch
from PIL import Image
from torchvision import transforms
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)

#filename = "../../rois/1.png"


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

#seed = 10
#random.seed(seed)
#torch.manual_seed(seed)
#np.random.seed(seed)

features = np.empty([1000,1000])

for batch in range(1, 1000):
    print(batch)
    filename = "../../rois/" + str(batch) + ".png"
    feature_vector = generate_feature_vector(filename)
    features[batch] = feature_vector.cpu().numpy()
#features_np = np.array(features)

tsne = TSNE(n_components=2).fit_transform(features)
print(tsne)

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



plt.scatter(tx, ty)
plt.title('TSNE')
plt.xlabel('x')
plt.ylabel('y')
plt.show()



"""
# initialize a matplotlib plot
fig = plt.figure()
ax = fig.add_subplot(111)

# for every class, we'll add a scatter plot separately
for label in colors_per_class:
    # find the samples of the current class in the data
    indices = [i for i, l in enumerate(labels) if l == label]

    # extract the coordinates of the points of this class only
    current_tx = np.take(tx, indices)
    current_ty = np.take(ty, indices)

    # convert the class color to matplotlib format
    color = np.array(colors_per_class[label], dtype=np.float) / 255

    # add a scatter plot with the corresponding color and label
    ax.scatter(current_tx, current_ty, c=color, label=label)

# build a legend using the labels we set previously
ax.legend(loc='best')

# finally, show the plot
plt.show()
"""
