from keras.models import load_model
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

# load the model
model = load_model('my_model.h5')

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
# gets the weights of the model
weights = model.get_weights()

# Get the weights of the x layer
weights1 = weights[0]

# plot the histogram
plt.hist(weights1.ravel(), bins=100)
plt.show()




# x = np.random.rand(1, model.input_shape[1])

# use the model to make predictions
# predictions = model.predict(x)

# prediction results
# print(predictions)
